from pathlib import Path
import pickle
from utils import log_stage, parse_args
import numpy as np
import pandas as pd

from gmm_gpu import (
    fit_best_gmm_gpu_dp,
    prepare_boxcox_features_gpu,
    invert_latent_samples_gpu,
    sample_with_max_cap,
)

FEATURE_COLS = [
    "st_files",
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_bytes_xfered",
    "st_faults",
    "st_files_skipped",
    "st_xfer_time_ms",
    "encrypt_data",
]

COUNT_LIKE_COLS = [
    "st_files",
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_faults",
    "st_files_skipped"
]

BOOL_LIKE_COLS = ["encrypt_data"]

# shrinking the range to only the cluster counts I "realistically" need
# DEFAULT_COMPONENT_GRID = [12, 18, 24, 30, 36] 
DEFAULT_COMPONENT_GRID = [56] 
BOXCOX_SHIFT = 1.0
DEFAULT_MAX_ITER = 400
DEFAULT_N_INIT = 4
DEFAULT_TOL = 1e-3
DEFAULT_REG_COVAR = 5e-3
DEFAULT_KMEANS_ITERS = 10
DEFAULT_USE_KMEANS_INIT = True

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    cache_path = csv_path.with_suffix(".parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    df_raw = pd.read_csv(csv_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(cache_path)
    return df_raw

def apply_count_constraints(synthetic_features: pd.DataFrame):
    present_counts = [col for col in COUNT_LIKE_COLS if col in synthetic_features.columns]
    if present_counts:
        rounded = synthetic_features[present_counts].round().clip(lower=0)
        synthetic_features[present_counts] = rounded.astype(int)

    if {"st_dirs", "st_files"}.issubset(synthetic_features.columns):
        dirs = synthetic_features["st_dirs"].astype(int)
        files = synthetic_features["st_files"].astype(int)
        synthetic_features["st_dirs"] = np.minimum(dirs, files).astype(int)

def apply_constant_columns(synthetic_features: pd.DataFrame, constant_cols, constant_values):
    for col in constant_cols:
        synthetic_features[col] = constant_values[col]

def apply_bool_constraints(synthetic_features: pd.DataFrame):
    present = [col for col in BOOL_LIKE_COLS if col in synthetic_features.columns]
    if not present:
        return
    clipped = synthetic_features[present].astype(float).clip(lower=0, upper=1)
    synthetic_features[present] = clipped.ge(0.5)

def build_dataset_metadata(df: pd.DataFrame, feature_names: list[str]) -> dict:
    feature_mins = df[feature_names].min()
    feature_maxs = df[feature_names].max()
    cap_values = feature_maxs.copy()
    for col in BOOL_LIKE_COLS:
        if col in cap_values.index:
            cap_values[col] = np.inf
    return {
        "row_count": len(df),
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs,
        "cap_values": cap_values,
        "column_order": list(df.columns),
    }

def assemble_dataset(synthetic_features: pd.DataFrame, dataset_info: dict):
    feature_mins = dataset_info.get("feature_mins")
    feature_maxs = dataset_info.get("feature_maxs")
    column_order = dataset_info.get("column_order") or synthetic_features.columns.tolist()
    synthetic_dataset = synthetic_features.copy()
    if feature_mins is not None and feature_maxs is not None:
        lower = feature_mins.reindex(synthetic_dataset.columns)
        upper = feature_maxs.reindex(synthetic_dataset.columns)
        synthetic_dataset = synthetic_dataset.clip(lower=lower, upper=upper, axis=1)
    if "st_files_skipped" in synthetic_dataset.columns:
        synthetic_dataset["sync"] = (synthetic_dataset["st_files_skipped"] > 0).astype(int)
    ordered_columns = [col for col in column_order if col in synthetic_dataset.columns]
    for col in synthetic_dataset.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    return synthetic_dataset[ordered_columns]

def save_gmm_artifacts(path: Path, artifacts: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifacts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_gmm_artifacts(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)

def generate_synthetic_dataset(
    output_path: Path,
    data_path: Path | None = None,
    component_grid=DEFAULT_COMPONENT_GRID,
    gpu_kwargs: dict | None = None,
    gpu_max_cap: bool = False,
    seed: int = 42,
    artifact_path: Path | None = None,
):
    gpu_kwargs = gpu_kwargs or {}
    gpu_device = gpu_kwargs.get("device", "cuda")
    artifact_path_provided = artifact_path is not None
    if artifact_path is None:
        artifact_path = output_path.with_suffix(".gmm.pkl")
    df = None
    dataset_info = None
    if artifact_path.exists():
        log_stage(f"Loading fitted GMM artifacts from {artifact_path}")
        artifacts = load_gmm_artifacts(artifact_path)
        best_gmm = artifacts["gmm"]
        bic_df = artifacts.get("bic_df")
        transformer_gpu = artifacts["transformer"]
        feature_names = artifacts["feature_names"]
        constant_cols = artifacts["constant_cols"]
        constant_values = artifacts["constant_values"]
        boxcox_shift = artifacts.get("boxcox_shift", BOXCOX_SHIFT)
        dataset_info = artifacts.get("dataset_info")
        print("Loaded components:", best_gmm.n_components)
    elif artifact_path_provided:
        raise FileNotFoundError(f"GMM pickle not found: {artifact_path}")
    else:
        log_stage(f"Loading data from {data_path}")
        df = load_dataframe(data_path)
        df = df[df['grp_delete'] != True]
        df = df[df['st_files'] != 0]
        log_stage("Preparing Box-Cox features on GPU")
        (
            X_boxcox_gpu,
            transformer_gpu,
            constant_cols,
            constant_values,
            feature_names,
        ) = prepare_boxcox_features_gpu(df, FEATURE_COLS, BOXCOX_SHIFT, device=gpu_device)
        log_stage("Training GMM on GPU")
        gpu_args = gpu_kwargs | {"dtype": transformer_gpu.dtype, "random_state": seed}
        best_gmm, bic_df = fit_best_gmm_gpu_dp(
            X_boxcox_gpu,
            component_grid=component_grid,
            **gpu_args,
        )

        # if not bic_df.empty:
        #     print("BIC scores:\n", bic_df)
        print("Selected components:", best_gmm.n_components)
        boxcox_shift = BOXCOX_SHIFT
        dataset_info = build_dataset_metadata(df, feature_names)
        log_stage(f"Saving fitted GMM artifacts to {artifact_path}")
        save_gmm_artifacts(
            artifact_path,
            {
                "gmm": best_gmm,
                "bic_df": bic_df,
                "transformer": transformer_gpu,
                "feature_names": feature_names,
                "constant_cols": constant_cols,
                "constant_values": constant_values,
                "boxcox_shift": BOXCOX_SHIFT,
                "dataset_info": dataset_info,
            },
        )
    if dataset_info is None:
        log_stage("Dataset metadata missing from artifacts; loading data to rebuild metadata")
        if df is None:
            if data_path is None:
                raise ValueError("Input data path required to rebuild metadata; provide --input.")
            df = load_dataframe(data_path)
            df = df[df['grp_delete'] != True]
            df = df[df['st_files'] != 0]
        dataset_info = build_dataset_metadata(df, feature_names)

    n_samples = dataset_info.get("row_count", len(df) if df is not None else None)
    if n_samples is None:
        raise ValueError("Unable to determine sample size from dataset metadata or input data.")
    n_samples = int(n_samples)
    if gpu_max_cap:
        log_stage("Sampling from fitted GMM with max caps")
        cap_values = dataset_info.get("cap_values")
        if cap_values is None:
            raise ValueError("Cap values missing from dataset metadata; re-train and re-save the GMM artifacts.")
        synthetic_features, labels = sample_with_max_cap(
            best_gmm,
            n_samples,
            feature_names,
            cap_values,
            boxcox_shift,
            transformer=transformer_gpu,
            device=gpu_device,
            seed=seed,
        )
        if df is not None:
            synthetic_features.index = df.index
    else:
        log_stage("Sampling from fitted GMM")
        latent_samples, labels = best_gmm.sample(n_samples, random_state=seed)
        log_stage("Inverting Box-Cox transform")
        synthetic_features = invert_latent_samples_gpu(
            latent_samples,
            transformer_gpu,
            feature_names,
            boxcox_shift,
            device=gpu_device,
            index=df.index if df is not None else None,
        )
    log_stage("Applying count constraints")
    apply_count_constraints(synthetic_features)
    log_stage("Restoring constant columns")
    apply_constant_columns(synthetic_features, constant_cols, constant_values)
    log_stage("Applying boolean constraints")
    apply_bool_constraints(synthetic_features)

    synthetic_features["cluster"] = labels
    log_stage("Assembling final dataset")
    synthetic_dataset = assemble_dataset(synthetic_features, dataset_info)

    log_stage(f"Writing synthetic data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_dataset.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")

def main():
    args = parse_args()
    if args.input is None and args.gmm_pickle is None:
        raise SystemExit("Provide an input CSV or an existing --gmm-pickle.")
    data_path = Path(args.input) if args.input is not None else None
    output_path = Path(args.output)
    artifact_path = Path(args.gmm_pickle) if args.gmm_pickle else None
    gpu_kwargs = None
    gpu_kwargs = {
        "device": args.gpu_device,
        "max_iter": DEFAULT_MAX_ITER,
        "n_init": DEFAULT_N_INIT,
        "tol": DEFAULT_TOL,
        "batch_size": args.batch_size,
        "reg_covar": DEFAULT_REG_COVAR,
        "use_kmeans_init": DEFAULT_USE_KMEANS_INIT,
        "kmeans_iters": DEFAULT_KMEANS_ITERS,
    }
    generate_synthetic_dataset(
        output_path,
        data_path,
        gpu_kwargs=gpu_kwargs,
        gpu_max_cap=args.max_cap,
        seed=args.seed,
        artifact_path=artifact_path,
    )

if __name__ == "__main__":
    main()
