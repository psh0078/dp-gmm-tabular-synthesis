from pathlib import Path
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
DEFAULT_COMPONENT_GRID = [12, 18, 24, 30, 36] 
# DEFAULT_COMPONENT_GRID = [12] 
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

def assemble_dataset(df: pd.DataFrame, synthetic_features: pd.DataFrame):
    synthetic_dataset = df.copy()
    for col in synthetic_features.columns:
        if col in synthetic_dataset.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            synthetic_dataset[col] = synthetic_features[col].clip(lower=col_min, upper=col_max)
    synthetic_dataset = synthetic_dataset[df.columns]
    synthetic_dataset['sync'] = (synthetic_dataset['st_files_skipped'] > 0).astype(int)
    return synthetic_dataset

def calibrate_feature_column(synthetic_features: pd.DataFrame, df: pd.DataFrame, col: str, quantile_levels):
    if col in synthetic_features.columns and col in df.columns:
        target_values = df[col].astype(float).values
        source_values = synthetic_features[col].astype(float).values
        calibrated = quantile_calibrate(source_values, target_values, quantile_levels=quantile_levels)
        synthetic_features[col] = np.clip(calibrated, 0, None)

def generate_synthetic_dataset(
    data_path: Path,
    output_path: Path,
    component_grid=DEFAULT_COMPONENT_GRID,
    gpu_kwargs: dict | None = None,
    gpu_max_cap: bool = False,
    seed: int = 42,
):
    log_stage(f"Loading data from {data_path}")
    df = load_dataframe(data_path)
    df = df[df['grp_delete'] != True]
    df = df[df['st_files'] != 0]
    gpu_kwargs = gpu_kwargs or {}
    gpu_device = gpu_kwargs.get("device", "cuda")
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
    if gpu_max_cap:
        log_stage("Sampling from fitted GMM with max caps")
        cap_values = df[feature_names].max()
        for col in BOOL_LIKE_COLS:
            if col in cap_values.index:
                cap_values[col] = np.inf
        synthetic_features, labels = sample_with_max_cap(
            best_gmm,
            len(df),
            feature_names,
            cap_values,
            BOXCOX_SHIFT,
            transformer=transformer_gpu,
            device=gpu_device,
            seed=seed,
        )
        synthetic_features.index = df.index
    else:
        log_stage("Sampling from fitted GMM")
        latent_samples, labels = best_gmm.sample(len(df), random_state=seed)
        log_stage("Inverting Box-Cox transform")
        synthetic_features = invert_latent_samples_gpu(
            latent_samples,
            transformer_gpu,
            feature_names,
            BOXCOX_SHIFT,
            device=gpu_device,
            index=df.index,
        )
    log_stage("Applying count constraints")
    apply_count_constraints(synthetic_features)
    log_stage("Restoring constant columns")
    apply_constant_columns(synthetic_features, constant_cols, constant_values)
    log_stage("Applying boolean constraints")
    apply_bool_constraints(synthetic_features)

    synthetic_features["cluster"] = labels
    log_stage("Assembling final dataset")
    synthetic_dataset = assemble_dataset(df, synthetic_features)

    log_stage(f"Writing synthetic data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_dataset.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")

def main():
    args = parse_args()
    data_path = Path(args.input)
    output_path = Path(args.output)
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
        data_path,
        output_path,
        gpu_kwargs=gpu_kwargs,
        gpu_max_cap=args.max_cap,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
