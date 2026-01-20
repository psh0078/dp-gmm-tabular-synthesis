from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

def prepare_boxcox_features_cpu(df: pd.DataFrame, feature_cols: Sequence[str], boxcox_shift: float):
    X = df[feature_cols].copy()
    X_boxcox_input = X + boxcox_shift

    constant_cols = X_boxcox_input.columns[X_boxcox_input.nunique() <= 1].tolist()
    if constant_cols:
        print(f"Skipping constant columns for Box-Cox: {constant_cols}")
    constant_values = {col: X[col].iloc[0] for col in constant_cols}

    train_cols = [col for col in feature_cols if col not in constant_cols]
    if not train_cols:
        raise ValueError("No features left after removing constant columns.")

    transformer = PowerTransformer(method="box-cox", standardize=True)
    X_boxcox = transformer.fit_transform(X_boxcox_input[train_cols])
    X_boxcox = pd.DataFrame(X_boxcox, columns=train_cols, index=df.index)
    return X_boxcox, transformer, constant_cols, constant_values


def fit_best_gmm_cpu(
    X_boxcox: pd.DataFrame,
    component_grid,
    random_state=42,
    reg_covar: float = 5e-3,
):
    bic_records = []
    best_model = None
    best_bic = np.inf
    for n in component_grid:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=random_state,
            reg_covar=reg_covar,
            max_iter=500,
        )
        gmm.fit(X_boxcox)
        bic = gmm.bic(X_boxcox)
        bic_records.append({"n_components": n, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
    if best_model is None:
        raise RuntimeError("GMM fitting failed to produce a model.")
    bic_df = pd.DataFrame(bic_records)
    return best_model, bic_df


def invert_latent_samples(latent_samples, transformer, feature_names, boxcox_shift: float):
    latent_df = pd.DataFrame(latent_samples, columns=feature_names)
    positive = transformer.inverse_transform(latent_df.values)
    positive = pd.DataFrame(positive, columns=feature_names, index=latent_df.index)
    synthetic = positive - boxcox_shift
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan)
    synthetic = synthetic.fillna(0)
    synthetic = synthetic.clip(lower=0)
    return synthetic
