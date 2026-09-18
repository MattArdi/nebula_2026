"""
SHM Method: Ridge Regression (log-transformed target)
======================================================
CV score (max(0, 1 − MAPE)) = 0.829  — best single model.

Key design choices
------------------
- Target is log-transformed before fitting and exp-transformed after
  prediction. MAPE is a multiplicative metric, so modelling in log-space
  penalises relative errors equally across the full damage range (0.03–0.93).
- Ridge regularisation (α = 0.5) prevents overfitting on only 64 training
  samples. Standardisation is applied inside the pipeline so feature scales
  don't affect the regularisation penalty.
- Chosen over XGBoost alone because with 64 samples and a smooth regression
  target, the linear regularised model generalises more reliably.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALPHA = 0.5   # regularisation strength — tuned by grid search


def build_pipeline() -> Pipeline:
    """Return a fresh (unfitted) StandardScaler → Ridge pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=ALPHA)),
    ])


def fit(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Fit on log(y); return fitted pipeline that still operates in log-space.

    Parameters
    ----------
    X : Feature matrix, shape (n_samples, n_features).
    y : Raw damage values (positive floats), shape (n_samples,).

    Returns
    -------
    pipeline : Fitted sklearn Pipeline (predicts log-damage).
    """
    pipe = build_pipeline()
    pipe.fit(X, np.log(y))
    return pipe


def predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Predict damage values (back-transformed from log-space).

    Parameters
    ----------
    pipeline : Fitted pipeline from fit().
    X        : Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    y_pred : Predicted damage values (positive floats).
    """
    return np.exp(pipeline.predict(X))
