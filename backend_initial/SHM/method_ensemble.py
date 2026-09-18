"""
SHM Method: Ensemble — Ridge (70%) + XGBoost (30%)
====================================================
CV score (max(0, 1 − MAPE)) = 0.840  — best overall model.

Strategy
--------
Simple weighted average of Ridge and XGBoost predictions (both in raw
damage space after back-transforming from log). Ridge handles the smooth,
low-damage end of the distribution well; XGBoost captures non-linear
relationships at higher damage values. Weighting found by grid search
over [0.3, 0.5, 0.7] for w_ridge.

Both sub-models are kept as attributes on the returned "model" object
so training.py can call a uniform fit() / predict() interface.
"""

import numpy as np

import method_ridge as ridge_mod
import method_xgboost as xgb_mod


W_RIDGE = 0.7   # weight for Ridge predictions
W_XGB   = 0.3   # weight for XGBoost predictions


class EnsembleModel:
    """Thin wrapper holding both fitted sub-models."""

    def __init__(self, ridge, xgb):
        self.ridge = ridge
        self.xgb   = xgb


def fit(X: np.ndarray, y: np.ndarray) -> EnsembleModel:
    """
    Fit both sub-models on the training set.

    Parameters
    ----------
    X : Feature matrix, shape (n_samples, n_features).
    y : Raw damage values (positive floats), shape (n_samples,).

    Returns
    -------
    model : EnsembleModel with fitted ridge and xgb attributes.
    """
    ridge = ridge_mod.fit(X, y)
    xgb   = xgb_mod.fit(X, y)
    return EnsembleModel(ridge, xgb)


def predict(model: EnsembleModel, X: np.ndarray) -> np.ndarray:
    """
    Predict as a weighted average of Ridge and XGBoost predictions.

    Parameters
    ----------
    model : Fitted EnsembleModel from fit().
    X     : Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    y_pred : Predicted damage values (positive floats).
    """
    y_ridge = ridge_mod.predict(model.ridge, X)
    y_xgb   = xgb_mod.predict(model.xgb,   X)
    return W_RIDGE * y_ridge + W_XGB * y_xgb
