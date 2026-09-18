"""
SHM Method: XGBoost Regression (log-transformed target)
=========================================================
CV score (max(0, 1 − MAPE)) = 0.817

Key design choices
------------------
- Same log-transform strategy as Ridge: fit on log(y), predict exp(ŷ).
- Shallow trees (depth=3) and low learning rate (0.03) to avoid
  overfitting on only 64 training samples.
- Weaker than Ridge as a standalone model on this dataset but contributes
  complementary signal when ensembled (the ensemble reaches 0.840).
"""

import numpy as np
from xgboost import XGBRegressor


PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)


def build_model() -> XGBRegressor:
    return XGBRegressor(**PARAMS)


def fit(X: np.ndarray, y: np.ndarray) -> XGBRegressor:
    """
    Fit on log(y).

    Parameters
    ----------
    X : Feature matrix, shape (n_samples, n_features).
    y : Raw damage values (positive floats), shape (n_samples,).

    Returns
    -------
    model : Fitted XGBRegressor (predicts log-damage).
    """
    model = build_model()
    model.fit(X, np.log(y))
    return model


def predict(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    """
    Predict damage values (back-transformed from log-space).

    Parameters
    ----------
    model : Fitted model from fit().
    X     : Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    y_pred : Predicted damage values (positive floats).
    """
    return np.exp(model.predict(X))
