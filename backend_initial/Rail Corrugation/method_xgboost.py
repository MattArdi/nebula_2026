"""
Rail Corrugation Method: XGBoost + SMOTE (best model)
======================================================
CV macro F1 = 0.850

Key design choices
------------------
- SMOTE with k=3 neighbours oversamples the rare classes (Side I: 14 samples,
  Side II: 24 samples) before fitting XGBoost.
- XGBoost hyperparameters: depth=5, lr=0.03, n=500, subsample=0.8,
  colsample_bytree=0.7 — found by grid search in the exploration phase.
- Uses imblearn Pipeline so SMOTE is applied inside each CV fold
  (preventing data leakage from the oversampled minority class).
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Best hyperparameters from tuning
PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)
SMOTE_K = 3  # small because Side I only has 14 training samples


def build_pipeline() -> ImbPipeline:
    """Return a fresh (unfitted) SMOTE → XGBoost pipeline."""
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=SMOTE_K)),
        ("xgb",   XGBClassifier(**PARAMS)),
    ])


def fit(X: np.ndarray, y: np.ndarray) -> ImbPipeline:
    """
    Fit the pipeline on the full training set.

    Parameters
    ----------
    X : Feature matrix, shape (n_samples, n_features).
    y : Integer-encoded labels, shape (n_samples,).

    Returns
    -------
    pipeline : Fitted ImbPipeline.
    """
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def predict(pipeline: ImbPipeline, X: np.ndarray) -> np.ndarray:
    """Return predicted integer-encoded labels."""
    return pipeline.predict(X)
