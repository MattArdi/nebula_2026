"""
Rail Corrugation Method: Random Forest + SMOTE
===============================================
CV macro F1 = 0.686

Baseline comparison model. Weaker than XGBoost on this imbalanced
multi-class problem, especially for the rare Side I class (14 samples).
Kept as a reference and for fast iteration.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier


PARAMS = dict(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
)
SMOTE_K = 3


def build_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=SMOTE_K)),
        ("rf",    RandomForestClassifier(**PARAMS)),
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
