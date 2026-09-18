"""
Door Method: Gradient Boosting Classifier (baseline)
=====================================================
CV macro F1 = 0.988 — strong baseline, marginally weaker than Random Forest.

Key design choices
------------------
- Shallow trees (max_depth=3) to prevent overfitting on 110 samples.
- 200 estimators with learning_rate=0.05.
- No class_weight parameter in GradientBoostingClassifier — imbalance
  (80:30) is mild enough not to require it here.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
)


def build_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(**PARAMS)


def fit(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[GradientBoostingClassifier, LabelEncoder]:
    """
    Fit the classifier.

    Parameters
    ----------
    X : Feature matrix, shape (n_segments, n_features).
    y : String labels array ('Normal' / 'Abnormal resistance').

    Returns
    -------
    (model, le) : Fitted classifier and LabelEncoder.
    """
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = build_model()
    model.fit(X, y_enc)
    return model, le


def predict(
    model: GradientBoostingClassifier,
    le: LabelEncoder,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict string labels.

    Parameters
    ----------
    model : Fitted GradientBoostingClassifier from fit().
    le    : Fitted LabelEncoder from fit().
    X     : Feature matrix, shape (n_segments, n_features).

    Returns
    -------
    labels : String label array ('Normal' / 'Abnormal resistance').
    """
    return le.inverse_transform(model.predict(X))
