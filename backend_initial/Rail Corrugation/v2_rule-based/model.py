"""
Rail Corrugation Subsystem (rule-based) — Soft-Voting Ensemble
==================================================================
Three classifiers, weighted soft-voting on predict_proba (weighted average
of class probabilities, then argmax). Weights and hyperparameters below
were selected by repeated cross-validation against the real competition
metric (macro F1) -- see algorithm.md Section 3 for what else was tried.

    CatBoost   (weight 2) -- class-balanced, best minority-class recall
    XGBoost    (weight 2) -- best single-model macro F1
    LogReg     (weight 1) -- linear signal on standardized features,
                              decorrelated enough from the two boosted
                              trees to help the vote

CatBoost and XGBoost train on raw (unscaled) features; Logistic Regression
trains on features standardized with a StandardScaler fit on the training
fold only.
"""

from dataclasses import dataclass, field

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

CATBOOST_WEIGHT = 2
XGBOOST_WEIGHT = 2
LOGREG_WEIGHT = 1
RANDOM_STATE = 42


def make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.1,
        auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=False,
    )


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
    )


def make_logreg() -> LogisticRegression:
    return LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )


@dataclass
class RailEnsemble:
    """Weighted soft-voting ensemble of CatBoost + XGBoost + LogReg."""

    label_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    catboost: CatBoostClassifier = field(default_factory=make_catboost)
    xgboost: XGBClassifier = field(default_factory=make_xgboost)
    logreg: LogisticRegression = field(default_factory=make_logreg)

    def fit(self, X: np.ndarray, y_str: np.ndarray) -> "RailEnsemble":
        y = self.label_encoder.fit_transform(y_str)
        X_scaled = self.scaler.fit_transform(X)
        self.catboost.fit(X, y)
        self.xgboost.fit(X, y)
        self.logreg.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        probs = (
            CATBOOST_WEIGHT * self.catboost.predict_proba(X)
            + XGBOOST_WEIGHT * self.xgboost.predict_proba(X)
            + LOGREG_WEIGHT * self.logreg.predict_proba(X_scaled)
        )
        return probs / (CATBOOST_WEIGHT + XGBOOST_WEIGHT + LOGREG_WEIGHT)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels as the original strings (e.g. 'Side I')."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(idx)
