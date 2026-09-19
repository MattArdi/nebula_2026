"""
Rail Corrugation Subsystem (v4, class-weighted ensemble) — Soft-Voting Ensemble
====================================================================================
Same three classifiers as v3 (CatBoost, XGBoost, LogReg on the 49-feature
set), same hyperparameters -- the only change is HOW their votes combine.

v2/v3 used one scalar weight per model, applied uniformly across all three
classes (cat=2, xgb=2, log=1). But solo per-class accuracy showed each
model is individually best at a DIFFERENT class:
    XGBoost  best at Normal   (98.8% solo)
    LogReg   best at Side I   (74.3% solo)
    CatBoost best at Side II  (95.8% solo)
so a single scalar per model forces a compromise: whatever weight helps a
model's strong class also inflates its vote on the classes it's weaker at.

v4 replaces the 3 scalars with a 3x3 model x class weight matrix -- each
model gets its own weight PER CLASS. Found by random search (3000 samples)
over repeated 5x5 stratified CV, then confirmed on a second, independent
set of CV seeds (both the search and the confirmation are in algorithm.md
Section 5): macro F1 0.8809 -> 0.8940 (seeds 0-4), 0.8931 -> 0.9034
(seeds 100-104) -- a robust, real gain, not a search artifact.

Final score for class c = sum_m WEIGHTS[m][c] * model_m.predict_proba(x)[c],
normalized per-row to sum to 1 (a per-row positive rescale, so argmax --
the only thing that determines the predicted label -- is unaffected; the
normalization exists purely so predict_proba's output stays a valid
probability distribution for anything downstream that reads it, e.g.
run_pipeline.py's --diagnostics-output).
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42

# Column order matches LabelEncoder's alphabetical sort of the 3 class
# strings: ["Normal", "Side I", "Side II"]. Verified against
# label_encoder.classes_ wherever this is used (see RailEnsemble.fit).
CAT_CLASS_WEIGHTS = np.array([4.0, 0.0, 4.0])  # Normal, Side I, Side II
XGB_CLASS_WEIGHTS = np.array([0.0, 4.0, 0.0])
LOG_CLASS_WEIGHTS = np.array([4.0, 0.0, 0.0])

# Filenames a saved ensemble is split across: the two boosted models use
# their own native formats (portable across library versions within a
# major release), everything else (label encoder, scaler, logreg -- all
# plain sklearn objects with no native serialization) goes in one joblib
# file together.
WEIGHTS_FILES = {
    "catboost": "catboost.cbm",
    "xgboost": "xgboost.json",
    "sklearn": "sklearn_components.joblib",
}


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


def combine_probas(cat_proba: np.ndarray, xgb_proba: np.ndarray, log_proba: np.ndarray) -> np.ndarray:
    """
    Per-class weighted combination (see module docstring), normalized so
    each row sums to 1. Shared by RailEnsemble.predict_proba and
    diagnostics.py's CV self-check, so the two can never drift apart.
    """
    raw = (
        cat_proba * CAT_CLASS_WEIGHTS[None, :]
        + xgb_proba * XGB_CLASS_WEIGHTS[None, :]
        + log_proba * LOG_CLASS_WEIGHTS[None, :]
    )
    return raw / raw.sum(axis=1, keepdims=True)


@dataclass
class RailEnsemble:
    """Class-weighted soft-voting ensemble of CatBoost + XGBoost + LogReg."""

    label_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    catboost: CatBoostClassifier = field(default_factory=make_catboost)
    xgboost: XGBClassifier = field(default_factory=make_xgboost)
    logreg: LogisticRegression = field(default_factory=make_logreg)

    def fit(self, X: np.ndarray, y_str: np.ndarray) -> "RailEnsemble":
        y = self.label_encoder.fit_transform(y_str)
        assert list(self.label_encoder.classes_) == ["Normal", "Side I", "Side II"], (
            f"CAT/XGB/LOG_CLASS_WEIGHTS assume column order [Normal, Side I, Side II], "
            f"got {list(self.label_encoder.classes_)}"
        )
        X_scaled = self.scaler.fit_transform(X)
        self.catboost.fit(X, y)
        self.xgboost.fit(X, y)
        self.logreg.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return combine_probas(
            self.catboost.predict_proba(X),
            self.xgboost.predict_proba(X),
            self.logreg.predict_proba(X_scaled),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels as the original strings (e.g. 'Side I')."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(idx)

    def save(self, weights_dir: Path) -> None:
        """Serialize a fitted ensemble to `weights_dir` (created if needed)."""
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.catboost.save_model(str(weights_dir / WEIGHTS_FILES["catboost"]))
        self.xgboost.save_model(str(weights_dir / WEIGHTS_FILES["xgboost"]))
        joblib.dump(
            {"label_encoder": self.label_encoder, "scaler": self.scaler, "logreg": self.logreg},
            weights_dir / WEIGHTS_FILES["sklearn"],
        )

    @staticmethod
    def weights_exist(weights_dir: Path) -> bool:
        weights_dir = Path(weights_dir)
        return all((weights_dir / fn).exists() for fn in WEIGHTS_FILES.values())

    @classmethod
    def load(cls, weights_dir: Path) -> "RailEnsemble":
        """Load a previously `save()`d ensemble -- no fitting, ready to predict."""
        weights_dir = Path(weights_dir)
        ensemble = cls()
        ensemble.catboost.load_model(str(weights_dir / WEIGHTS_FILES["catboost"]))
        ensemble.xgboost.load_model(str(weights_dir / WEIGHTS_FILES["xgboost"]))
        components = joblib.load(weights_dir / WEIGHTS_FILES["sklearn"])
        ensemble.label_encoder = components["label_encoder"]
        ensemble.scaler = components["scaler"]
        ensemble.logreg = components["logreg"]
        return ensemble
