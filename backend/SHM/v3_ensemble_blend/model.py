"""
SHM Subsystem (v3, ensemble blend) — Physics + ML Weighted Blend
======================================================================
SHIPPED DESPITE THE EVIDENCE, ON EXPLICIT INSTRUCTION -- see algorithm.md
Section 5 for the full honest account. Every individual ML model tested
(19 of them, across regression/bagging/boosting/kernel families) scored
below the pure physics model (v2's shipped approach, kept here as one of
this blend's 4 members). This module blends 4 members -- physics, plain
linear regression on the physics proxy, bagged Ridge on the physics
proxy, and Extra Trees on a 47-feature set -- with weights found by
grid-searching all 64 training files' leave-one-out predictions at once:

    PHYSICS_WEIGHT = 7
    LINEAR_WEIGHT = 0
    BAGGED_RIDGE_WEIGHT = 0
    EXTRA_TREES_WEIGHT = 1

giving LOO score 0.9750 vs the pure physics model's 0.9744 -- a +0.0006
gain, found by optimizing 4 weights directly against all 64 LOO errors
with no second, independent dataset to confirm it on (unlike every
weight search done for Rail Corrugation in this same session, all of
which were confirmed on a disjoint set of CV seeds before being trusted).
This gain is the same size as several other changes this session
confirmed were noise (SHM's own calibration-loss and exponent-sweep
experiments, both +/-0.0005). Treat this shipped weighting as unvalidated.

LINEAR_WEIGHT and BAGGED_RIDGE_WEIGHT are 0 in the found optimum -- both
members are still fit and predicted-from below (not deleted), so the
weights stay swappable without restructuring the pipeline if a future
retrain (see run_pipeline.py --retrain) finds a different optimum.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import features as feat_mod
import physics

RANDOM_STATE = 42

PHYSICS_WEIGHT = 7.0
LINEAR_WEIGHT = 0.0
BAGGED_RIDGE_WEIGHT = 0.0
EXTRA_TREES_WEIGHT = 1.0

ARTIFACT_FILES = {
    "physics": "physics_calibration.json",
    "sklearn": "sklearn_components.joblib",
}


def make_linear() -> LinearRegression:
    return LinearRegression()


def make_bagged_ridge() -> BaggingRegressor:
    return BaggingRegressor(estimator=Ridge(alpha=0.1), n_estimators=300, random_state=RANDOM_STATE)


def make_extra_trees() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE)


def fit_physics_calibration(proxies: np.ndarray, damages: np.ndarray) -> float:
    """C such that damage ~= proxy/C -- identical to v2's calibration (median of ratios)."""
    return float(np.median(proxies / damages))


@dataclass
class EnsembleBlend:
    """4-member weighted-average blend: physics + linear + bagged_ridge + extra_trees."""

    physics_C: float = 0.0
    physics_m: float = physics.M_EXPONENT
    proxy_scaler: StandardScaler = field(default_factory=StandardScaler)
    linear: LinearRegression = field(default_factory=make_linear)
    bagged_ridge: BaggingRegressor = field(default_factory=make_bagged_ridge)
    extra_trees: ExtraTreesRegressor = field(default_factory=make_extra_trees)

    def fit(self, feature_rows: list[dict], damages: np.ndarray) -> "EnsembleBlend":
        X_all = np.array([[row[c] for c in feat_mod.FEATURE_NAMES] for row in feature_rows])
        proxies = X_all[:, feat_mod.FEATURE_NAMES.index(feat_mod.PHYSICS_PROXY_COL)]

        self.physics_C = fit_physics_calibration(proxies, damages)

        X_proxy = proxies.reshape(-1, 1)
        X_proxy_scaled = self.proxy_scaler.fit_transform(X_proxy)
        self.linear.fit(X_proxy_scaled, damages)
        self.bagged_ridge.fit(X_proxy_scaled, damages)

        self.extra_trees.fit(X_all, damages)
        return self

    def predict(self, feature_rows: list[dict]) -> np.ndarray:
        X_all = np.array([[row[c] for c in feat_mod.FEATURE_NAMES] for row in feature_rows])
        proxies = X_all[:, feat_mod.FEATURE_NAMES.index(feat_mod.PHYSICS_PROXY_COL)]

        physics_pred = proxies / self.physics_C

        X_proxy_scaled = self.proxy_scaler.transform(proxies.reshape(-1, 1))
        linear_pred = self.linear.predict(X_proxy_scaled)
        bagged_ridge_pred = self.bagged_ridge.predict(X_proxy_scaled)

        extra_trees_pred = self.extra_trees.predict(X_all)

        total_weight = PHYSICS_WEIGHT + LINEAR_WEIGHT + BAGGED_RIDGE_WEIGHT + EXTRA_TREES_WEIGHT
        blend = (
            PHYSICS_WEIGHT * physics_pred
            + LINEAR_WEIGHT * linear_pred
            + BAGGED_RIDGE_WEIGHT * bagged_ridge_pred
            + EXTRA_TREES_WEIGHT * extra_trees_pred
        ) / total_weight
        return blend

    def save(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / ARTIFACT_FILES["physics"]).write_text(
            json.dumps({"C": self.physics_C, "m": self.physics_m}, indent=2)
        )
        joblib.dump(
            {
                "proxy_scaler": self.proxy_scaler, "linear": self.linear,
                "bagged_ridge": self.bagged_ridge, "extra_trees": self.extra_trees,
            },
            artifacts_dir / ARTIFACT_FILES["sklearn"],
        )

    @staticmethod
    def artifacts_exist(artifacts_dir: Path) -> bool:
        artifacts_dir = Path(artifacts_dir)
        return all((artifacts_dir / fn).exists() for fn in ARTIFACT_FILES.values())

    @classmethod
    def load(cls, artifacts_dir: Path) -> "EnsembleBlend":
        artifacts_dir = Path(artifacts_dir)
        physics_json = json.loads((artifacts_dir / ARTIFACT_FILES["physics"]).read_text())
        components = joblib.load(artifacts_dir / ARTIFACT_FILES["sklearn"])
        blend = cls(physics_C=physics_json["C"], physics_m=physics_json["m"])
        blend.proxy_scaler = components["proxy_scaler"]
        blend.linear = components["linear"]
        blend.bagged_ridge = components["bagged_ridge"]
        blend.extra_trees = components["extra_trees"]
        return blend
