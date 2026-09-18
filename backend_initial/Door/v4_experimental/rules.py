"""
Door Subsystem (v4, EXPERIMENTAL) — Feature Extraction & Hybrid Classification
================================================================================
v4 exists to answer one question: can a more sophisticated, multi-feature rule
correctly flip the one segment v3 is suspected to get wrong (Open cycle at
2023-7-5-0-20-55-731 in Test.csv, cur_mean=543.28 -- see algorithm.md Section 1)
without breaking anything else?

Answer, found by building and testing this: NO. See algorithm.md Section 3 for
the full account. This module is kept exactly as tested -- including the
approach that made things worse -- so the result is reproducible, not just
asserted. **Do not point predict.py at this version.**

Close: unchanged from v2/v3 -- the fixed cur_max < 2060 threshold. A
per-class-scaled multivariate version was tried for Close too and scored
worse under leave-one-out (53/55 vs. the fixed threshold's 55/55), so it was
not adopted there either.

Open: a per-class-variance-scaled nearest-centroid distance over
(cur_mean, cur_early_mean), instead of the single-feature cur_mean > 700
threshold. This DOES flip the suspected segment toward Abnormal -- but it
also flips 8 of Test.csv's other 18 Open cycles from Normal to Abnormal,
a wholesale collapse in precision, not a targeted correction.
"""

import numpy as np
import pandas as pd

NORMAL = "Normal"
ABNORMAL = "Abnormal resistance"

CLOSE_CUR_MAX_THRESHOLD = 2060.0  # unchanged from v2/v3
OPEN_DIST_FEATURES = ["cur_mean", "cur_early_mean"]


def extract_features(seg: pd.DataFrame) -> dict:
    """Identical to v2/v3."""
    cur = seg["Motor current(mA)"].to_numpy(dtype=float)
    pos = seg["Door leaf position"].to_numpy(dtype=float)
    third = max(len(cur) // 3, 1)
    return {
        "n_rows": float(len(seg)),
        "cur_max": float(cur.max()),
        "cur_mean": float(cur.mean()),
        "cur_early_mean": float(cur[:third].mean()),
        "pos_min": float(pos.min()),
        "pos_max": float(pos.max()),
    }


def get_operation(seg: pd.DataFrame) -> str:
    """Identical to v2/v3."""
    return "Close" if seg["Close command"].mean() > 0.5 else "Open"


def _per_class_scaled_distance(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> float:
    """Euclidean distance after scaling each feature by ONE class's own std
    (not pooled) -- see algorithm.md Section 3 for why this is the specific
    thing that goes wrong: a tight class's distance becomes hypersensitive,
    a loose (small-sample) class's becomes under-sensitive."""
    sd = np.where(sd < 1e-6, 1e-6, sd)
    return float(np.sqrt(np.sum(((x - mu) / sd) ** 2)))


def classify(feats: dict, operation: str, open_class_stats: dict) -> tuple[str, dict]:
    if operation == "Close":
        label = ABNORMAL if feats["cur_max"] < CLOSE_CUR_MAX_THRESHOLD else NORMAL
        return label, {"method": "fixed_threshold_1d"}

    x = np.array([feats[f] for f in OPEN_DIST_FEATURES])
    muN, sdN = np.array(open_class_stats["normal_mean"]), np.array(open_class_stats["normal_std"])
    muA, sdA = np.array(open_class_stats["abnormal_mean"]), np.array(open_class_stats["abnormal_std"])
    dN = _per_class_scaled_distance(x, muN, sdN)
    dA = _per_class_scaled_distance(x, muA, sdA)
    label = NORMAL if dN < dA else ABNORMAL
    return label, {"method": "per_class_scaled_distance", "dist_normal": dN, "dist_abnormal": dA}
