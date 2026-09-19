"""
Door Subsystem (rule-based) — Feature Extraction & Classification
====================================================================
Fixed, hand-derived thresholds on motor current. No fitting step, no
training data beyond the EDA used to pick the two constants below.

Derivation (from Train.csv / Train_Segments_Answer.csv, 110 labelled cycles)
-----------------------------------------------------------------------------
Close cycles: max(current) perfectly separates the two classes in training
  data — Normal 2078-2266 mA, Abnormal resistance 1993-2046 mA (no overlap).
  Abnormal Close cycles show a LOWER peak current, not higher — this
  controller's stall/torque-limit logic caps current lower once it senses
  resistance while closing.

Open cycles: peak current does NOT separate the classes (both cluster
  ~2490-2560 mA), but mean current over the whole cycle does — Normal
  630-666 mA, Abnormal resistance 738-973 mA (no overlap). Abnormal Open
  cycles draw sustained excess current, most pronounced early in the
  stroke (first-third mean: Normal 722-758 mA vs Abnormal 941-1342 mA).

Both thresholds below sit at the midpoint of their respective training gap
and are the only two "learned" parameters in this model — everything else
is a fixed rule.
"""

import numpy as np
import pandas as pd

CLOSE_CUR_MAX_THRESHOLD = 2060.0   # gap sits at [1993-2046] Abnormal / [2078-2266] Normal
OPEN_CUR_MEAN_THRESHOLD = 700.0    # gap sits at [630-666] Normal / [738-973] Abnormal

NORMAL = "Normal"
ABNORMAL = "Abnormal resistance"


def extract_features(seg: pd.DataFrame) -> dict:
    """
    Compute the small feature set this rule (and diagnostics.py) reads.

    Parameters
    ----------
    seg : Sensor rows for a single cycle (a slice of a loaded sensor DataFrame).

    Returns
    -------
    Flat dict of feature_name -> float.
    """
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
    """'Close' or 'Open', read directly off the command flags (not predicted)."""
    return "Close" if seg["Close command"].mean() > 0.5 else "Open"


def classify(seg: pd.DataFrame, operation: str | None = None) -> tuple[str, dict]:
    """
    Classify one cycle as Normal / Abnormal resistance.

    Parameters
    ----------
    seg       : Sensor rows for a single cycle.
    operation : 'Close' or 'Open'; computed from `seg` if not supplied.

    Returns
    -------
    (label, features) : predicted label plus the feature dict it was
    computed from (diagnostics.py reuses this rather than recomputing).
    """
    if operation is None:
        operation = get_operation(seg)
    feats = extract_features(seg)

    if operation == "Close":
        label = ABNORMAL if feats["cur_max"] < CLOSE_CUR_MAX_THRESHOLD else NORMAL
    else:
        label = ABNORMAL if feats["cur_mean"] > OPEN_CUR_MEAN_THRESHOLD else NORMAL

    return label, feats
