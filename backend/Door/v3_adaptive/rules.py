"""
Door Subsystem (v3, adaptive) — Feature Extraction & Adaptive Classification
================================================================================
Extends v2's two fixed-threshold rule with two things v2 didn't have (see
algorithm.md Section 1 for why these were needed):

1. A ROLLING LOCAL BASELINE, so the decision threshold shifts with whatever
   baseline a given door/stream is actually running at, instead of assuming
   every door matches Train's exact current level. Whenever the local
   baseline matches Train's (no drift -- the only case this dataset ever
   demonstrates), the adjusted threshold reduces EXACTLY to v2's fixed
   constant, by construction, not by coincidence.

2. A BOOTSTRAP CONFIDENCE HALF-WIDTH per threshold, quantifying how much the
   original gap-midpoint threshold could plausibly have landed elsewhere
   given only 15 Abnormal training examples per operation, so a prediction
   near that boundary can be flagged as a genuinely uncertain call instead of
   reported with the same confidence as one far from it.

Both are calibrated once from Train (see calibration.py) and shipped as an
artifact, same convention as every other subsystem's v2 pipeline.
"""

from collections import deque

import numpy as np
import pandas as pd

NORMAL = "Normal"
ABNORMAL = "Abnormal resistance"


def extract_features(seg: pd.DataFrame) -> dict:
    """Identical to v2 -- see v2_rule-based/rules.py for the derivation."""
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
    """Identical to v2: 'Close' or 'Open', read directly off the command flags."""
    return "Close" if seg["Close command"].mean() > 0.5 else "Open"


class AdaptiveThreshold:
    """
    Tracks a robust rolling baseline (median of the decision feature, over
    the most recent `window_n` cycles PREDICTED Normal for this operation)
    and exposes a threshold that shifts by exactly the baseline's drift from
    Train's own baseline.

    Only cycles predicted Normal ever enter the window -- a cycle flagged
    Abnormal must never pull the "what does normal look like" estimate
    toward itself, or the threshold would progressively desensitize after
    every detection (see algorithm.md Section 3).
    """

    def __init__(self, fixed_threshold: float, train_baseline: float,
                 window_n: int, seed: list[float] | None = None):
        self.fixed_threshold = fixed_threshold
        self.train_baseline = train_baseline
        self.window: deque = deque(seed or [], maxlen=window_n)

    @property
    def baseline(self) -> float:
        return float(np.median(self.window)) if self.window else self.train_baseline

    @property
    def adjusted_threshold(self) -> float:
        return self.fixed_threshold + (self.baseline - self.train_baseline)

    def update(self, value: float, predicted_label: str) -> None:
        if predicted_label == NORMAL:
            self.window.append(value)

    def to_state(self) -> dict:
        return {"window": list(self.window)}

    @classmethod
    def from_calibration(cls, calib: dict, seed: list[float] | None = None) -> "AdaptiveThreshold":
        return cls(
            fixed_threshold=calib["fixed_threshold"],
            train_baseline=calib["train_baseline"],
            window_n=calib["window_n"],
            seed=seed if seed is not None else list(calib.get("seed_window", [])),
        )


def classify_one(feats: dict, operation: str, thresholds: dict[str, AdaptiveThreshold]) -> tuple[str, dict]:
    """
    Classify one cycle against its operation's adjusted threshold.

    Deliberately does NOT call `AdaptiveThreshold.update()` here -- the
    threshold stays FROZEN at whatever the calibration artifact seeded it
    with for the entire run. An earlier version updated the window live as
    each cycle was classified (test-time self-training on the pipeline's own
    Normal predictions); on Door's real Test.csv that let the baseline drift
    by ~13 mA over the stream and flipped one borderline segment's label
    relative to v2 with no accuracy benefit to show for it -- see
    algorithm.md Section 3. Adaptation now happens only BETWEEN runs, via
    `--retrain` recalibrating the seed window from a new door's own history,
    never within one.
    """
    at = thresholds[operation]
    threshold = at.adjusted_threshold

    if operation == "Close":
        value = feats["cur_max"]
        label = ABNORMAL if value < threshold else NORMAL
    else:
        value = feats["cur_mean"]
        label = ABNORMAL if value > threshold else NORMAL

    return label, {"threshold_used": threshold, "baseline_used": at.baseline}
