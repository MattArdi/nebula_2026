"""
Door Subsystem (v3, adaptive) — Calibration
================================================
Computed once from Train.csv + Train_Segments_Answer.csv, saved as
artifacts/calibration.json. Two things this calibrates, one per operation:

1. `train_baseline`, `window_n`, `seed_window` — for rules.AdaptiveThreshold.
   train_baseline is the median decision-feature value across Train's Normal
   cycles for that operation; seed_window is the most recent `window_n` of
   those (in stream order) cycles' values, so a fresh run_pipeline.py starts
   its rolling baseline already primed with Train's own recent history
   instead of cold.

2. `ci_half_width` — how much the original gap-midpoint threshold could
   plausibly have landed elsewhere, estimated by bootstrap resampling the
   same 15 Abnormal / 40 Normal training examples this rule was derived
   from. See algorithm.md Section 2 for how this is turned into a
   per-prediction confidence flag.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd

WINDOW_N = 20
N_BOOTSTRAP = 2000
CI_LOWER_PCT = 5
CI_UPPER_PCT = 95
RANDOM_STATE = 42

FIXED_THRESHOLD = {"Close": 2060.0, "Open": 700.0}
DECISION_FEATURE = {"Close": "cur_max", "Open": "cur_mean"}
# Close: Abnormal is the LOWER class (peak current drops under resistance).
# Open:  Abnormal is the UPPER class (mean current rises under resistance).
ABNORMAL_IS_LOWER = {"Close": True, "Open": False}


def _gap_midpoint_threshold(normal_vals: np.ndarray, abnormal_vals: np.ndarray, abnormal_is_lower: bool) -> float:
    """Same derivation as v2's original threshold: midpoint of the gap between the two classes."""
    if abnormal_is_lower:
        return (abnormal_vals.max() + normal_vals.min()) / 2.0
    return (normal_vals.max() + abnormal_vals.min()) / 2.0


def build_features_df(data_dir: Path) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import rules
    import segmentation as seg_mod

    df = seg_mod.load_sensor_csv(data_dir / "Train.csv")
    ans = pd.read_csv(data_dir / "Train_Segments_Answer.csv")

    seg_idx = np.zeros(len(df), dtype=int)
    cursor = 0
    for i, n in enumerate(ans["n_rows"]):
        seg_idx[cursor: cursor + n] = i
        cursor += n
    df = df.copy()
    df["seg_idx"] = seg_idx

    rows = []
    for i, row in ans.iterrows():
        seg = df[df["seg_idx"] == i]
        feats = rules.extract_features(seg)
        feats["operation"] = row["operation"]
        feats["status"] = row["status"]
        feats["order"] = i
        rows.append(feats)
    return pd.DataFrame(rows)


def calibrate(data_dir: Path) -> dict:
    fd = build_features_df(data_dir)
    rng = np.random.RandomState(RANDOM_STATE)

    result = {}
    for op in ("Close", "Open"):
        feat = DECISION_FEATURE[op]
        sub = fd[fd["operation"] == op].sort_values("order")
        normal = sub[sub["status"] == "Normal"]
        abnormal = sub[sub["status"] == "Abnormal resistance"]

        seed_window = normal[feat].tail(WINDOW_N).tolist()
        # Defined as the median of the seed window itself (not all Normal
        # examples) -- so a freshly-calibrated run starts at EXACTLY zero
        # offset from the fixed threshold, by construction, not by chance.
        train_baseline = float(np.median(seed_window))

        # Bootstrap the gap-midpoint threshold over resamples of the same
        # training examples it was originally derived from.
        n_vals, a_vals = normal[feat].to_numpy(), abnormal[feat].to_numpy()
        boot_thresholds = []
        for _ in range(N_BOOTSTRAP):
            n_bs = rng.choice(n_vals, size=len(n_vals), replace=True)
            a_bs = rng.choice(a_vals, size=len(a_vals), replace=True)
            boot_thresholds.append(_gap_midpoint_threshold(n_bs, a_bs, ABNORMAL_IS_LOWER[op]))
        boot_thresholds = np.array(boot_thresholds)
        lo, hi = np.percentile(boot_thresholds, [CI_LOWER_PCT, CI_UPPER_PCT])
        ci_half_width = float((hi - lo) / 2.0)

        result[op] = {
            "fixed_threshold": FIXED_THRESHOLD[op],
            "train_baseline": train_baseline,
            "window_n": WINDOW_N,
            "seed_window": seed_window,
            "ci_half_width": ci_half_width,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_normal": int(len(normal)),
            "n_abnormal": int(len(abnormal)),
        }
    return result


def save_calibration(path: Path, calib: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))


def load_calibration(path: Path) -> dict:
    return json.loads(Path(path).read_text())
