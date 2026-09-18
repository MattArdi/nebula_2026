"""
Door Subsystem (v4, EXPERIMENTAL) — Calibration
====================================================
Computes Open's per-class (mean, std) for (cur_mean, cur_early_mean) from
Train, used by rules.classify()'s per-class-scaled distance. Close needs no
calibration -- it reuses v2/v3's fixed 2060 mA threshold unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rules import OPEN_DIST_FEATURES


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
        rows.append(feats)
    return pd.DataFrame(rows)


def calibrate(data_dir: Path) -> dict:
    fd = build_features_df(data_dir)
    sub = fd[fd["operation"] == "Open"]
    normal = sub[sub["status"] == "Normal"][OPEN_DIST_FEATURES].values
    abnormal = sub[sub["status"] == "Abnormal resistance"][OPEN_DIST_FEATURES].values

    return {
        "normal_mean": normal.mean(axis=0).tolist(),
        "normal_std": normal.std(axis=0).tolist(),
        "abnormal_mean": abnormal.mean(axis=0).tolist(),
        "abnormal_std": abnormal.std(axis=0).tolist(),
        "n_normal": int(len(normal)),
        "n_abnormal": int(len(abnormal)),
    }


def save_calibration(path: Path, calib: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))


def load_calibration(path: Path) -> dict:
    return json.loads(Path(path).read_text())
