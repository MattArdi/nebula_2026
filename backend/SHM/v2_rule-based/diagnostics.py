"""
SHM Subsystem (rule-based) — Diagnostics
============================================
Two checks, run before predictions are trusted:

1. self_check_loo() -- leave-one-out cross-validation of the full pipeline
   (physics.py + model.py) against Train_Labels.csv, scored with the real
   competition metric (max(0, 1-MAPE)). C is refit on the other 63 files
   for every fold, so this is a genuine held-out estimate, not an
   in-sample fit re-reported as a score.

2. flag_extrapolation() -- for a file with no label (Test), the model can
   still be checked for whether its Miner's-rule proxy falls inside the
   range Train ever demonstrated. Outside that range, the proxy/C linear
   relationship is being extrapolated rather than interpolated -- worth
   surfacing given the training data's own residual error concentrates
   in the extremes of the damage distribution (see algorithm.md).
"""

import numpy as np
import pandas as pd


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mape = float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))
    return max(0.0, 1 - mape), mape


def self_check_loo(proxy_df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out CV of the calibration fit, from a precomputed proxy table."""
    proxies = proxy_df["proxy"].values
    damages = proxy_df["damage"].values
    n = len(damages)

    rows = []
    for i in range(n):
        train_idx = np.arange(n) != i
        C = np.median(proxies[train_idx] / damages[train_idx])
        pred = proxies[i] / C
        _, mape = mape_score(np.array([damages[i]]), np.array([pred]))
        rows.append({
            "filename": proxy_df.iloc[i]["filename"], "true": damages[i],
            "pred": pred, "rel_err_pct": mape * 100,
        })
    df = pd.DataFrame(rows)
    df.attrs["mean_score"] = max(0.0, 1 - df["rel_err_pct"].mean() / 100)
    return df


def build_train_proxy_range(proxy_df: pd.DataFrame) -> tuple[float, float]:
    return float(proxy_df["proxy"].min()), float(proxy_df["proxy"].max())


def flag_extrapolation(proxy: float, train_range: tuple[float, float]) -> dict:
    lo, hi = train_range
    if lo <= proxy <= hi:
        return {"flagged": False, "reason": "proxy is within Train's demonstrated range"}
    direction = "below" if proxy < lo else "above"
    return {
        "flagged": True,
        "reason": f"proxy is {direction} anything seen in Train (range: [{lo:.3e}, {hi:.3e}])",
    }
