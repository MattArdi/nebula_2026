"""
SHM Subsystem (v3, incremental) — Diagnostics
==================================================
Same leave-one-out self-check and extrapolation flag as v2
(`v2_rule-based/diagnostics.py`), plus one addition needed for the
cycles-to-failure confidence interval: `fit_log_residual_sigma()`,
which turns the LOO self-check's per-file errors into the scatter
parameter that interval uses.
"""

import numpy as np
import pandas as pd


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mape = float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))
    return max(0.0, 1 - mape), mape


def self_check_loo(proxy_df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out CV of the calibration fit, from a precomputed proxy table.
    Unchanged from v2 except that `proxy_df` here also carries `n_cycles`
    (passed through untouched) for the caller's convenience."""
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


def fit_log_residual_sigma(loo_df: pd.DataFrame) -> float:
    """
    Scatter parameter for the damage-prediction confidence interval,
    estimated the same way ASTM E739 estimates fatigue-life scatter: as
    the standard deviation of log(true / predicted) (log life is assumed
    normally distributed -- see model.py's confidence-interval functions
    for the full derivation and citations).

    This is an approximation forced by what data actually exists here:
    ASTM E739 assumes repeated fatigue tests at the same stress level to
    estimate this scatter directly. This dataset has none of that --
    every file is a different, unrepeated loading history. What we do
    have is this pipeline's own leave-one-out residuals against
    Train_Labels.csv (64 independent files), which is the best available
    empirical stand-in: it is the actual observed spread between this
    model's damage predictions and ground truth, honestly held out
    (C is refit on the other 63 files for each fold).
    """
    resid = np.log(loo_df["true"].values / loo_df["pred"].values)
    return float(np.std(resid, ddof=1))


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
