"""
SHM Subsystem (v3, ensemble blend) — Diagnostics
=====================================================
Leave-one-out cross-validation of the full 4-member blend (model.py),
scored with the real competition metric (max(0, 1-MAPE)) -- same protocol
as v2's diagnostics.py, extended to refit all 4 ensemble members (not
just the physics constant) on the other 63 files every fold. This is
what produced the 0.9750 LOO score cited in model.py and algorithm.md;
running this against a fresh feature/label table should reproduce it.
"""

import numpy as np
import pandas as pd

import model as model_mod


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mape = float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))
    return max(0.0, 1 - mape), mape


def self_check_loo(feature_rows: list[dict], damages: np.ndarray, filenames: list[str]) -> pd.DataFrame:
    n = len(damages)
    rows = []
    for i in range(n):
        train_rows = [feature_rows[j] for j in range(n) if j != i]
        train_damages = np.delete(damages, i)

        blend = model_mod.EnsembleBlend().fit(train_rows, train_damages)
        pred = blend.predict([feature_rows[i]])[0]
        _, mape = mape_score(np.array([damages[i]]), np.array([pred]))
        rows.append({"filename": filenames[i], "true": damages[i], "pred": pred, "rel_err_pct": mape * 100})

    df = pd.DataFrame(rows)
    df.attrs["mean_score"] = max(0.0, 1 - df["rel_err_pct"].mean() / 100)
    return df


def build_train_proxy_range(feature_rows: list[dict]) -> tuple[float, float]:
    import features as feat_mod
    proxies = [row[feat_mod.PHYSICS_PROXY_COL] for row in feature_rows]
    return float(min(proxies)), float(max(proxies))


def flag_extrapolation(proxy: float, train_range: tuple[float, float]) -> dict:
    lo, hi = train_range
    if lo <= proxy <= hi:
        return {"flagged": False, "reason": "proxy is within Train's demonstrated range"}
    direction = "below" if proxy < lo else "above"
    return {
        "flagged": True,
        "reason": f"proxy is {direction} anything seen in Train (range: [{lo:.3e}, {hi:.3e}])",
    }
