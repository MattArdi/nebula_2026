"""
SHM Subsystem (rule-based) — Calibration & Prediction
==========================================================
Fits the single free parameter of the physics model (the S-N curve's
scale constant C) from labelled training data, and predicts cumulative
damage for a new file from its Miner's-rule proxy alone.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import physics


def fit_calibration_constant(proxy_df: pd.DataFrame) -> float:
    """
    C such that damage ~= proxy / C, fit as the median of (proxy / damage)
    across every labelled training file (from physics.compute_train_proxies).
    Median rather than a least-squares fit because it isn't dominated by the
    handful of large-damage files -- see algorithm.md Section 2 for why that
    matters for this metric.
    """
    return float(np.median(proxy_df["proxy"] / proxy_df["damage"]))


def predict_damage(x: np.ndarray, C: float, m: float = physics.M_EXPONENT) -> float:
    """Predicted cumulative damage for one file's raw stress series."""
    return physics.miner_proxy(x, m) / C


def save_calibration(
    path: Path, C: float, m: float, train_proxy_range: tuple[float, float],
    loo_mape_pct: float | None = None,
) -> None:
    """Serialize the single fitted constant C, the fixed exponent m, the
    train proxy range (for extrapolation flagging), and -- if the caller
    already ran the leave-one-out self-check -- its mean absolute
    percentage error, so a fresh run_pipeline.py can report a prediction's
    error band without ever re-running LOO against training data. Optional
    and backward compatible: an artifact saved without it, or loaded by
    older code, works exactly as before."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "C": C, "m": m,
        "train_proxy_min": train_proxy_range[0], "train_proxy_max": train_proxy_range[1],
    }
    if loo_mape_pct is not None:
        payload["loo_mape_pct"] = loo_mape_pct
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> dict:
    return json.loads(Path(path).read_text())
