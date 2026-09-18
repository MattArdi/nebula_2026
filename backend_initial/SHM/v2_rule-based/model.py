"""
SHM Subsystem (rule-based) — Calibration & Prediction
==========================================================
Fits the single free parameter of the physics model (the S-N curve's
scale constant C) from labelled training data, and predicts cumulative
damage for a new file from its Miner's-rule proxy alone.
"""

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
