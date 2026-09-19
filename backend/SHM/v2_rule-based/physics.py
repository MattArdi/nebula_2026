"""
SHM Subsystem (rule-based) — Rainflow Feature Extraction
============================================================
Loads a raw dynamic-stress CSV and reduces it to the single quantity the
prediction model needs: a Miner's-rule proxy computed from real rainflow
cycle counting (the ASTM E1049 algorithm, via the `rainflow` package) --
not manual counting, not a hand-approximated summary statistic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rainflow

M_EXPONENT = 5.0   # S-N curve exponent -- see model.py and algorithm.md


def load_stress_series(path) -> np.ndarray:
    """Load a raw single-column stress CSV (no header) as a 1-D array."""
    return np.loadtxt(path)


def rainflow_cycles(x: np.ndarray):
    """Extract (range, mean, count, i_start, i_end) tuples via ASTM-standard rainflow counting."""
    return list(rainflow.extract_cycles(x))


def miner_proxy(x: np.ndarray, m: float = M_EXPONENT) -> float:
    """
    Sum(n_i * range_i^m) over all rainflow cycles -- proportional to Miner's-rule
    cumulative damage for a power-law S-N curve, up to the unknown material
    constant C (damage = proxy / C).
    """
    cycles = rainflow_cycles(x)
    if not cycles:
        return 0.0
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    return float(np.sum(counts * ranges**m))


def compute_train_proxies(data_dir: Path, m: float = M_EXPONENT) -> pd.DataFrame:
    """
    Rainflow + Miner's-rule proxy for every labelled training file, computed
    once. Both the leave-one-out self-check and the final calibration fit
    need this same table -- computing it once and sharing it (rather than
    each recomputing it independently) roughly halves total runtime, since
    rainflow counting is the pipeline's only real computational cost.
    """
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    rows = []
    for _, row in labels.iterrows():
        x = load_stress_series(data_dir / "Train" / row["filename"])
        rows.append({"filename": row["filename"], "proxy": miner_proxy(x, m), "damage": row["damage"]})
    return pd.DataFrame(rows)
