"""
SHM Subsystem (v3, incremental) — Calibration, Prediction & Cycles-to-Failure
================================================================================
Same single-parameter calibration as v2 (`v2_rule-based/model.py`): fit
C such that damage ~= proxy / C, as the median of proxy/damage across
training files. Two additions on top of that:

1. Cycles-to-failure / cycles-remaining, from the current damage estimate
   and the number of rainflow cycles it was accumulated over, under a
   stationary-loading extrapolation (Palmgren-Miner rule: damage
   accumulates linearly in cycles under an unchanging load spectrum, so
   the file's own damage-per-cycle rate projects forward).

2. A confidence interval on that cycles-to-failure number, adapted from
   the standard statistical treatment of S-N fatigue-life scatter
   (ASTM E739: log(life) ~ Normal, constant variance -> symmetric
   z * sigma bounds in log-space). See `damage_confidence_interval()`
   docstring for the exact derivation, citations, and what is and is not
   captured by the interval.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import physics

DEFAULT_D_FAIL = 1.0    # Miner's-rule failure threshold, D = sum(n_i/N_i) = 1 by convention
DEFAULT_CONFIDENCE = 0.90


def fit_calibration_constant(proxy_df: pd.DataFrame) -> float:
    """C such that damage ~= proxy / C -- unchanged from v2."""
    return float(np.median(proxy_df["proxy"] / proxy_df["damage"]))


def predict_damage(proxy: float, C: float) -> float:
    """Predicted cumulative damage from an already-computed Miner's-rule proxy."""
    return proxy / C


def cycles_to_failure(n_cycles: float, damage: float, d_fail: float = DEFAULT_D_FAIL) -> float:
    """
    Stationary-loading extrapolation: if `n_cycles` rainflow cycles have
    produced `damage` so far, and the loading spectrum that produced them
    keeps recurring unchanged, damage accumulates linearly in cycles
    (that is what Miner's rule assumes), so failure (damage = d_fail)
    is reached at `n_cycles * d_fail / damage` cycles.

    This is a real, standard engineering extrapolation, not something
    invented for this codebase -- see e.g. "Understanding Miner's Rule"
    (Regal Rexnord, https://www.regalrexnord.com/en/regal-rexnord-insights/what-is-miners-rule)
    and "Miner Rule, Cumulative Fatigue Damage and Load Spectra"
    (https://atlasofengineering.com/materials-engineering/miner-rule/).
    Two assumptions it carries, stated plainly rather than hidden:
      - The recorded loading spectrum continues unchanged (stationarity).
        Nothing in a single file can validate this; it is the price of
        extrapolating from one snapshot instead of a run-to-failure history.
      - d_fail = 1.0 is the textbook Miner's-rule convention, but
        experimentally observed critical damage sums for real materials
        range roughly 0.7-2.2 depending on material and load spectrum
        (ReliaSoft, https://help.reliasoft.com/articles/content/hotwire/issue116/hottopics116.htm).
        That extra scatter is NOT included in the confidence interval
        below -- it is a separate, uncaptured source of uncertainty.
    Returns float('inf') if damage is 0 or negative (no cycles closed yet).
    """
    if damage <= 0:
        return float("inf")
    return n_cycles * d_fail / damage


def cycles_remaining(n_cycles: float, damage: float, d_fail: float = DEFAULT_D_FAIL) -> float:
    """Cycles still to run before failure, under the same stationary-loading
    extrapolation as `cycles_to_failure()`."""
    total = cycles_to_failure(n_cycles, damage, d_fail)
    return total if total == float("inf") else total - n_cycles


def damage_confidence_interval(
    damage: float, sigma_log: float, confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """
    Confidence interval on the damage estimate itself, which the cycles-
    to-failure interval below is derived from.

    Method: fatigue life (equivalently, damage at a fixed cycle count) is
    standardly treated as log-normally distributed with roughly constant
    log-variance -- this is the ASTM E739 statistical model for S-N/e-N
    fatigue data (log(N) ~ Normal(mu, sigma^2), sigma assumed constant
    across the tested range; confidence bounds are then mu +/- z*sigma in
    log space). Sources: ASTM E739 "Standard Practice for Statistical
    Analysis of ... Fatigue Data"; summarized in "Probabilistic S-N
    fields based on statistical distributions" (Barbosa et al. 2019,
    https://journals.sagepub.com/doi/10.1177/1687814019870395), which
    states the log-normal-life / constant-variance assumption explicitly.

    What's adapted for this codebase: ASTM E739's sigma comes from
    repeated fatigue tests at a fixed stress level, which this dataset
    does not have (every file is one unrepeated loading history). Instead
    `sigma_log` here is fit empirically from this pipeline's own
    leave-one-out residuals against Train_Labels.csv (see
    `diagnostics.fit_log_residual_sigma`) -- the actual observed
    prediction scatter on held-out files, honestly cross-validated,
    substituting for a same-stress-level replicate scatter this dataset
    doesn't provide.

    Returns (damage_lo, damage_hi) for the given two-sided confidence
    level (default 90%, i.e. z ~= 1.645).
    """
    z = stats.norm.ppf(0.5 + confidence / 2)
    lo = damage * np.exp(-z * sigma_log)
    hi = damage * np.exp(z * sigma_log)
    return float(lo), float(hi)


def cycles_to_failure_interval(
    n_cycles: float, damage: float, sigma_log: float,
    d_fail: float = DEFAULT_D_FAIL, confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """
    Confidence interval on cycles-to-failure, propagated from the damage
    interval above. Because cycles_to_failure = n_cycles * d_fail / damage
    is a pure inverse relationship, log(cycles_to_failure) = const -
    log(damage): the SAME sigma_log applies, just with the bounds
    flipped (a damage underestimate implies a life overestimate).
    d_fail's own uncertainty (see `cycles_to_failure` docstring) is not
    propagated here -- only the damage-estimation scatter is.
    """
    damage_lo, damage_hi = damage_confidence_interval(damage, sigma_log, confidence)
    life_hi = cycles_to_failure(n_cycles, damage_lo, d_fail)   # lower damage -> longer life
    life_lo = cycles_to_failure(n_cycles, damage_hi, d_fail)   # higher damage -> shorter life
    return life_lo, life_hi


def save_calibration(
    path: Path, C: float, m: float, train_proxy_range: tuple[float, float], sigma_log: float,
) -> None:
    """Serialize everything a fresh run_pipeline.py needs to predict
    damage, cycles-to-failure, and its confidence interval without ever
    seeing training data again: C, the fixed exponent m, the train proxy
    range (extrapolation flagging), and the fitted log-scatter sigma
    (confidence intervals)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "C": C, "m": m,
        "train_proxy_min": train_proxy_range[0], "train_proxy_max": train_proxy_range[1],
        "sigma_log": sigma_log,
    }, indent=2))


def load_calibration(path: Path) -> dict:
    return json.loads(Path(path).read_text())
