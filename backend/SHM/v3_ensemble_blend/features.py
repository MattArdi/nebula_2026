"""
SHM Subsystem (v3, ensemble blend) — ML Feature Extraction
================================================================
Only used by the 3 ML members of the blend (LinearRegression and
BaggingRegressor(Ridge) use just the physics proxy column; ExtraTrees
uses the full 47-feature set here). The physics member itself
(model.py's PhysicsModel) doesn't use this file at all -- it goes
straight from physics.rainflow_cycles to the Miner's proxy, same as v2.

47 features per file: 17 time-domain summary stats, 12 rainflow-cycle-
derived stats, 14 Miner's-proxy variants (7 exponents x raw + log), 5
FFT-based stats. See algorithm.md Section 3 for the full list and why
each group is here.
"""

from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

import physics

PROXY_EXPONENTS = [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]


def extract_features(x: np.ndarray, cycles: list) -> dict:
    ranges = np.array([c[0] for c in cycles]) if cycles else np.array([0.0])
    means = np.array([c[1] for c in cycles]) if cycles else np.array([0.0])
    counts = np.array([c[2] for c in cycles]) if cycles else np.array([0.0])

    feats: dict = {}

    # time-domain (17)
    feats["td_mean"] = float(np.mean(x))
    feats["td_std"] = float(np.std(x))
    feats["td_rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats["td_min"] = float(np.min(x))
    feats["td_max"] = float(np.max(x))
    feats["td_ptp"] = float(np.ptp(x))
    feats["td_skew"] = float(skew(x))
    feats["td_kurt"] = float(kurtosis(x, fisher=True, bias=False))
    for p in [5, 25, 50, 75, 95, 99]:
        feats[f"td_p{p}"] = float(np.percentile(x, p))
    feats["td_crest"] = feats["td_max"] / (feats["td_rms"] + 1e-9)
    feats["td_zero_cross_rate"] = float(np.mean(np.diff(np.sign(x - np.mean(x))) != 0))

    # rainflow-cycle-derived (11)
    feats["rf_n_cycles"] = float(len(ranges))
    feats["rf_range_mean"] = float(np.mean(ranges))
    feats["rf_range_max"] = float(np.max(ranges))
    feats["rf_range_std"] = float(np.std(ranges))
    for p in [50, 75, 90, 95, 99]:
        feats[f"rf_range_p{p}"] = float(np.percentile(ranges, p))
    feats["rf_count_sum"] = float(np.sum(counts))
    feats["rf_mean_stress_mean"] = float(np.mean(means))
    feats["rf_mean_stress_std"] = float(np.std(means))

    # Miner's-proxy variants (14) -- proxy_m5.0 is the one the physics
    # model and the linear/bagged-ridge ensemble members use directly.
    for m in PROXY_EXPONENTS:
        feats[f"proxy_m{m}"] = float(np.sum(counts * ranges ** m))
        feats[f"log_proxy_m{m}"] = float(np.log(feats[f"proxy_m{m}"] + 1.0))

    # FFT-based (5)
    n = len(x)
    fft_mag = np.abs(np.fft.rfft(x - np.mean(x)))
    freqs = np.fft.rfftfreq(n)
    power = fft_mag ** 2
    total_power = power.sum() + 1e-12
    feats["fft_dominant_freq"] = float(freqs[np.argmax(power)])
    feats["fft_spectral_centroid"] = float(np.sum(freqs * power) / total_power)
    feats["fft_total_power"] = float(total_power)
    feats["fft_top1_frac"] = float(np.sort(power)[-1] / total_power)
    feats["fft_top5_frac"] = float(np.sum(np.sort(power)[-5:]) / total_power)

    return feats


def extract_from_path(path: Path) -> dict:
    x = physics.load_stress_series(path)
    cycles = physics.rainflow_cycles(x)
    return extract_features(x, cycles)


def _feature_names() -> list[str]:
    rng = np.random.RandomState(0)
    dummy_x = rng.normal(size=2000)
    dummy_cycles = physics.rainflow_cycles(dummy_x)
    return list(extract_features(dummy_x, dummy_cycles).keys())


FEATURE_NAMES = _feature_names()
PHYSICS_PROXY_COL = "proxy_m5.0"
