"""
SHM Subsystem — Shared Feature Extraction
==========================================
Extracts a 27-feature vector from a single stress time-series file.
Each file is 581,120 rows × 1 column (no header), sampled at an unknown
but consistent rate.

Feature groups
--------------
Time-domain stats
    rms, std, mean, absmean, absmax, p2p, kurtosis, skewness, crest factor,
    percentile abs values (90th, 95th, 99th)

Frequency-domain stats (normalised FFT)
    Energy in each frequency quartile (q1–q4),
    spectral centroid, spectral spread

Rainflow cycle counting (downsampled 100×)
    n_cycles, weighted mean range, max range, range std,
    range 90th/99th percentile,
    Miner's damage proxies with exponents m=3, 5, 8
    (D ≈ Σ n_i · Δσ_i^m, the standard S-N fatigue law form)

Note on downsampling for rainflow
----------------------------------
581,120 samples is too large for rainflow directly; downsampling 100× to
~5,811 points retains the large cycle amplitudes that dominate fatigue
damage while keeping runtime fast.
"""

import numpy as np
import pandas as pd
import rainflow
from scipy import stats


def extract(sig: np.ndarray) -> dict:
    """
    Extract all features from a 1-D stress signal array.

    Parameters
    ----------
    sig : 1-D numpy array of stress values.

    Returns
    -------
    feats : Flat dict of feature_name → float.
    """
    feats: dict = {}

    # ------------------------------------------------------------------
    # Time-domain
    # ------------------------------------------------------------------
    rms = float(np.sqrt(np.mean(sig ** 2)))
    feats["rms"]     = rms
    feats["std"]     = float(np.std(sig))
    feats["mean"]    = float(np.mean(sig))
    feats["absmean"] = float(np.mean(np.abs(sig)))
    feats["absmax"]  = float(np.max(np.abs(sig)))
    feats["p2p"]     = float(np.max(sig) - np.min(sig))
    feats["kurt"]    = float(stats.kurtosis(sig))
    feats["skew"]    = float(stats.skew(sig))
    feats["crest"]   = float(np.max(np.abs(sig)) / (rms + 1e-9))

    for q in [90, 95, 99]:
        feats[f"pct{q}"] = float(np.percentile(np.abs(sig), q))

    # ------------------------------------------------------------------
    # Frequency-domain (normalised so amplitude doesn't dominate)
    # ------------------------------------------------------------------
    n        = len(sig)
    freqs    = np.fft.rfftfreq(n, d=1.0)          # normalised [0, 0.5]
    fft_mag  = np.abs(np.fft.rfft(sig))
    total_e  = float(np.sum(fft_mag ** 2)) + 1e-9  # total spectral energy

    q_bounds = np.percentile(freqs, [25, 50, 75, 100])
    prev     = freqs[0]
    for qi, qb in enumerate(q_bounds, start=1):
        mask = (freqs >= prev) & (freqs <= qb)
        feats[f"fft_q{qi}_energy"] = float(np.sum(fft_mag[mask] ** 2) / total_e)
        prev = qb

    # Spectral centroid and spread
    denom = float(np.sum(fft_mag)) + 1e-9
    sc    = float(np.sum(freqs * fft_mag) / denom)
    feats["spectral_centroid"] = sc
    feats["spectral_spread"]   = float(
        np.sqrt(np.sum((freqs - sc) ** 2 * fft_mag) / denom)
    )

    # ------------------------------------------------------------------
    # Rainflow cycle counting (downsampled)
    # ------------------------------------------------------------------
    sig_ds = sig[::100]                            # ~5,811 points
    cycles = list(rainflow.extract_cycles(sig_ds))

    if cycles:
        ranges = np.array([c[0] for c in cycles], dtype=float)  # stress range
        counts = np.array([c[1] for c in cycles], dtype=float)  # cycle count

        feats["rf_n_cycles"]   = float(len(cycles))
        feats["rf_range_mean"] = float(np.average(ranges, weights=counts))
        feats["rf_range_max"]  = float(np.max(ranges))
        feats["rf_range_std"]  = float(np.std(ranges))
        feats["rf_range_p90"]  = float(np.percentile(ranges, 90))
        feats["rf_range_p99"]  = float(np.percentile(ranges, 99))

        # Miner's rule damage proxies: D ≈ Σ n_i · Δσ_i^m
        for m in [3, 5, 8]:
            feats[f"rf_damage_proxy_m{m}"] = float(np.sum(counts * ranges ** m))
    else:
        for key in [
            "rf_n_cycles", "rf_range_mean", "rf_range_max", "rf_range_std",
            "rf_range_p90", "rf_range_p99",
            "rf_damage_proxy_m3", "rf_damage_proxy_m5", "rf_damage_proxy_m8",
        ]:
            feats[key] = 0.0

    return feats


def load_signal(path) -> np.ndarray:
    """Load a single-column headerless CSV and return a 1-D float array."""
    return pd.read_csv(path, header=None).values.flatten().astype(float)
