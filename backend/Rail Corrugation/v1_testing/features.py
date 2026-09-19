"""
Rail Corrugation — Shared Feature Extraction
=============================================
Extracts a 58-feature vector from a single 10,000-row sensor file.

Feature groups
--------------
- Time-domain stats per side (RMS, std, kurtosis, skewness, p2p, absmax, crest factor)
- Per-channel RMS spread (std and max across the 32 channels per side)
- FFT energy in four frequency bands (0-100, 100-500, 500-2000, 2000-5000 Hz)
- Cross-side ratios (vib RMS, vib kurtosis, shk RMS, vib p2p)
- Rotating speed (mean, std)

Side definitions (from the info kit)
--------------------------------------
  Side I  = odd bearing positions  (1, 3, 5, 7)
  Side II = even bearing positions (2, 4, 6, 8)
"""

import numpy as np
import pandas as pd
from scipy import stats


FS = 10_000  # sampling frequency (Hz)
ODD_POSITIONS  = [1, 3, 5, 7]
EVEN_POSITIONS = [2, 4, 6, 8]
FFT_BANDS = [(0, 100, "low"), (100, 500, "mid"), (500, 2000, "high"), (2000, 5000, "vhigh")]


def _side_cols(df: pd.DataFrame, signal: str, positions: list[int]) -> list[str]:
    """Return columns matching a signal type and a set of bearing positions."""
    return [
        c for c in df.columns
        if signal in c and any(f" position {p} of" in c for p in positions)
    ]


def _time_features(sig: np.ndarray, prefix: str) -> dict:
    rms = float(np.sqrt(np.mean(sig ** 2)))
    feats = {
        f"{prefix}_rms":    rms,
        f"{prefix}_std":    float(np.std(sig)),
        f"{prefix}_kurt":   float(stats.kurtosis(sig)),
        f"{prefix}_skew":   float(stats.skew(sig)),
        f"{prefix}_p2p":    float(np.max(sig) - np.min(sig)),
        f"{prefix}_absmax": float(np.max(np.abs(sig))),
        f"{prefix}_crest":  float(np.max(np.abs(sig)) / (rms + 1e-9)),
    }
    return feats


def _channel_spread(df: pd.DataFrame, cols: list[str], prefix: str) -> dict:
    """Per-channel RMS, then std and max across channels (captures spatial variation)."""
    per_ch = np.array([float(np.sqrt(np.mean(df[c].values ** 2))) for c in cols])
    return {
        f"{prefix}_ch_rms_std": float(np.std(per_ch)),
        f"{prefix}_ch_rms_max": float(np.max(per_ch)),
    }


def _fft_bands(sig: np.ndarray, prefix: str) -> dict:
    freqs   = np.fft.rfftfreq(len(sig), d=1.0 / FS)
    fft_mag = np.abs(np.fft.rfft(sig))
    feats   = {}
    for lo, hi, name in FFT_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"{prefix}_fft_{name}"] = float(np.sum(fft_mag[mask] ** 2))
    return feats


def extract(df: pd.DataFrame) -> dict:
    """
    Extract all features from a single training/test file DataFrame.

    Parameters
    ----------
    df : DataFrame with 10,000 rows and 129 sensor columns.

    Returns
    -------
    feats : Flat dict of feature_name → float.
    """
    feats: dict = {}

    for side_name, positions in [("s1", ODD_POSITIONS), ("s2", EVEN_POSITIONS)]:
        vib_cols = _side_cols(df, "Vibration", positions)
        shk_cols = _side_cols(df, "Shock",     positions)

        vib = df[vib_cols].values.flatten()
        shk = df[shk_cols].values.flatten()

        feats.update(_time_features(vib, f"{side_name}_vib"))
        feats.update(_time_features(shk, f"{side_name}_shk"))
        feats.update(_channel_spread(df, vib_cols, f"{side_name}_vib"))
        feats.update(_channel_spread(df, shk_cols, f"{side_name}_shk"))
        feats.update(_fft_bands(vib, f"{side_name}_vib"))
        feats.update(_fft_bands(shk, f"{side_name}_shk"))

    # Cross-side ratios — key discriminator between Side I and Side II
    feats["vib_rms_ratio"]  = feats["s1_vib_rms"]  / (feats["s2_vib_rms"]  + 1e-9)
    feats["vib_kurt_ratio"] = feats["s1_vib_kurt"] / (feats["s2_vib_kurt"] + 1e-9)
    feats["shk_rms_ratio"]  = feats["s1_shk_rms"]  / (feats["s2_shk_rms"]  + 1e-9)
    feats["vib_p2p_ratio"]  = feats["s1_vib_p2p"]  / (feats["s2_vib_p2p"]  + 1e-9)

    # Rotating speed
    feats["speed_mean"] = float(df["Rotating speed"].mean())
    feats["speed_std"]  = float(df["Rotating speed"].std())

    return feats
