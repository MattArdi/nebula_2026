"""
Rail Corrugation Subsystem (v3, frequency-augmented) — Feature Extraction
============================================================================
Extracts a 49-feature vector from a single raw sensor file (10,000 rows x
129 columns): column 0 is a 0/1 rotating-speed pulse train, columns 1-128
are Vibration/Shock readings for 8 bearing positions x 8 cars, interleaved
(vib, shock, vib, shock, ...).

This is v2's exact 41 time-domain features (unchanged -- see algorithm.md
Section 3.1 for the full derivation) plus 8 new ones: FFT energy in the
0-100 Hz band, per (car, position), aggregated side1/side2 x mean/max the
same way every other stat here is. See algorithm.md Section 3.1 for why
this band specifically, and Section 5 for the real, validated numbers
behind that choice (+0.05 macro F1 in repeated CV, confirmed on a second,
independent set of CV seeds).

Side definitions (from the info kit) -- this is what "Side I" / "Side II"
physically mean for this subsystem:
    Side I  = odd bearing positions  (1, 3, 5, 7)
    Side II = even bearing positions (2, 4, 6, 8)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

N_CARS = 8
N_POS = 8

# 0-indexed position offsets within each car's 8 positions.
SIDE1_IDX = [0, 2, 4, 6]  # bearing positions 1, 3, 5, 7 -> Side I
SIDE2_IDX = [1, 3, 5, 7]  # bearing positions 2, 4, 6, 8 -> Side II

# Encoder assumption behind speed_kmh -- see algorithm.md Section 3.1.
PULSES_PER_REV = 90
WHEEL_DIAMETER_M = 0.85

# Confirmed against the info kit (Rail_Corrugation_Info_Kit.md Section 2.1):
# "The sampling frequency is 10,000 Hz, each file has a duration of 1 s."
# Not an assumption -- this is the one frequency-domain feature group that
# needed it confirmed before being trusted, and it checks out exactly.
FS_HZ = 10_000
LOW_BAND_HZ = (0, 100)  # see algorithm.md Section 3.1 for why this band


def load_file(path) -> tuple[float, np.ndarray]:
    """
    Load one raw sensor CSV and split it into (speed_kmh, sensor_matrix).

    sensor_matrix has shape (10000, 8 car, 8 pos, 2 kind), kind 0 =
    vibration, kind 1 = shock.
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    speed_pulse = arr[:, 0]
    transitions = np.sum(np.abs(np.diff(speed_pulse)) > 0.5)
    speed_kmh = (transitions / 2 / PULSES_PER_REV) * np.pi * WHEEL_DIAMETER_M * 3.6

    rest = arr[:, 1:]  # 128 cols: car(8) x pos(8) x kind(2: vib, shock)
    rest = rest.reshape(rest.shape[0], N_CARS, N_POS, 2)
    return float(speed_kmh), rest


def _stats(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rms, kurtosis (Fisher, bias-corrected), crest factor, peak-to-peak, along axis."""
    rms = np.sqrt(np.mean(x ** 2, axis=axis))
    kurt = kurtosis(x, axis=axis, fisher=True, bias=False)
    peak = np.max(np.abs(x), axis=axis)
    crest = peak / (rms + 1e-9)
    ptp = np.ptp(x, axis=axis)
    return rms, kurt, crest, ptp


def _low_band_energy(sig: np.ndarray) -> np.ndarray:
    """
    FFT energy in LOW_BAND_HZ, per channel. sig: (10000, 8 car, 8 pos).
    Returns (8 car, 8 pos).
    """
    sig_flat = sig.reshape(sig.shape[0], -1)  # (10000, 64)
    freqs = np.fft.rfftfreq(sig_flat.shape[0], d=1.0 / FS_HZ)
    mag2 = np.abs(np.fft.rfft(sig_flat, axis=0)) ** 2  # (n_freqs, 64)
    lo, hi = LOW_BAND_HZ
    mask = (freqs >= lo) & (freqs < hi)
    energy = mag2[mask].sum(axis=0)  # (64,)
    return energy.reshape(N_CARS, N_POS)


def extract_features(speed_kmh: float, sensor_matrix: np.ndarray) -> dict:
    """
    Build the 49-feature dict from (speed_kmh, sensor_matrix) as returned
    by load_file(). See algorithm.md Section 3.1 for the full derivation.
    """
    vib = sensor_matrix[:, :, :, 0]
    shock = sensor_matrix[:, :, :, 1]

    vib_rms, vib_kurt, vib_crest, vib_ptp = _stats(vib)      # each (8 car, 8 pos)
    shock_rms, shock_kurt, shock_crest, shock_ptp = _stats(shock)

    feats: dict = {}
    for kind_name, (rms, kurt, crest, ptp) in [
        ("vib", (vib_rms, vib_kurt, vib_crest, vib_ptp)),
        ("shock", (shock_rms, shock_kurt, shock_crest, shock_ptp)),
    ]:
        for stat_name, mat in [("rms", rms), ("kurt", kurt), ("crest", crest), ("ptp", ptp)]:
            side1 = mat[:, SIDE1_IDX]
            side2 = mat[:, SIDE2_IDX]
            feats[f"side1_{kind_name}_{stat_name}_mean"] = float(np.mean(side1))
            feats[f"side1_{kind_name}_{stat_name}_max"] = float(np.max(side1))
            feats[f"side2_{kind_name}_{stat_name}_mean"] = float(np.mean(side2))
            feats[f"side2_{kind_name}_{stat_name}_max"] = float(np.max(side2))
    # 2 sides x 2 kinds x 4 stats x 2 aggregations = 32 features so far.

    feats["speed_kmh"] = speed_kmh

    # Speed-normalized variants of every RMS feature (8 base RMS features).
    for k in [k for k in feats if k.endswith("_rms_mean") or k.endswith("_rms_max")]:
        feats[f"{k}_norm"] = feats[k] / (speed_kmh + 1)

    # 32 + 8 + speed_kmh = 41 features -- identical to v2 up to this point.

    # New in v3: low-band (0-100 Hz) FFT energy, per (car, position), same
    # side1/side2 x mean/max aggregation as every stat above.
    for kind_name, sig in [("vib", vib), ("shock", shock)]:
        low = _low_band_energy(sig)  # (8 car, 8 pos)
        side1 = low[:, SIDE1_IDX]
        side2 = low[:, SIDE2_IDX]
        feats[f"side1_{kind_name}_fft_low_mean"] = float(np.mean(side1))
        feats[f"side1_{kind_name}_fft_low_max"] = float(np.max(side1))
        feats[f"side2_{kind_name}_fft_low_mean"] = float(np.mean(side2))
        feats[f"side2_{kind_name}_fft_low_max"] = float(np.max(side2))
    # 41 + 8 = 49 total.

    return feats


def extract_from_path(path) -> dict:
    speed_kmh, sensor_matrix = load_file(path)
    return extract_features(speed_kmh, sensor_matrix)


def worst_offender(sensor_matrix: np.ndarray) -> dict:
    """
    Diagnostics only -- never fed to the model. Identical to v2: per side,
    the single (car, position) whose vibration RMS is highest -- still a
    CatBoost/XGBoost top feature here too (see algorithm.md Section 2).
    """
    vib = sensor_matrix[:, :, :, 0]  # (10000, 8 car, 8 pos)
    vib_rms = np.sqrt(np.mean(vib ** 2, axis=0))  # (8 car, 8 pos)

    out = {}
    for side_name, pos_idx in [("side1", SIDE1_IDX), ("side2", SIDE2_IDX)]:
        sub = vib_rms[:, pos_idx]  # (8 car, 4 pos)
        car_i, pos_j = np.unravel_index(np.argmax(sub), sub.shape)
        out[side_name] = {
            "car": int(car_i) + 1,
            "position": int(pos_idx[pos_j]) + 1,
            "vib_rms": float(sub[car_i, pos_j]),
        }
    return out


def _feature_names() -> list[str]:
    """Feature key order, derived once from a small dummy (non-degenerate) input."""
    rng = np.random.RandomState(0)
    dummy = rng.normal(size=(8, N_CARS, N_POS, 2))
    return list(extract_features(1.0, dummy).keys())


FEATURE_NAMES = _feature_names()


def build_feature_matrix(paths: list[Path], verbose: bool = True) -> pd.DataFrame:
    """Extract features for a list of file paths, one row per file."""
    rows = []
    for i, p in enumerate(paths):
        rows.append(extract_from_path(p))
        if verbose and (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(paths)} files...")
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
