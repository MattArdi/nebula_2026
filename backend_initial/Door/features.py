"""
Door Subsystem — Shared Feature Extraction
==========================================
Extracts a 22-feature vector from a single door cycle (segment).

Feature groups
--------------
Motor current
    mean, max, std, 90th/95th percentile, RMS, kurtosis, skewness,
    peak position (normalised), integral (work done by motor)

Motor voltage
    mean, max, std

Back-EMF
    mean, max, std

Resistance proxy  (V − back-EMF) / I   [only computed where I > 50 mA]
    mean, max, std, 90th/95th percentile
    Key discriminator: abnormal resistance → motor works harder (higher
    current) while calculated resistance appears lower (voltage also rises,
    masking the true increase — but mean/integral signals are clear).

Door leaf position
    range (max − min)

Metadata
    n_rows, operation (1=Close, 0=Open)
"""

import numpy as np
import pandas as pd
from scipy import stats


# Current threshold below which resistance is not computed (avoids /0 noise)
CURR_THRESHOLD = 50.0


def extract(d: pd.DataFrame, operation: str) -> dict:
    """
    Extract all features from one door cycle DataFrame slice.

    Parameters
    ----------
    d         : Sensor rows for a single cycle (from segmentation.detect_segments).
    operation : 'Close' or 'Open' (determines the 'operation' feature).

    Returns
    -------
    feats : Flat dict of feature_name → float.
    """
    curr = d["Motor current(mA)"].values.astype(float)
    volt = d["Motor Voltage(10mV)"].values.astype(float)
    emf  = d["Motor electrodynamic force"].values.astype(float)
    pos  = d["Door leaf position"].values.astype(float)

    feats: dict = {}

    # ------------------------------------------------------------------
    # Motor current
    # ------------------------------------------------------------------
    rms = float(np.sqrt(np.mean(curr ** 2)))
    feats["curr_mean"]      = float(curr.mean())
    feats["curr_max"]       = float(curr.max())
    feats["curr_std"]       = float(curr.std())
    feats["curr_p90"]       = float(np.percentile(curr, 90))
    feats["curr_p95"]       = float(np.percentile(curr, 95))
    feats["curr_rms"]       = rms
    feats["curr_kurt"]      = float(stats.kurtosis(curr))
    feats["curr_skew"]      = float(stats.skew(curr))
    feats["curr_peak_pos"]  = float(np.argmax(curr) / len(curr))
    feats["curr_integral"]  = float(np.trapezoid(curr))

    # ------------------------------------------------------------------
    # Motor voltage
    # ------------------------------------------------------------------
    feats["volt_mean"] = float(volt.mean())
    feats["volt_max"]  = float(volt.max())
    feats["volt_std"]  = float(volt.std())

    # ------------------------------------------------------------------
    # Back-EMF
    # ------------------------------------------------------------------
    feats["emf_mean"] = float(emf.mean())
    feats["emf_max"]  = float(emf.max())
    feats["emf_std"]  = float(emf.std())

    # ------------------------------------------------------------------
    # Resistance proxy: (V − back-EMF) / I
    # ------------------------------------------------------------------
    safe_curr = np.where(curr > CURR_THRESHOLD, curr, np.nan)
    resist    = (volt - emf) / safe_curr

    feats["resist_mean"] = float(np.nanmean(resist))
    feats["resist_max"]  = float(np.nanmax(resist))
    feats["resist_std"]  = float(np.nanstd(resist))
    feats["resist_p90"]  = float(np.nanpercentile(resist, 90))
    feats["resist_p95"]  = float(np.nanpercentile(resist, 95))

    # ------------------------------------------------------------------
    # Position and metadata
    # ------------------------------------------------------------------
    feats["pos_range"] = float(pos.max() - pos.min())
    feats["n_rows"]    = float(len(d))
    feats["operation"] = 1.0 if operation == "Close" else 0.0

    return feats
