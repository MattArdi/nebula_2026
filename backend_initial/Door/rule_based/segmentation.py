"""
Door Subsystem (rule-based) — Segmentation
============================================
Splits a continuous door sensor stream into individual open/close cycles.

Segmentation rule
------------------
Train.csv / Test.csv are not truly continuous streams: they are individual
door cycles concatenated back-to-back with the idle time between them
stripped out. Every row-to-row timestamp delta is either exactly 20ms
(inside a cycle, 50 Hz sampling) or 10+ seconds (between cycles) — nothing
in between. A cycle boundary is therefore just a gap in the timestamp
column bigger than a threshold sitting comfortably between those two
regimes.

This was cross-checked against a more elaborate position/current-based
rule (direction reversal, current dropping from >1500 mA to <200 mA) and
the two produce IDENTICAL boundaries on both Train.csv and Test.csv — the
elaborate rule was implicitly keying off the same timestamp discontinuity,
not an independent sensor signature. The gap rule is used here because
it's what is actually doing the work, and its margin (20ms vs 10s+) is
enormous, so it is not sensitive to the exact threshold chosen.
"""

import numpy as np
import pandas as pd

GAP_THRESHOLD_S = 0.1   # boundary if the timestamp delta exceeds this
MIN_CYCLE_ROWS = 50     # guardrail: candidate shorter than this is implausible
MAX_CYCLE_ROWS = 250    # guardrail: candidate longer than this is implausible


def parse_ts(s: str) -> pd.Timestamp:
    """Parse 'YYYY-M-D-H-M-S-ms' (not zero-padded) into a Timestamp."""
    y, mo, d, h, mi, se, ms = map(int, s.split("-"))
    return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=se, microsecond=ms * 1000)


def load_sensor_csv(path) -> pd.DataFrame:
    """Load a raw Door sensor CSV (Train.csv / Test.csv) and add a 'ts' column."""
    df = pd.read_csv(path)
    df["ts"] = df["Datetime"].apply(parse_ts)
    return df


def detect_cycles(df: pd.DataFrame, gap_threshold_s: float = GAP_THRESHOLD_S) -> list[tuple[int, int]]:
    """
    Split df into cycles by timestamp gap.

    Returns
    -------
    List of (start_idx, end_idx) inclusive row-index pairs, one per cycle.
    """
    dt = df["ts"].diff().dt.total_seconds().fillna(0.0).values
    boundary_rows = np.where(dt > gap_threshold_s)[0].tolist()
    starts = [0] + boundary_rows
    ends = [b - 1 for b in boundary_rows] + [len(df) - 1]
    return list(zip(starts, ends))


def flag_implausible(
    segments: list[tuple[int, int]],
    min_len: int = MIN_CYCLE_ROWS,
    max_len: int = MAX_CYCLE_ROWS,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split segments into (plausible, implausible) by row count."""
    plausible, implausible = [], []
    for s, e in segments:
        n = e - s + 1
        (plausible if min_len <= n <= max_len else implausible).append((s, e))
    return plausible, implausible
