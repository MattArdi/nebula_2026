"""
Door Subsystem — Shared Segmentation Logic
==========================================
Handles timestamp parsing and the rule-based cycle detection that splits
a continuous door sensor stream into individual open/close cycles.

Timestamp format
----------------
'2023-7-5-0-0-23-999'  →  year-month-day-hour-min-sec-millisecond
Sampling interval: 20 ms (50 Hz)

Segmentation rules
------------------
Every row in the stream belongs to exactly one cycle (no gaps between
segments in the training data). A new cycle boundary is detected when
ANY of the following conditions holds:

1. Position jump > 200 units
   → The door reverses direction (Open→Close or Close→Open).

2. Motor current drops from >1500 mA to <200 mA AND command signal changes
   → The motor stops and a new command is issued (consecutive opposite cycles
     where the position doesn't jump because the door is already at the limit).

3. Motor current drops from >1500 mA to <200 mA AND position resets to
   near 0 or near 700 AND command does NOT change
   → Consecutive same-direction cycles (e.g. two Close cycles in a row).

All three conditions were validated against the 110-segment training answer
file; they recover all 110 boundaries with zero false positives and zero
false negatives.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def parse_ts_ms(ts: str) -> int:
    """
    Convert a door timestamp string to total milliseconds from midnight.

    Parameters
    ----------
    ts : Timestamp string, e.g. '2023-7-5-0-0-23-999'.

    Returns
    -------
    ms : Milliseconds from midnight (int).
    """
    p = ts.split("-")
    hour, minute, sec, ms = int(p[3]), int(p[4]), int(p[5]), int(p[6])
    return hour * 3_600_000 + minute * 60_000 + sec * 1_000 + ms


def add_timestamp_col(df: pd.DataFrame, col: str = "ts_ms") -> pd.DataFrame:
    """Add a 'ts_ms' column parsed from the 'Datetime' column."""
    df = df.copy()
    df[col] = df["Datetime"].apply(parse_ts_ms)
    return df


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def detect_segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Split a continuous door sensor stream into (start_idx, end_idx) pairs.

    The returned indices are inclusive: df.iloc[start_idx : end_idx + 1]
    gives the full cycle.

    Parameters
    ----------
    df : Raw sensor DataFrame (must contain 'Door leaf position',
         'Motor current(mA)', 'Close command', 'Open command').

    Returns
    -------
    segments : List of (start_idx, end_idx) tuples, one per detected cycle.
    """
    pos       = df["Door leaf position"].values.astype(float)
    curr      = df["Motor current(mA)"].values.astype(float)
    close_cmd = df["Close command"].values
    open_cmd  = df["Open command"].values

    # Previous-row quantities (roll shifts array right by 1)
    curr_prev = np.roll(curr, 1);      curr_prev[0] = curr[0]
    cmd       = close_cmd + open_cmd   # 1 when any command is active
    cmd_prev  = np.roll(cmd, 1);       cmd_prev[0]  = cmd[0]

    pos_diff   = np.diff(pos, prepend=pos[0])
    cmd_change = cmd != cmd_prev
    pos_reset  = (pos > 690) | (pos < 10)  # near start of a new cycle

    # Boundary conditions
    cond1 = np.abs(pos_diff) > 200                              # direction reversal
    cond2 = (curr_prev > 1500) & (curr < 200) & cmd_change     # opposite consecutive
    cond3 = (curr_prev > 1500) & (curr < 200) & pos_reset & ~cmd_change  # same-dir

    is_boundary = cond1 | cond2 | cond3
    is_boundary[0] = True  # first row is always a segment start

    boundary_indices = np.where(is_boundary)[0].tolist()
    boundary_indices.append(len(df))  # sentinel

    segments = [
        (boundary_indices[i], boundary_indices[i + 1] - 1)
        for i in range(len(boundary_indices) - 1)
    ]
    return segments


def segment_operation(d: pd.DataFrame) -> str:
    """
    Determine whether a segment is a Close or Open cycle.

    Parameters
    ----------
    d : Slice of the sensor DataFrame for one cycle.

    Returns
    -------
    'Close' or 'Open'
    """
    return "Close" if d["Close command"].mean() > 0.5 else "Open"
