"""
ACV Subsystem (rule-based) — Schema Detection & Deviation Extraction
========================================================================
Each case file has its own column headers — the info kit warns explicitly
that the parameter set differs between files, and this was confirmed
directly: 5 of 6 training files use a "standard" 67-column schema (8
params x 8 cars), one (acv_case_04.xlsx) uses a much richer ~483-column
schema (60+ params x 8 cars) with direct refrigeration-circuit telemetry
instead of just temperature.

This module never assumes a fixed column list. It parses whatever headers
are actually present and picks the best available "deviation from target"
signal per car:

  Standard schema : Indoor Average Temperature - ACV Control Temperature (Cooling)
  Rich schema      : Passenger Cabin Temperature Detected Value - Target Temperature Value

A car with no usable columns at all (confirmed to happen: cars 05-08 are
100% NaN across every parameter in acv_case_04.xlsx) returns None rather
than a fabricated zero — ranking.py is responsible for placing such cars
last, not for inventing a signal that doesn't exist.
"""

import re

import pandas as pd

CAR_COL_RE = re.compile(r"Car (\d+) - (.+)")

STANDARD_INDOOR = "Indoor Average Temperature"
STANDARD_SETPOINT = "ACV Control Temperature (Cooling)"
RICH_INDOOR = "Passenger Cabin Temperature Detected Value"
RICH_SETPOINT = "Target Temperature Value"


def parse_car_columns(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Map car identifier -> {parameter name: column name}, using each car's
    identifier exactly as it appears in the file's own headers (e.g. '03',
    not '3' or 'Car 3') -- this is also the identifier the submission's
    ranked_cars field must use.
    """
    out: dict[str, dict[str, str]] = {}
    for col in df.columns:
        m = CAR_COL_RE.match(col)
        if m:
            car, param = m.group(1), m.group(2)
            out.setdefault(car, {})[param] = col
    return out


def get_deviation_series(df: pd.DataFrame, car_cols: dict[str, str]) -> pd.Series | None:
    """
    Return this car's (actual - target) temperature series, trying the
    standard schema's columns first and falling back to the rich schema's.

    Returns None if neither schema's columns are usable for this car
    (either missing entirely, or present but 100% NaN).
    """
    if STANDARD_INDOOR in car_cols and STANDARD_SETPOINT in car_cols:
        indoor = pd.to_numeric(df[car_cols[STANDARD_INDOOR]], errors="coerce")
        setpoint = pd.to_numeric(df[car_cols[STANDARD_SETPOINT]], errors="coerce")
    elif RICH_INDOOR in car_cols and RICH_SETPOINT in car_cols:
        indoor = pd.to_numeric(df[car_cols[RICH_INDOOR]], errors="coerce")
        setpoint = pd.to_numeric(df[car_cols[RICH_SETPOINT]], errors="coerce")
    else:
        return None

    dev = indoor - setpoint
    if dev.notna().sum() == 0:
        return None
    return dev


def build_deviation_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the full car x time deviation matrix for a case file.

    Returns
    -------
    dev_df       : DataFrame, columns = car identifiers WITH usable data,
                   values = (actual - target) temperature per timestep.
    excluded_cars : List of car identifiers found in the header but with
                    no usable data at all (e.g. fully-NaN cars).
    """
    car_cols = parse_car_columns(df)
    devs, excluded = {}, []
    for car, cols in sorted(car_cols.items()):
        series = get_deviation_series(df, cols)
        if series is None:
            excluded.append(car)
        else:
            devs[car] = series
    dev_df = pd.DataFrame(devs)
    return dev_df, excluded


def get_mode_transitions(df: pd.DataFrame, car_cols: dict[str, str]) -> int | None:
    """Count of ACV Running Mode changes for one car (secondary/tiebreak signal)."""
    col = car_cols.get("ACV Running Mode")
    if col is None:
        return None
    rm = df[col].fillna("NA")
    return int((rm != rm.shift(1)).sum())
