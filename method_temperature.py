"""
ACV Method: Temperature Deviation Anomaly Scoring

For cases with indoor temperature sensors (cases 01, 02, 03, 05, 06).
A refrigerant leak prevents the ACV from cooling → the faulty car runs
hotter than the fleet median.

Returns a ranked list of cars (most → least suspicious).
"""

import numpy as np
import pandas as pd


def rank_cars(df: pd.DataFrame, n_cars: int = 8) -> list[int]:
    """
    Rank all 8 cars by indoor temperature deviation from fleet median.

    Parameters
    ----------
    df      : Raw case DataFrame (already loaded from .xlsx)
    n_cars  : Number of cars in the train (default 8)

    Returns
    -------
    ranked : List of car numbers [1..8], most suspicious first.
    """
    # Collect indoor temperature columns
    indoor_cols = [
        c for c in df.columns
        if "Indoor" in c and "Temperature" in c
    ]
    if not indoor_cols:
        # Fallback: passenger cabin / observation area temperature
        indoor_cols = [
            c for c in df.columns
            if "Passenger Cabin Temperature" in c
            or "Observation Area Temperature" in c
        ]

    if not indoor_cols:
        return list(range(1, n_cars + 1))  # no signal — neutral order

    # Mean indoor temp per car
    car_temps: dict[int, list[float]] = {}
    for col in indoor_cols:
        car_num = int(col.split("Car ")[1].split(" ")[0])
        vals = df[col].dropna()
        if len(vals) > 0:
            car_temps.setdefault(car_num, []).append(float(vals.mean()))

    car_means = {c: float(np.mean(v)) for c, v in car_temps.items()}

    # Fleet median as baseline
    fleet_median = float(np.median(list(car_means.values())))

    # Anomaly score: deviation above fleet median (higher = more suspicious)
    anomaly = {c: car_means[c] - fleet_median for c in car_means}

    # Fill cars with no data as neutral (0)
    for c in range(1, n_cars + 1):
        if c not in anomaly:
            anomaly[c] = 0.0

    ranked = sorted(range(1, n_cars + 1), key=lambda c: anomaly[c], reverse=True)
    return ranked
