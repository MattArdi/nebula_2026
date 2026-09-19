"""
ACV Method: Solenoid Valve Open Rate Anomaly Scoring

For cases with rich refrigeration sensor data (e.g. case 04) that lack
indoor temperature columns. A refrigerant leak causes the solenoid valve
to open more frequently as the system tries to compensate.

Returns a ranked list of cars (most → least suspicious).
"""

import numpy as np
import pandas as pd


def rank_cars(df: pd.DataFrame, n_cars: int = 8) -> list[int]:
    """
    Rank all 8 cars by refrigeration solenoid valve open rate.

    Parameters
    ----------
    df      : Raw case DataFrame (already loaded from .xlsx)
    n_cars  : Number of cars in the train (default 8)

    Returns
    -------
    ranked : List of car numbers [1..8], most suspicious first.
    """
    car_scores: dict[int, float] = {}

    for car_num in range(1, n_cars + 1):
        vals = []
        for sys_num in [1, 2]:
            col = (
                f"Car {car_num:02d} - "
                f"Refrigeration System {sys_num} Energized Solenoid Valve Open"
            )
            if col in df.columns:
                v = df[col].mean()
                if not np.isnan(v):
                    vals.append(v)
        car_scores[car_num] = float(np.mean(vals)) if vals else float("nan")

    # Cars with data ranked by score (highest = most suspicious)
    has_data = {c: s for c, s in car_scores.items() if not np.isnan(s)}
    no_data  = [c for c in range(1, n_cars + 1) if np.isnan(car_scores.get(c, float("nan")))]

    ranked = sorted(has_data.keys(), key=lambda c: has_data[c], reverse=True) + no_data
    return ranked
