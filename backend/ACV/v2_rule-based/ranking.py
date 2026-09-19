"""
ACV Subsystem (rule-based) — CUSUM Ranking
==============================================
Ranks the 8 cars in one case file from most- to least-likely to have the
refrigerant leak.

Primary signal: local cross-sectional CUSUM
--------------------------------------------
For each car, at every timestep:
  1. gap = this car's (indoor - setpoint) deviation minus the CROSS-CAR
     MEDIAN deviation at that same timestep. Comparing against the other
     cars at the same moment (not against this car's own history, and not
     against a fixed setpoint alone) cancels out shared environmental
     noise -- outdoor temperature swings hit all 8 cars at once, so they
     wash out of the comparison automatically.
  2. That gap is standardized by the LOCAL spread across all 8 cars,
     pooled over a trailing window (not a single instant, not a global
     constant) -- this is what lets one file's inherently noisier cars
     (e.g. acv_case_06, where even healthy cars deviate by >1 degC) not
     get treated the same as a much quieter file's cars.
  3. The standardized series is fed into a decayed, slack-adjusted CUSUM
     (see algorithm.md for the full derivation and the numbers behind
     every constant below). The car with the highest CUSUM value is the
     lead suspect.

Secondary signal: ACV Running Mode transition count, as a TIEBREAK ONLY
--------------------------------------------------------------------------
An independent corroborating signal from a different data source
(categorical control-mode switching, not temperature). It is deliberately
NOT blended into every ranking as a co-equal vote -- an earlier version of
this file averaged each signal's own rank (a Borda average) unconditionally,
and that broke acv_case_06 on real data: the primary signal was completely
unambiguous there (1543.8 vs. a runner-up of 8.4, a ~180x margin), but the
mode-transition counts happened to be near-flat noise (290-293 transitions
across all 8 cars, a spread of essentially nothing) with the true car at
the bottom of that noise by chance -- and a blanket rank average let 3
transitions' worth of noise overrule a 180x primary margin. See
algorithm.md Section 3 for the full account.

The fix: cars are ranked by the primary CUSUM score, full stop, UNLESS two
(or more) cars' primary scores are within TIE_EPSILON of each other --
only then is the secondary signal consulted, to decide among genuinely
close candidates. A weak secondary signal can never move a decisive
primary result.
"""

import numpy as np
import pandas as pd

import schema

WINDOW_ROWS = 120        # ~1 hour at 30s sampling: the local normalization window
K_SLACK = 0.5             # CUSUM reference value, in local-sigma units (SPC default)
HALF_LIFE_ROWS = 480      # ~4 hours at 30s sampling: CUSUM forgetting half-life
SIGMA_FLOOR = 0.05        # avoid divide-by-zero in near-constant stretches

# Two cars are a "near tie" if their primary scores differ by less than this
# fraction of the higher score. Set well below the smallest real margin seen
# in training (acv_case_02, 19.0%), so no genuine training result is ever
# close enough to trigger a tiebreak -- it only fires on truly ambiguous calls.
TIE_EPSILON = 0.10


def local_cross_sectional_gap_and_sigma(dev_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    gap_df : each car's deviation minus the cross-car median, per timestep.
    sigma  : one spread estimate per timestep, from a trailing window pooling
             ALL cars' gaps (robust to one car being genuinely anomalous).
    """
    median_dev = dev_df.median(axis=1)
    gap_df = dev_df.sub(median_dev, axis=0)
    pooled_abs = gap_df.abs().mean(axis=1)
    sigma = pooled_abs.rolling(WINDOW_ROWS, min_periods=10).median()
    sigma = sigma.bfill().fillna(SIGMA_FLOOR).clip(lower=SIGMA_FLOOR)
    return gap_df, sigma


def cusum_score(z_values: np.ndarray, k: float = K_SLACK, half_life_rows: int = HALF_LIFE_ROWS) -> float:
    """One-sided decayed CUSUM, read as the CURRENT value (not the all-time peak) --
    see algorithm.md Section 2 for why peak-tracking was rejected."""
    decay = 0.5 ** (1.0 / half_life_rows)
    c = 0.0
    for v in z_values:
        v = 0.0 if np.isnan(v) else v
        c = max(0.0, decay * c + (v - k))
    return c


def primary_scores(dev_df: pd.DataFrame) -> pd.Series:
    """CUSUM score per car (higher = more suspicious). Index = car identifiers."""
    gap_df, sigma = local_cross_sectional_gap_and_sigma(dev_df)
    z_df = gap_df.div(sigma, axis=0)
    return pd.Series({car: cusum_score(z_df[car].values) for car in dev_df.columns})


def secondary_scores(df: pd.DataFrame, cars_with_data: list[str]) -> pd.Series:
    """Mode-transition count per car (higher = more suspicious; weak, corroborating only)."""
    car_cols = schema.parse_car_columns(df)
    out = {}
    for car in cars_with_data:
        n = schema.get_mode_transitions(df, car_cols.get(car, {}))
        out[car] = n if n is not None else np.nan
    return pd.Series(out)


def _cluster_by_near_ties(sorted_cars: list[str], primary: pd.Series, eps: float = TIE_EPSILON) -> list[list[str]]:
    """
    Group a primary-score-sorted car list into clusters, chaining adjacent
    cars whose primary scores differ by less than `eps` of the higher score.
    A clear leader (score not within eps of the runner-up) is its own
    cluster of one, and is never reordered by the tiebreak signal.
    """
    clusters: list[list[str]] = []
    current = [sorted_cars[0]]
    for prev_car, car in zip(sorted_cars, sorted_cars[1:]):
        hi, lo = primary[prev_car], primary[car]
        close = hi > 0 and (hi - lo) / hi < eps
        if close:
            current.append(car)
        else:
            clusters.append(current)
            current = [car]
    clusters.append(current)
    return clusters


def rank_cars(df: pd.DataFrame) -> dict:
    """
    Rank every car in the file from most- to least-likely faulty.

    Returns
    -------
    dict with:
      'ranked'      : list of car identifiers, most suspicious first.
      'primary'     : raw CUSUM score per car (data-bearing cars only).
      'secondary'   : raw mode-transition count per car (data-bearing cars only).
      'excluded'    : car identifiers with no usable data at all.
      'margin'      : primary score gap between rank 1 and rank 2 (data-bearing cars only).
    """
    dev_df, excluded = schema.build_deviation_frame(df)
    cars_with_data = list(dev_df.columns)

    if not cars_with_data:
        # Degenerate case: no car in this file has any usable telemetry at all.
        all_cars = sorted(schema.parse_car_columns(df).keys())
        return {"ranked": all_cars, "primary": pd.Series(dtype=float),
                "secondary": pd.Series(dtype=float), "excluded": all_cars, "margin": float("nan")}

    primary = primary_scores(dev_df)
    secondary = secondary_scores(df, cars_with_data)

    # Base order: primary CUSUM score, descending.
    sorted_cars = primary.sort_values(ascending=False).index.tolist()
    clusters = _cluster_by_near_ties(sorted_cars, primary)

    # Within each near-tie cluster only, break ties with the secondary
    # signal, then deterministically by car identifier.
    ranked: list[str] = []
    for cluster in clusters:
        if len(cluster) == 1:
            ranked.extend(cluster)
            continue
        cluster_df = pd.DataFrame({
            "secondary": secondary.loc[cluster].fillna(-1),
            "car": cluster,
        }).sort_values(by=["secondary", "car"], ascending=[False, True])
        ranked.extend(cluster_df["car"].tolist())

    ranked = ranked + sorted(excluded)  # no-data cars always ranked last

    sorted_primary = primary.sort_values(ascending=False)
    margin = (
        sorted_primary.iloc[0] - sorted_primary.iloc[1]
        if len(sorted_primary) > 1 else float("nan")
    )

    return {"ranked": ranked, "primary": primary, "secondary": secondary,
            "excluded": excluded, "margin": margin}
