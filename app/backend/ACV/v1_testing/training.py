"""
ACV Subsystem — Training & Evaluation Script
=============================================
Ranks 8 cars by likelihood of refrigerant leak, then scores the ranking
against ground-truth labels using the linear rank-decay metric.

Linear rank-decay score per case:
    score = (n_cars - (rank - 1)) / n_cars
    where rank is the position of the true faulty car in the predicted ranking.
    Rank 1 → score 1.0, Rank 8 → score 0.125.

Usage
-----
    python training.py --method temperature   # indoor temp deviation (default)
    python training.py --method solenoid      # solenoid valve open rate
    python training.py --method auto          # auto-select per case (recommended)

    # Point to a different data directory
    python training.py --method auto --data-dir /path/to/ACV

    # Also write per-case predictions to a CSV
    python training.py --method auto --output predictions.csv
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CARS = 8
METHODS = ("temperature", "solenoid", "auto")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def linear_rank_decay(faulty_car: int, ranked: list[int], n_cars: int = N_CARS) -> float:
    """
    Score a single ranking.

    Parameters
    ----------
    faulty_car : Ground-truth faulty car number (1-indexed).
    ranked     : Predicted ranking, most suspicious first.
    n_cars     : Total number of cars.

    Returns
    -------
    score : float in (0, 1].
    """
    rank = ranked.index(faulty_car) + 1  # 1-indexed
    return (n_cars - (rank - 1)) / n_cars


def score_all(results: list[dict]) -> dict:
    """
    Aggregate per-case scores.

    Parameters
    ----------
    results : List of dicts with keys 'filename', 'faulty_car', 'ranked', 'score'.

    Returns
    -------
    metrics : Dict with mean_score, min_score, max_score, per_case scores.
    """
    scores = [r["score"] for r in results]
    return {
        "mean_score": float(np.mean(scores)),
        "min_score":  float(np.min(scores)),
        "max_score":  float(np.max(scores)),
        "n_cases":    len(scores),
        "per_case":   {r["filename"]: r["score"] for r in results},
    }


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------
def _has_indoor_temp(df: pd.DataFrame) -> bool:
    return any(
        ("Indoor" in c and "Temperature" in c) or
        "Passenger Cabin Temperature" in c or
        "Observation Area Temperature" in c
        for c in df.columns
    )


def _has_solenoid(df: pd.DataFrame) -> bool:
    return any("Solenoid Valve Open" in c for c in df.columns)


def select_method(df: pd.DataFrame, method: str) -> str:
    """
    Given a user-specified method and a loaded DataFrame, return the
    method name to actually use (resolves 'auto').
    """
    if method != "auto":
        return method
    if _has_indoor_temp(df):
        return "temperature"
    if _has_solenoid(df):
        return "solenoid"
    # Last resort: temperature (will return neutral ranking)
    return "temperature"


def load_method(method_name: str):
    """Dynamically import and return the ranking method module."""
    module_name = f"method_{method_name}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        sys.exit(f"[ERROR] Unknown method '{method_name}'. Choose from: {METHODS}")
    return mod


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_case(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_labels(data_dir: Path) -> pd.DataFrame:
    label_path = data_dir / "Train_Labels.csv"
    if not label_path.exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")
    return pd.read_csv(label_path)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def run(data_dir: Path, method: str, output: Path | None, verbose: bool) -> dict:
    """
    Load all training cases, rank cars for each, score, and report.

    Parameters
    ----------
    data_dir : Directory containing Train/ and Train_Labels.csv.
    method   : One of 'temperature', 'solenoid', 'auto'.
    output   : Optional path to write prediction CSV.
    verbose  : Print per-case details.

    Returns
    -------
    metrics : Aggregated scoring metrics dict.
    """
    labels    = load_labels(data_dir)
    train_dir = data_dir / "Train"

    results = []

    for _, row in labels.iterrows():
        fname      = row["filename"]
        faulty_car = int(row["faulty_car"])
        fpath      = train_dir / fname

        if not fpath.exists():
            print(f"[WARN] {fname} not found — skipping.")
            continue

        df = load_case(fpath)

        # Resolve method per case (important for 'auto')
        resolved = select_method(df, method)

        # Dynamically load the method module
        # Add the script directory to sys.path so imports work when called
        # from any working directory.
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        mod    = load_method(resolved)
        ranked = mod.rank_cars(df, n_cars=N_CARS)
        score  = linear_rank_decay(faulty_car, ranked)

        ranked_str = "|".join(f"{c:02d}" for c in ranked)

        results.append({
            "filename":   fname,
            "faulty_car": faulty_car,
            "method":     resolved,
            "ranked":     ranked,
            "ranked_str": ranked_str,
            "score":      score,
        })

        if verbose:
            print(
                f"  {fname}  faulty=Car{faulty_car:02d}  "
                f"method={resolved:<12}  rank=#{ranked.index(faulty_car)+1}  "
                f"score={score:.3f}  ranking={ranked_str}"
            )

    metrics = score_all(results)

    # Write prediction CSV if requested
    if output is not None:
        pred_df = pd.DataFrame(
            [{"file_id": r["filename"], "ranked_cars": r["ranked_str"]} for r in results]
        )
        pred_df.to_csv(output, index=False)
        print(f"\n[INFO] Predictions written to {output}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACV subsystem — train and evaluate car-fault ranking methods."
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="auto",
        help=(
            "Ranking method to use:\n"
            "  temperature — indoor temp deviation from fleet median (default for most cases)\n"
            "  solenoid    — solenoid valve open rate (for rich-sensor cases)\n"
            "  auto        — choose best method per case automatically (recommended)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "../../../02_Datasets/ACV",
        help="Path to the ACV dataset directory (must contain Train/ and Train_Labels.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If provided, write ranked predictions to this CSV file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        sys.exit(f"[ERROR] Data directory not found: {data_dir}")

    print(f"ACV Training — method={args.method}  data_dir={data_dir}")
    print("-" * 60)

    metrics = run(
        data_dir=data_dir,
        method=args.method,
        output=args.output,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)
    print(f"  Cases evaluated : {metrics['n_cases']}")
    print(f"  Mean score      : {metrics['mean_score']:.4f}")
    print(f"  Min score       : {metrics['min_score']:.4f}")
    print(f"  Max score       : {metrics['max_score']:.4f}")
    print()
    print("  Per-case scores:")
    for fname, score in metrics["per_case"].items():
        print(f"    {fname:<25} {score:.3f}")


if __name__ == "__main__":
    main()
