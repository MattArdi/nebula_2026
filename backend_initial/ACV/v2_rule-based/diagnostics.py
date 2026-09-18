"""
ACV Subsystem (rule-based) — Diagnostics
============================================
Two checks, run before a submission is trusted:

1. self_check_on_train() -- runs the real pipeline (schema.py + ranking.py,
   not a shortcut) on every labelled training case and scores it with the
   ACTUAL competition metric (linear rank-decay), so there is a local
   number that means what the leaderboard means.

2. margin_report() -- for the file actually being predicted (which has no
   label to score against), reports how large the gap is between the top
   pick and the runner-up. A thin margin does not mean the prediction is
   wrong -- there is no way to know that without the true label -- but it
   does mean the top pick is a closer call than the training cases mostly
   were, and that is worth surfacing rather than hiding behind a bare
   ranked list.
"""

import json
from pathlib import Path

import pandas as pd

import ranking

N_CARS = 8
# Thin-margin threshold: below this, flag the file for review. Calibrated
# from training margins -- see algorithm.md Section 3 for the full table;
# acv_case_02 and acv_case_03's real margins (the two closest calls in
# training) were noticeably tighter than the others.
THIN_MARGIN_THRESHOLD_PERCENTILE = 0.25


def rank_decay_score(rank: int, n_cars: int = N_CARS) -> float:
    return (n_cars - (rank - 1)) / n_cars


def self_check_on_train(data_dir: Path) -> pd.DataFrame:
    """
    Run the full ranking pipeline on every Train case and score it against
    Train_Labels.csv with the real rank-decay metric.
    """
    labels = pd.read_csv(data_dir / "Train_Labels.csv", dtype=str)
    rows = []
    for _, row in labels.iterrows():
        fname, faulty = row["filename"], row["faulty_car"]
        path = data_dir / "Train" / fname
        df = pd.read_excel(path)
        result = ranking.rank_cars(df)
        ranked = result["ranked"]

        if faulty not in ranked:
            rows.append({"file": fname, "faulty_car": faulty, "rank": None,
                         "score": 0.0, "margin": result["margin"],
                         "n_excluded": len(result["excluded"]), "note": "faulty car has no data"})
            continue

        rank = ranked.index(faulty) + 1
        rows.append({
            "file": fname, "faulty_car": faulty, "rank": rank,
            "score": rank_decay_score(rank, len(ranked)),
            "margin": result["margin"], "n_excluded": len(result["excluded"]), "note": "",
        })
    return pd.DataFrame(rows)


def save_train_margins(path: Path, train_margins: pd.Series) -> None:
    """Serialize the training margins margin_report() compares against --
    everything a fresh run_pipeline.py needs to flag thin margins without
    ever seeing training data again."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train_margins": train_margins.tolist()}, indent=2))


def load_train_margins(path: Path) -> pd.Series:
    return pd.Series(json.loads(Path(path).read_text())["train_margins"])


def margin_report(result: dict, train_margins: pd.Series) -> dict:
    """
    Compare a prediction's top-vs-runner-up margin against the training
    margins actually observed, to flag unusually close calls.
    """
    margin = result["margin"]
    if pd.isna(margin) or train_margins.empty:
        return {"margin": margin, "flagged": False, "reason": "no comparison available"}

    threshold = train_margins.quantile(THIN_MARGIN_THRESHOLD_PERCENTILE)
    flagged = margin < threshold
    return {
        "margin": margin,
        "threshold": threshold,
        "flagged": flagged,
        "reason": (
            f"margin {margin:.3f} is below the 25th percentile of training margins ({threshold:.3f})"
            if flagged else "margin is within the typical training range"
        ),
    }
