"""
ACV Subsystem (rule-based) — Single-Command Pipeline
=========================================================
Run this one file to go from a raw case file to a submission-ready
predictions file:

    python run_pipeline.py --input /path/to/acv_test_case.xlsx

By default this LOADS the training margins already committed in
artifacts/train_margins.json (used only to flag an unusually thin
top-vs-runner-up margin) and ranks immediately -- no --data-dir, no
training data. The ranking rule itself (schema.py + ranking.py) has no
fitted parameters at all -- every constant it uses (K_SLACK,
HALF_LIFE_ROWS, TIE_EPSILON, ...) is fixed in ranking.py -- so training
data was never needed to rank, only to calibrate what counts as a
"thin" margin worth flagging.

To refresh that comparison set (e.g. after changing ranking.py, or to
verify the shipped margins still reproduce), pass --retrain together
with --data-dir:

    python run_pipeline.py --retrain --data-dir /path/to/ACV \\
        --input /path/to/ACV/Test/acv_test_case.xlsx --output acv_predictions.csv

That path must contain Train/ and Train_Labels.csv. Retraining does, in
order:
1. Runs the FULL pipeline (schema.py + ranking.py -- no shortcuts) on
   every labelled Train case and scores it against Train_Labels.csv with
   the real competition metric (linear rank-decay).
2. Saves the resulting per-case margins to artifacts/ -- OVERWRITING it
   with the result -- so future default runs pick up the refreshed
   comparison set.

Either way, the final step is the same: ranks the target file's cars,
reports the top pick's margin against the training comparison set, and
writes acv_predictions.csv in the exact submission schema: file_id,
ranked_cars (pipe-separated, car identifiers exactly as they appear in
the file's own headers).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import ranking
import schema

DEFAULT_ARTIFACTS_PATH = Path(__file__).parent / "artifacts" / "train_margins.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACV subsystem — rule-based CUSUM ranking, one-command run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Case file to predict on. Required unless --data-dir sets a default via Test/.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("acv_predictions.csv"),
        help="Where to write the submission CSV (default: ./acv_predictions.csv).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing Train/ and Train_Labels.csv. Only needed with --retrain "
             "(or if --input is omitted, to default it to <data-dir>/Test/acv_test_case.xlsx).",
    )
    parser.add_argument(
        "--artifacts-path", type=Path, default=None,
        help=f"Where the training margins are loaded from / saved to (default: {DEFAULT_ARTIFACTS_PATH}).",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Recompute training margins from --data-dir instead of loading the shipped set, and "
             "overwrite --artifacts-path with the result. Requires --data-dir.",
    )
    parser.add_argument(
        "--diagnostics-output", type=Path, default=None,
        help="Optional: write a JSON object alongside --output with the per-car breakdown "
             "rank_cars() already computes but that doesn't belong in the submission CSV -- "
             "primary/secondary score and rank per car, the top-vs-runner-up margin and its "
             "flagged-thin verdict, and each data-bearing car's full CUSUM trajectory (for a "
             "'when did this car's score start climbing' chart). Purely additive: --output's "
             "contents are identical whether or not this is passed.",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def refresh_train_margins(data_dir: Path, artifacts_path: Path) -> pd.Series:
    if not (data_dir / "Train_Labels.csv").exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")

    print_header("STEP 1/2 — Self-check: full pipeline vs. Train_Labels.csv")
    check_df = diagnostics.self_check_on_train(data_dir)
    print(check_df.to_string(index=False))
    print(f"\n  Mean rank-decay score: {check_df['score'].mean():.4f}")
    if check_df["n_excluded"].sum() > 0:
        print(f"  [INFO] {check_df['n_excluded'].sum()} car(s) across training cases had no "
              f"usable telemetry and were ranked last by convention.")

    train_margins = check_df["margin"].dropna()

    print_header("STEP 2/2 — Saving training margins")
    diagnostics.save_train_margins(artifacts_path, train_margins)
    print(f"  Saved {len(train_margins)} margin(s) to {artifacts_path}")
    return train_margins


def main() -> None:
    args = parse_args()
    artifacts_path = (args.artifacts_path or DEFAULT_ARTIFACTS_PATH).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    if args.retrain:
        if data_dir is None:
            sys.exit("[ERROR] --retrain requires --data-dir.")
        train_margins = refresh_train_margins(data_dir, artifacts_path)
    elif artifacts_path.exists():
        print_header(f"Loading training margins from {artifacts_path}")
        print("  (pass --retrain --data-dir <dir> to refresh instead)")
        train_margins = diagnostics.load_train_margins(artifacts_path)
    elif data_dir is not None:
        print_header(f"No saved margins at {artifacts_path} — computing from --data-dir")
        train_margins = refresh_train_margins(data_dir, artifacts_path)
    else:
        sys.exit(
            f"[ERROR] No saved training margins found at {artifacts_path}, and no --data-dir "
            f"given to compute them from. Either point --artifacts-path at an existing "
            f"train_margins.json, or pass --data-dir (add --retrain to force refreshing even if "
            f"margins already exist)."
        )

    input_path = args.input
    if input_path is None:
        if data_dir is None:
            sys.exit("[ERROR] --input is required when --data-dir is not given.")
        input_path = data_dir / "Test" / "acv_test_case.xlsx"
    input_path = input_path.resolve()
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    print_header(f"Ranking cars in {input_path.name}")
    df_input = pd.read_excel(input_path)
    result = ranking.rank_cars(df_input)
    print(f"  Ranked cars (most to least suspicious): {' | '.join(result['ranked'])}")
    if result["excluded"]:
        print(f"  [INFO] Car(s) with no usable telemetry, ranked last: {result['excluded']}")

    print_header("Confidence check (margin vs. training)")
    mr = diagnostics.margin_report(result, train_margins)
    if mr["flagged"]:
        print(f"  [WARN] Top pick's margin ({mr['margin']:.3f}) is thinner than usual — {mr['reason']}")
        print("  This does not mean the prediction is wrong, only that it is a closer call")
        print("  than most of the training cases. Worth a manual look before trusting it.")
    else:
        print(f"  Margin looks typical: {mr['reason']} (margin={mr['margin']:.3f}).")

    out_df = pd.DataFrame([{
        "file_id": input_path.name,
        "ranked_cars": "|".join(result["ranked"]),
    }])
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote prediction for {input_path.name} to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/acv_predictions.csv)")

    if args.diagnostics_output:
        import json
        dev_df, _ = schema.build_deviation_frame(df_input)
        trajectories = ranking.primary_trajectories(dev_df) if not dev_df.empty else {}
        cars = [
            {
                "id": car,
                "rank": rank,
                "primary_score": float(result["primary"][car]) if car in result["primary"] else None,
                "secondary_score": float(result["secondary"][car]) if car in result["secondary"] and pd.notna(result["secondary"][car]) else None,
            }
            for rank, car in enumerate(result["ranked"], start=1)
        ]
        diag = {
            "cars": cars,
            "margin": None if pd.isna(mr["margin"]) else float(mr["margin"]),
            "margin_flagged": bool(mr["flagged"]),
            "margin_reason": mr["reason"],
            "trajectories": trajectories,
            "indoor_temperature": schema.build_indoor_temperature(df_input),
        }
        args.diagnostics_output.write_text(json.dumps(diag, indent=2))
        print(f"  Wrote per-car diagnostics to {args.diagnostics_output}")


if __name__ == "__main__":
    main()
