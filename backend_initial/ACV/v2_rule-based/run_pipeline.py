"""
ACV Subsystem (rule-based) — Single-Command Pipeline
=========================================================
Run this one file to go from raw case files to a submission-ready
predictions file, with diagnostics printed along the way:

    python run_pipeline.py --data-dir /path/to/ACV

That path must contain Train/, Train_Labels.csv, and (by default)
Test/acv_test_case.xlsx. Everything else is optional:

    python run_pipeline.py --data-dir /path/to/ACV \\
        --input /path/to/ACV/Test/acv_test_case.xlsx \\
        --output acv_predictions.csv

What it does, in order
-----------------------
1. Runs the FULL pipeline (schema.py + ranking.py -- no shortcuts) on
   every labelled Train case and scores it against Train_Labels.csv with
   the real competition metric (linear rank-decay).
2. Runs the same pipeline on --input (acv_test_case.xlsx by default).
3. Reports the top-pick margin for that file against the spread of
   margins actually observed in training, flagging an unusually close call.
4. Writes acv_predictions.csv in the exact submission schema: file_id,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACV subsystem — rule-based CUSUM ranking, one-command run."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing Train/, Train_Labels.csv, and Test/.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Case file to predict on. Defaults to <data-dir>/Test/acv_test_case.xlsx.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("acv_predictions.csv"),
        help="Where to write the submission CSV (default: ./acv_predictions.csv).",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    input_path = (args.input or data_dir / "Test" / "acv_test_case.xlsx").resolve()

    if not (data_dir / "Train_Labels.csv").exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    # -----------------------------------------------------------------
    # 1. Self-check: full pipeline vs. Train_Labels.csv, real metric
    # -----------------------------------------------------------------
    print_header("STEP 1/3 — Self-check: full pipeline vs. Train_Labels.csv")
    check_df = diagnostics.self_check_on_train(data_dir)
    print(check_df.to_string(index=False))
    print(f"\n  Mean rank-decay score: {check_df['score'].mean():.4f}")
    if check_df["n_excluded"].sum() > 0:
        print(f"  [INFO] {check_df['n_excluded'].sum()} car(s) across training cases had no "
              f"usable telemetry and were ranked last by convention.")

    train_margins = check_df["margin"].dropna()

    # -----------------------------------------------------------------
    # 2. Run the pipeline on the target input file
    # -----------------------------------------------------------------
    print_header(f"STEP 2/3 — Running pipeline on {input_path.name}")
    df_input = pd.read_excel(input_path)
    result = ranking.rank_cars(df_input)
    print(f"  Ranked cars (most to least suspicious): {' | '.join(result['ranked'])}")
    if result["excluded"]:
        print(f"  [INFO] Car(s) with no usable telemetry, ranked last: {result['excluded']}")

    # -----------------------------------------------------------------
    # 3. Margin diagnostics
    # -----------------------------------------------------------------
    print_header("STEP 3/3 — Confidence check (margin vs. training)")
    mr = diagnostics.margin_report(result, train_margins)
    if mr["flagged"]:
        print(f"  [WARN] Top pick's margin ({mr['margin']:.3f}) is thinner than usual — {mr['reason']}")
        print("  This does not mean the prediction is wrong, only that it is a closer call")
        print("  than most of the training cases. Worth a manual look before trusting it.")
    else:
        print(f"  Margin looks typical: {mr['reason']} (margin={mr['margin']:.3f}).")

    # -----------------------------------------------------------------
    # Write submission file
    # -----------------------------------------------------------------
    out_df = pd.DataFrame([{
        "file_id": input_path.name,
        "ranked_cars": "|".join(result["ranked"]),
    }])
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote prediction for {input_path.name} to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/acv_predictions.csv)")


if __name__ == "__main__":
    main()
