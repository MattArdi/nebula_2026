"""
Door Subsystem (rule-based) — Single-Command Pipeline
=========================================================
Run this one file to go from raw sensor CSVs to a submission-ready
predictions file, with diagnostics printed along the way:

    python run_pipeline.py --data-dir /path/to/Door

That path must contain Train.csv, Train_Segments_Answer.csv, and (by
default) Test.csv. Everything else is optional:

    python run_pipeline.py --data-dir /path/to/Door \\
        --input /path/to/Door/Test.csv \\
        --output door_predictions.csv

What it does, in order
-----------------------
1. Runs the FULL pipeline (segmentation.py + rules.py — no shortcuts, no
   reading answer-file boundaries) on Train.csv and scores it against
   Train_Segments_Answer.csv using the real competition metric
   (IoU-weighted F1, diagnostics.iou_weighted_f1). This is the one
   trustworthy estimate of how the pipeline performs end to end.
2. Builds per-operation feature ranges from the 110 labelled Train
   cycles.
3. Runs the same pipeline on --input (Test.csv by default).
4. Flags any predicted segment whose decision-relevant features fall
   outside what Train ever demonstrated for that operation — these
   predictions are extrapolations, not interpolations, and are called
   out explicitly rather than silently trusted.
5. Writes door_predictions.csv in the exact submission schema:
   start_time, end_time, prediction (no file_id — Door has none).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import segmentation as seg_mod

SUBMISSION_COLUMNS = ["start_time", "end_time", "prediction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door subsystem — rule-based segmentation + classification, one-command run."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing Train.csv, Train_Segments_Answer.csv, and Test.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Sensor CSV to predict on. Defaults to <data-dir>/Test.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("door_predictions.csv"),
        help="Where to write the submission CSV (default: ./door_predictions.csv).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the out-of-range segment table (summary counts still print).",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    input_path = (args.input or data_dir / "Test.csv").resolve()

    for required in ("Train.csv", "Train_Segments_Answer.csv"):
        if not (data_dir / required).exists():
            sys.exit(f"[ERROR] {required} not found in {data_dir}")
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    # -----------------------------------------------------------------
    # 1. Self-check: full pipeline on Train.csv, real competition metric
    # -----------------------------------------------------------------
    print_header("STEP 1/4 — Self-check: full pipeline vs. Train_Segments_Answer.csv")
    metrics = diagnostics.self_check_on_train(data_dir)
    print(f"  True segments (answer file)         : {metrics['n_true']}")
    print(f"  Predicted segments (our pipeline)    : {metrics['n_pred']}")
    print(f"  Matched pairs                       : {metrics['n_matches']}")
    print(f"  Implausible/discarded candidates     : {metrics['n_implausible']}")
    print(f"  Soft recall                         : {metrics['soft_recall']:.4f}")
    print(f"  Soft precision                      : {metrics['soft_precision']:.4f}")
    print(f"  >>> IoU-weighted F1 (real metric)    : {metrics['score']:.4f}")
    print(
        f"  Classification accuracy | matched   : "
        f"{metrics['classification_accuracy_given_matched_segment']:.4f}"
    )
    if metrics["n_true"] != metrics["n_pred"] or metrics["n_matches"] != metrics["n_true"]:
        print("  [WARN] Segmentation is not a perfect 1:1 match on Train — inspect before trusting Test.")

    # -----------------------------------------------------------------
    # 2. Train feature ranges (for OOD flagging)
    # -----------------------------------------------------------------
    print_header("STEP 2/4 — Building Train feature ranges (per operation)")
    ranges = diagnostics.build_train_ranges(data_dir)
    print(ranges.to_string())

    # -----------------------------------------------------------------
    # 3. Run the pipeline on the target input file
    # -----------------------------------------------------------------
    print_header(f"STEP 3/4 — Running pipeline on {input_path.name}")
    df_input = seg_mod.load_sensor_csv(input_path)
    pred = diagnostics.run_pipeline(df_input)
    print(f"  Rows in input file                  : {len(df_input):,}")
    print(f"  Detected cycles                     : {len(pred)}")
    print(f"  Discarded implausible candidates     : {pred.attrs.get('n_implausible', 0)}")
    print("\n  Predicted label balance:")
    print(pred["prediction"].value_counts().to_string())
    print("\n  Predicted operation balance:")
    print(pred["operation"].value_counts().to_string())

    # -----------------------------------------------------------------
    # 4. Out-of-range diagnostics
    # -----------------------------------------------------------------
    print_header("STEP 4/4 — Out-of-range check (extrapolation risk)")
    flags = diagnostics.flag_out_of_range(pred, ranges)
    if flags.empty:
        print("  No predicted segment falls outside Train's demonstrated feature range.")
    else:
        n_segments_flagged = flags["segment_row"].nunique()
        print(
            f"  [WARN] {n_segments_flagged}/{len(pred)} predicted segments have at least one "
            f"feature outside Train's range for their operation."
        )
        print("  These predictions are extrapolations — review before trusting them:")
        if not args.quiet:
            print(flags.to_string(index=False))

    # -----------------------------------------------------------------
    # Write submission file
    # -----------------------------------------------------------------
    out_df = pred[SUBMISSION_COLUMNS]
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(out_df)} predicted segments to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/door_predictions.csv)")


if __name__ == "__main__":
    main()
