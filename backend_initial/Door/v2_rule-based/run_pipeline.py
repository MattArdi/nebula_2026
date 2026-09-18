"""
Door Subsystem (rule-based) — Single-Command Pipeline
=========================================================
Run this one file to go from a raw sensor stream to a submission-ready
predictions file:

    python run_pipeline.py --input /path/to/Test.csv

By default this LOADS the per-operation feature ranges already committed
in artifacts/train_ranges.json (used only to flag a predicted segment
whose decision-relevant features fall outside what training ever
demonstrated) and predicts immediately -- no --data-dir, no training
data. The segmentation and classification rule itself (segmentation.py +
rules.py) has no fitted parameters at all -- the two thresholds
(CLOSE_CUR_MAX_THRESHOLD, OPEN_CUR_MEAN_THRESHOLD) are fixed constants
in rules.py -- so training data was never needed to classify, only to
calibrate what counts as an out-of-range (extrapolated) prediction worth
flagging.

To refresh that comparison set (e.g. after changing rules.py, or to
verify the shipped ranges still reproduce), pass --retrain together with
--data-dir:

    python run_pipeline.py --retrain --data-dir /path/to/Door \\
        --input /path/to/Door/Test.csv --output door_predictions.csv

That path must contain Train.csv and Train_Segments_Answer.csv.
Retraining does, in order:
1. Runs the FULL pipeline (segmentation.py + rules.py -- no shortcuts, no
   reading answer-file boundaries) on Train.csv and scores it against
   Train_Segments_Answer.csv using the real competition metric
   (IoU-weighted F1) -- the one trustworthy estimate of how the pipeline
   performs end to end.
2. Rebuilds per-operation feature ranges from the 110 labelled Train
   cycles and OVERWRITES artifacts/ with the result -- --retrain is
   destructive to the shipped ranges by design, so future default runs
   pick up the refreshed comparison set.

Either way, the final step is the same: segments and classifies --input,
flags any predicted segment whose decision-relevant features fall
outside the loaded ranges, and writes door_predictions.csv in the exact
submission schema: start_time, end_time, prediction (no file_id — Door
has none).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import segmentation as seg_mod

SUBMISSION_COLUMNS = ["start_time", "end_time", "prediction"]
DEFAULT_ARTIFACTS_PATH = Path(__file__).parent / "artifacts" / "train_ranges.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door subsystem — rule-based segmentation + classification, one-command run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Sensor CSV to predict on. Required unless --data-dir sets a default via Test.csv.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("door_predictions.csv"),
        help="Where to write the submission CSV (default: ./door_predictions.csv).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing Train.csv and Train_Segments_Answer.csv. Only needed with "
             "--retrain (or if --input is omitted, to default it to <data-dir>/Test.csv).",
    )
    parser.add_argument(
        "--artifacts-path", type=Path, default=None,
        help=f"Where feature ranges are loaded from / saved to (default: {DEFAULT_ARTIFACTS_PATH}).",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Rebuild feature ranges from --data-dir instead of loading the shipped set, and "
             "overwrite --artifacts-path with the result. Requires --data-dir.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the out-of-range segment table (summary counts still print).",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def refresh_train_ranges(data_dir: Path, artifacts_path: Path) -> pd.DataFrame:
    for required in ("Train.csv", "Train_Segments_Answer.csv"):
        if not (data_dir / required).exists():
            sys.exit(f"[ERROR] {required} not found in {data_dir}")

    print_header("STEP 1/2 — Self-check: full pipeline vs. Train_Segments_Answer.csv")
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

    print_header("STEP 2/2 — Rebuilding per-operation feature ranges")
    ranges = diagnostics.build_train_ranges(data_dir)
    print(ranges.to_string())
    diagnostics.save_train_ranges(artifacts_path, ranges)
    print(f"  Saved ranges to {artifacts_path}")
    return ranges


def main() -> None:
    args = parse_args()
    artifacts_path = (args.artifacts_path or DEFAULT_ARTIFACTS_PATH).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    if args.retrain:
        if data_dir is None:
            sys.exit("[ERROR] --retrain requires --data-dir.")
        ranges = refresh_train_ranges(data_dir, artifacts_path)
    elif artifacts_path.exists():
        print_header(f"Loading feature ranges from {artifacts_path}")
        print("  (pass --retrain --data-dir <dir> to refresh instead)")
        ranges = diagnostics.load_train_ranges(artifacts_path)
    elif data_dir is not None:
        print_header(f"No saved ranges at {artifacts_path} — computing from --data-dir")
        ranges = refresh_train_ranges(data_dir, artifacts_path)
    else:
        sys.exit(
            f"[ERROR] No saved feature ranges found at {artifacts_path}, and no --data-dir given "
            f"to compute them from. Either point --artifacts-path at an existing "
            f"train_ranges.json, or pass --data-dir (add --retrain to force refreshing even if "
            f"ranges already exist)."
        )

    input_path = args.input
    if input_path is None:
        if data_dir is None:
            sys.exit("[ERROR] --input is required when --data-dir is not given.")
        input_path = data_dir / "Test.csv"
    input_path = input_path.resolve()
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    print_header(f"Running pipeline on {input_path.name}")
    df_input = seg_mod.load_sensor_csv(input_path)
    pred = diagnostics.run_pipeline(df_input)
    print(f"  Rows in input file                  : {len(df_input):,}")
    print(f"  Detected cycles                     : {len(pred)}")
    print(f"  Discarded implausible candidates     : {pred.attrs.get('n_implausible', 0)}")
    print("\n  Predicted label balance:")
    print(pred["prediction"].value_counts().to_string())
    print("\n  Predicted operation balance:")
    print(pred["operation"].value_counts().to_string())

    print_header("Out-of-range check (extrapolation risk)")
    flags = diagnostics.flag_out_of_range(pred, ranges)
    if flags.empty:
        print("  No predicted segment falls outside the loaded feature ranges.")
    else:
        n_segments_flagged = flags["segment_row"].nunique()
        print(
            f"  [WARN] {n_segments_flagged}/{len(pred)} predicted segments have at least one "
            f"feature outside range for their operation."
        )
        print("  These predictions are extrapolations — review before trusting them:")
        if not args.quiet:
            print(flags.to_string(index=False))

    out_df = pred[SUBMISSION_COLUMNS]
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(out_df)} predicted segments to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/door_predictions.csv)")


if __name__ == "__main__":
    main()
