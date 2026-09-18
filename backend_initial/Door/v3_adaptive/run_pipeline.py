"""
Door Subsystem (v3, adaptive) — Single-Command Pipeline
=========================================================
Run this one file to go from a raw sensor stream to a submission-ready
predictions file:

    python run_pipeline.py --input /path/to/Test.csv

By default this LOADS the calibration already committed in
artifacts/calibration.json and artifacts/train_ranges.json and predicts
immediately -- no --data-dir, no training data. Whenever a stream's local
baseline matches what calibration.json recorded for Train (the only case
this dataset ever demonstrates), the adaptive threshold reduces EXACTLY to
v2's fixed 2060 mA / 700 mA constants, so this version reproduces v2's
predictions on Door's actual Test.csv while being architecturally able to
track a genuinely different door's baseline if one ever shows up.

To recalibrate (e.g. after changing calibration.py, or to verify the
shipped calibration still reproduces), pass --retrain together with
--data-dir:

    python run_pipeline.py --retrain --data-dir /path/to/Door \\
        --input /path/to/Door/Test.csv --output door_predictions.csv

That path must contain Train.csv and Train_Segments_Answer.csv.
Retraining does, in order:
1. Runs the FULL v3 pipeline (segmentation.py + rules.py, adaptive
   thresholds seeded from Train's own history) on Train.csv and scores it
   against Train_Segments_Answer.csv with the real competition metric
   (IoU-weighted F1).
2. Recalibrates train_baseline / seed_window / bootstrap confidence
   half-width per operation, and rebuilds the OOD feature ranges, then
   OVERWRITES artifacts/ with the result.

Either way, the final step is the same: adaptively classifies --input in
stream order, flags any segment that's either out-of-range (an
extrapolation, like v2) or within the bootstrap confidence half-width of
the threshold it was judged against (a genuinely close call, new in v3),
and writes door_predictions.csv in the exact submission schema: start_time,
end_time, prediction (no file_id — Door has none).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import calibration
import diagnostics
import segmentation as seg_mod

SUBMISSION_COLUMNS = ["start_time", "end_time", "prediction"]
DEFAULT_CALIB_PATH = Path(__file__).parent / "artifacts" / "calibration.json"
DEFAULT_RANGES_PATH = Path(__file__).parent / "artifacts" / "train_ranges.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door subsystem (v3) — adaptive-threshold segmentation + classification.",
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
        "--calibration-path", type=Path, default=None,
        help=f"Where calibration is loaded from / saved to (default: {DEFAULT_CALIB_PATH}).",
    )
    parser.add_argument(
        "--ranges-path", type=Path, default=None,
        help=f"Where OOD feature ranges are loaded from / saved to (default: {DEFAULT_RANGES_PATH}).",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Recalibrate from --data-dir instead of loading the shipped calibration, and "
             "overwrite the artifact paths with the result. Requires --data-dir.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the out-of-range / low-confidence segment tables (summary counts still print).",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def recalibrate(data_dir: Path, calib_path: Path, ranges_path: Path) -> tuple[dict, "pd.DataFrame"]:
    for required in ("Train.csv", "Train_Segments_Answer.csv"):
        if not (data_dir / required).exists():
            sys.exit(f"[ERROR] {required} not found in {data_dir}")

    print_header("STEP 1/3 — Calibrating adaptive thresholds from Train")
    calib = calibration.calibrate(data_dir)
    for op, c in calib.items():
        print(f"  {op}: train_baseline={c['train_baseline']:.2f}  window_n={c['window_n']}  "
              f"ci=[{c['ci_lo']:.1f}, {c['ci_hi']:.1f}] (half-width {c['ci_half_width']:.2f})")
    calibration.save_calibration(calib_path, calib)
    print(f"  Saved calibration to {calib_path}")

    print_header("STEP 2/3 — Self-check: full v3 pipeline vs. Train_Segments_Answer.csv")
    metrics = diagnostics.self_check_on_train(data_dir, calib)
    print(f"  True segments (answer file)         : {metrics['n_true']}")
    print(f"  Predicted segments (our pipeline)    : {metrics['n_pred']}")
    print(f"  Matched pairs                       : {metrics['n_matches']}")
    print(f"  >>> IoU-weighted F1 (real metric)    : {metrics['score']:.4f}")
    print(f"  Classification accuracy | matched   : {metrics['classification_accuracy_given_matched_segment']:.4f}")
    print(f"  Low-confidence predictions on Train  : {metrics['n_low_confidence']}")
    if metrics["n_true"] != metrics["n_pred"] or metrics["n_matches"] != metrics["n_true"]:
        print("  [WARN] Segmentation is not a perfect 1:1 match on Train — inspect before trusting Test.")

    print_header("STEP 3/3 — Rebuilding OOD feature ranges")
    ranges = diagnostics.build_train_ranges(data_dir)
    diagnostics.save_train_ranges(ranges_path, ranges)
    print(f"  Saved ranges to {ranges_path}")

    return calib, ranges


def main() -> None:
    args = parse_args()
    calib_path = (args.calibration_path or DEFAULT_CALIB_PATH).resolve()
    ranges_path = (args.ranges_path or DEFAULT_RANGES_PATH).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    have_artifacts = calib_path.exists() and ranges_path.exists()

    if args.retrain:
        if data_dir is None:
            sys.exit("[ERROR] --retrain requires --data-dir.")
        calib, ranges = recalibrate(data_dir, calib_path, ranges_path)
    elif have_artifacts:
        print_header(f"Loading calibration from {calib_path}")
        print("  (pass --retrain --data-dir <dir> to recalibrate instead)")
        calib = calibration.load_calibration(calib_path)
        ranges = diagnostics.load_train_ranges(ranges_path)
    elif data_dir is not None:
        print_header(f"No saved calibration at {calib_path} — calibrating from --data-dir")
        calib, ranges = recalibrate(data_dir, calib_path, ranges_path)
    else:
        sys.exit(
            f"[ERROR] No saved calibration found at {calib_path} / {ranges_path}, and no "
            f"--data-dir given to compute them from. Either point --calibration-path / "
            f"--ranges-path at existing artifacts, or pass --data-dir (add --retrain to force "
            f"recalibrating even if artifacts already exist)."
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
    seed_windows = {op: calib[op]["seed_window"] for op in ("Close", "Open")}
    pred = diagnostics.run_pipeline(df_input, calib, seed_windows=seed_windows)
    print(f"  Rows in input file                  : {len(df_input):,}")
    print(f"  Detected cycles                     : {len(pred)}")
    print(f"  Discarded implausible candidates     : {pred.attrs.get('n_implausible', 0)}")
    print("\n  Predicted label balance:")
    print(pred["prediction"].value_counts().to_string())
    print("\n  Predicted operation balance:")
    print(pred["operation"].value_counts().to_string())
    print("\n  Thresholds used this run (frozen from calibration, not updated mid-run):")
    for op, threshold in pred.attrs.get("thresholds_used", {}).items():
        print(f"    {op}: {threshold:.2f}")
    print("\n  This run's own Normal-predicted cycles (observational only -- not fed back into")
    print("  the thresholds above; would only apply on a future --retrain against this door):")
    for op, vals in pred.attrs.get("observed_normal", {}).items():
        if vals:
            print(f"    {op}: n={len(vals)}  median={pd.Series(vals).median():.2f}  "
                  f"(calibrated baseline: {calib[op]['train_baseline']:.2f})")

    print_header("Out-of-range check (extrapolation risk)")
    flags = diagnostics.flag_out_of_range(pred, ranges)
    if flags.empty:
        print("  No predicted segment falls outside the loaded feature ranges.")
    else:
        n_flagged = flags["segment_row"].nunique()
        print(f"  [WARN] {n_flagged}/{len(pred)} predicted segments have at least one feature "
              f"outside range for their operation.")
        if not args.quiet:
            print(flags.to_string(index=False))

    print_header("Low-confidence check (near the bootstrap threshold uncertainty)")
    low_conf = pred[pred["low_confidence"]]
    if low_conf.empty:
        print("  No prediction falls within its threshold's bootstrap confidence half-width.")
    else:
        print(f"  [WARN] {len(low_conf)}/{len(pred)} predicted segments are close calls — a "
              f"different resample of the same training data could plausibly have flipped them:")
        if not args.quiet:
            cols = ["start_time", "end_time", "operation", "prediction", "threshold_used"]
            print(low_conf[cols].to_string(index=False))

    out_df = pred[SUBMISSION_COLUMNS]
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(out_df)} predicted segments to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/door_predictions.csv)")


if __name__ == "__main__":
    main()
