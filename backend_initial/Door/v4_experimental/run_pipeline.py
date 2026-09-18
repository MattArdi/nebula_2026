"""
Door Subsystem (v4, EXPERIMENTAL) — Single-Command Pipeline
=================================================================
*** THIS VERSION IS A VALIDATED NEGATIVE RESULT. DO NOT USE IT FOR ***
*** SUBMISSIONS OR POINT predict.py AT IT. See algorithm.md.        ***

Built to test one specific question: can a more sophisticated Open-cycle
rule correctly flip the one Test.csv segment v3 is suspected to get wrong,
without breaking anything else? Tested against the real Test.csv: it flips
the suspected segment, but also flips 8 of the other 17 Open cycles from
Normal to Abnormal -- a wholesale collapse in precision, not a fix.

Kept in the repo, isolated from v3, specifically so this result is
reproducible rather than just asserted -- and so it can be deleted without
touching v3 at all (they share no files; v3's own segmentation.py is copied
here unchanged, not imported).

    python run_pipeline.py --retrain --data-dir /path/to/Door \\
        --input /path/to/Door/Test.csv --output door_predictions_v4.csv \\
        --diff-against /path/to/v3/door_predictions.csv
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
DEFAULT_CALIB_PATH = Path(__file__).parent / "artifacts" / "open_class_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door subsystem (v4, EXPERIMENTAL) — hybrid threshold/distance classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("door_predictions_v4.csv"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument(
        "--diff-against", type=Path, default=None,
        help="A v3 door_predictions.csv to diff this run's output against.",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    args = parse_args()
    calib_path = (args.calibration_path or DEFAULT_CALIB_PATH).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    if args.retrain or not calib_path.exists():
        if data_dir is None:
            sys.exit("[ERROR] No saved calibration and no --data-dir to build one from.")
        for required in ("Train.csv", "Train_Segments_Answer.csv"):
            if not (data_dir / required).exists():
                sys.exit(f"[ERROR] {required} not found in {data_dir}")

        print_header("STEP 1/2 — Calibrating Open's per-class distance stats from Train")
        calib = calibration.calibrate(data_dir)
        calibration.save_calibration(calib_path, calib)
        print(f"  Saved to {calib_path}")

        print_header("STEP 2/2 — Self-check: full v4 pipeline vs. Train_Segments_Answer.csv")
        metrics = diagnostics.self_check_on_train(data_dir, calib)
        print(f"  True segments (answer file)         : {metrics['n_true']}")
        print(f"  Predicted segments (our pipeline)    : {metrics['n_pred']}")
        print(f"  Matched pairs                       : {metrics['n_matches']}")
        print(f"  >>> IoU-weighted F1 (real metric)    : {metrics['score']:.4f}")
        if metrics["score"] < 1.0:
            print("  [WARN] This is already below v2/v3's perfect 1.0000 self-check on Train.")
    else:
        print_header(f"Loading calibration from {calib_path}")
        calib = calibration.load_calibration(calib_path)

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
    pred = diagnostics.run_pipeline(df_input, calib)
    print(f"  Detected cycles: {len(pred)}")
    print("\n  Predicted label balance:")
    print(pred["prediction"].value_counts().to_string())

    if args.diff_against:
        print_header(f"Diff against v3's predictions: {args.diff_against}")
        diffs = diagnostics.diff_against_v3(pred, args.diff_against)
        if diffs.empty:
            print("  No differences from v3.")
        else:
            print(f"  [WARN] {len(diffs)} segment(s) differ from v3's (already Test-verified) output:")
            print(diffs.to_string(index=False))

    out_df = pred[SUBMISSION_COLUMNS]
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(out_df)} predicted segments to {args.output}")
    print("  Reminder: this version is a validated negative result. See algorithm.md before using it.")


if __name__ == "__main__":
    main()
