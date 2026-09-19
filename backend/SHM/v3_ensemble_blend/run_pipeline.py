"""
SHM Subsystem (v3, ensemble blend) — Single-Command Pipeline
==================================================================
Run this one file to go from raw stress CSVs to a submission-ready
predictions file:

    python run_pipeline.py --input /path/to/Test

By default this LOADS the pre-fitted blend already committed in
artifacts/ (physics_calibration.json, sklearn_components.joblib) and
predicts immediately -- no --data-dir, no training data, no refitting.

--input accepts either a directory (every *.csv inside it is predicted)
or a single CSV file.

v3 blends 4 members -- the physics model (v2, unchanged), linear
regression and bagged Ridge on the physics proxy, and Extra Trees on a
47-feature set -- with fixed weights (see model.py). SHIPPED DESPITE THE
EVIDENCE: every individual ML model tried scored below the pure physics
model under honest leave-one-out validation; this blend's +0.0006 LOO
gain over pure physics was found by a weight search with no independent
confirmation set, the same size as several changes this session
confirmed were noise. See algorithm.md Section 5 for the full account.

To refit instead of loading the shipped artifacts, pass --retrain
together with --data-dir:

    python run_pipeline.py --retrain --data-dir /path/to/SHM \\
        --input /path/to/SHM/Test --output shm_predictions.csv

That path must contain Train/ and Train_Labels.csv. Retraining does, in
order:
1. Extracts the 47-feature vector (features.py) for every labelled
   training file.
2. Runs the leave-one-out self-check on the full 4-member blend
   (diagnostics.py), scored with the real competition metric.
3. Fits the final blend on all 64 training files and OVERWRITES
   artifacts/ with the result.

Either way, the final step is the same: predicts on every file under
--input, flags any prediction whose physics proxy falls outside the range
Train ever demonstrated, and writes shm_predictions.csv in the exact
submission schema: file_id, prediction.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import features as feat_mod
import model as model_mod
import physics

DEFAULT_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHM subsystem (v3) — physics + ML ensemble blend, one-command run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None,
                         help="File or directory to predict on. Required unless --data-dir sets a default via Test/.")
    parser.add_argument("--output", type=Path, default=Path("shm_predictions.csv"),
                         help="Where to write the submission CSV (default: ./shm_predictions.csv).")
    parser.add_argument("--data-dir", type=Path, default=None,
                         help="Directory containing Train/ and Train_Labels.csv. Only needed with --retrain.")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                         help=f"Where fitted artifacts are loaded from / saved to (default: {DEFAULT_ARTIFACTS_DIR}).")
    parser.add_argument("--retrain", action="store_true",
                         help="Refit the full blend from --data-dir instead of loading shipped artifacts.")
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def build_train_tables(data_dir: Path):
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    feature_rows, filenames = [], []
    for _, row in labels.iterrows():
        x = physics.load_stress_series(data_dir / "Train" / row["filename"])
        cycles = physics.rainflow_cycles(x)
        feature_rows.append(feat_mod.extract_features(x, cycles))
        filenames.append(row["filename"])
    damages = labels["damage"].values
    return feature_rows, damages, filenames


def train_blend(data_dir: Path, artifacts_dir: Path) -> model_mod.EnsembleBlend:
    if not (data_dir / "Train_Labels.csv").exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")

    print_header(f"STEP 1/3 — Extracting features for training files")
    feature_rows, damages, filenames = build_train_tables(data_dir)
    print(f"  {len(feature_rows)} files x {len(feat_mod.FEATURE_NAMES)} features")

    print_header("STEP 2/3 — Self-check: leave-one-out CV of the full blend")
    loo = diagnostics.self_check_loo(feature_rows, damages, filenames)
    print(loo.sort_values("rel_err_pct", ascending=False).head(10).to_string(index=False))
    print(f"\n  Mean LOO score (max(0, 1-MAPE)): {loo.attrs['mean_score']:.4f}")

    print_header("STEP 3/3 — Fitting final blend on all training files")
    blend = model_mod.EnsembleBlend().fit(feature_rows, damages)
    blend.save(artifacts_dir)
    print(f"  Saved artifacts to {artifacts_dir}")
    return blend


def resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return [input_path]


def main() -> None:
    args = parse_args()
    artifacts_dir = (args.artifacts_dir or DEFAULT_ARTIFACTS_DIR).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    if args.retrain:
        if data_dir is None:
            sys.exit("[ERROR] --retrain requires --data-dir.")
        blend = train_blend(data_dir, artifacts_dir)
        train_feature_rows, train_damages, _ = build_train_tables(data_dir)
        train_range = diagnostics.build_train_proxy_range(train_feature_rows)
    elif model_mod.EnsembleBlend.artifacts_exist(artifacts_dir):
        print_header(f"Loading pre-fitted blend from {artifacts_dir}")
        blend = model_mod.EnsembleBlend.load(artifacts_dir)
        if data_dir is not None and (data_dir / "Train_Labels.csv").exists():
            train_feature_rows, _, _ = build_train_tables(data_dir)
            train_range = diagnostics.build_train_proxy_range(train_feature_rows)
        else:
            train_range = None
    elif data_dir is not None:
        print_header(f"No saved artifacts at {artifacts_dir} — fitting from --data-dir")
        blend = train_blend(data_dir, artifacts_dir)
        train_feature_rows, train_damages, _ = build_train_tables(data_dir)
        train_range = diagnostics.build_train_proxy_range(train_feature_rows)
    else:
        sys.exit(f"[ERROR] No saved artifacts at {artifacts_dir}, and no --data-dir given to fit from.")

    input_path = args.input
    if input_path is None:
        if data_dir is None:
            sys.exit("[ERROR] --input is required when --data-dir is not given.")
        input_path = data_dir / "Test"
    input_path = input_path.resolve()
    if not input_path.exists():
        sys.exit(f"[ERROR] Input path not found: {input_path}")

    print_header(f"Predicting on {input_path}")
    files = resolve_input_files(input_path)
    if not files:
        sys.exit(f"[ERROR] No .csv files found under {input_path}")

    feature_rows = [feat_mod.extract_from_path(f) for f in files]
    preds = blend.predict(feature_rows)

    results = []
    for f, row, pred in zip(files, feature_rows, preds):
        proxy = row[feat_mod.PHYSICS_PROXY_COL]
        flag = diagnostics.flag_extrapolation(proxy, train_range) if train_range else {"flagged": False, "reason": "n/a (no --data-dir given, extrapolation check skipped)"}
        results.append({"file_id": f.name, "prediction": pred, **flag})
    pred_df = pd.DataFrame(results)
    print(pred_df[["file_id", "prediction"]].to_string(index=False))

    flagged = pred_df[pred_df["flagged"]]
    if flagged.empty:
        print("\n  No prediction's proxy falls outside Train's demonstrated range.")
    else:
        print(f"\n  [WARN] {len(flagged)}/{len(pred_df)} file(s) flagged:")
        for _, row in flagged.iterrows():
            print(f"    {row['file_id']}: {row['reason']}")

    out_df = pred_df[["file_id", "prediction"]]
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(out_df)} predictions to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/shm_predictions.csv)")


if __name__ == "__main__":
    main()
