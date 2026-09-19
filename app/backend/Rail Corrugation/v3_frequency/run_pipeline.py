"""
Rail Corrugation Subsystem (v3, frequency-augmented) — Single-Command Pipeline
====================================================================================
Run this one file to go from raw sensor CSVs to a submission-ready
predictions file:

    python run_pipeline.py --input /path/to/Test

By default this LOADS the pre-trained ensemble already committed in
weights/ (catboost.cbm, xgboost.json, sklearn_components.joblib) and
predicts straight away -- no --data-dir, no training data, no retraining
needed.

--input accepts either a directory (every *.csv inside it is predicted,
one row each) or a single CSV file.

v3 adds one thing to v2: 8 frequency-domain features (FFT energy in the
0-100 Hz band, per bearing position, aggregated the same way every other
feature here is) alongside v2's original 41 time-domain features, for 49
total. The ensemble architecture (model.py) is untouched -- see
algorithm.md Section 5 for the validated numbers behind this change
(macro F1 0.8411 -> 0.8935 in repeated CV, confirmed on two independent
sets of CV seeds) and why this specific band was chosen over the others
tried.

To retrain instead of loading the shipped weights (e.g. after changing
features.py or model.py, or to verify the shipped weights reproduce),
pass --retrain together with --data-dir:

    python run_pipeline.py --retrain --data-dir /path/to/Rail_Corrugation \\
        --input /path/to/Rail_Corrugation/Test --output rail_predictions.csv

That path must contain Train/ and Train_Labels.csv. Retraining does, in
order:
1. Extracts the 49-feature vector (features.py) for every labelled
   training file.
2. Runs the self-check: repeated stratified 5-fold cross-validation of
   the full CatBoost+XGBoost+LogReg soft-voting ensemble (model.py),
   scored with the real competition metric -- macro F1 -- plus a
   per-class F1 breakdown and confusion matrix (diagnostics.py).
3. Fits the final ensemble on all 272 training files and OVERWRITES
   weights/ with the result -- --retrain is destructive to the shipped
   weights by design, so future default runs pick up the retrained model.

Either way, the final step is the same: predicts on every file under
--input and writes rail_predictions.csv in the exact submission schema:
file_id, prediction (string labels, not encoded integers).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import features as feat_mod
import model as model_mod

DEFAULT_WEIGHTS_DIR = Path(__file__).parent / "weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rail Corrugation subsystem (v3) — frequency-augmented soft-voting ensemble, one-command run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="File or directory to predict on. Required unless --data-dir sets a default via Test/.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("rail_predictions.csv"),
        help="Where to write the submission CSV (default: ./rail_predictions.csv).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing Train/ and Train_Labels.csv. Only needed with --retrain "
             "(or if --input is omitted, to default it to <data-dir>/Test).",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=None,
        help=f"Where trained weights are loaded from / saved to (default: {DEFAULT_WEIGHTS_DIR}).",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain from --data-dir instead of loading saved weights, and overwrite --weights-dir "
             "with the result. Requires --data-dir.",
    )
    parser.add_argument(
        "--diagnostics-output", type=Path, default=None,
        help="Optional: write a JSON array alongside --output, one object per file (same order), "
             "with values the pipeline already computes but that don't belong in the submission "
             "CSV -- the ensemble's own class probabilities, speed_kmh, and which (car, bearing "
             "position) is the worst offender on each side. Purely additive: --output's contents "
             "are identical whether or not this is passed.",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return [input_path]


def train_ensemble(data_dir: Path, weights_dir: Path) -> model_mod.RailEnsemble:
    labels_path = data_dir / "Train_Labels.csv"
    if not labels_path.exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")
    labels = pd.read_csv(labels_path)

    print_header(f"STEP 1/3 — Extracting features for {len(labels)} training files")
    train_paths = [data_dir / "Train" / fn for fn in labels["filename"]]
    X_train = feat_mod.build_feature_matrix(train_paths).values
    y_train = labels["label"].values
    print(f"  Feature matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")
    print(f"  Class counts: { {c: int((y_train == c).sum()) for c in sorted(set(y_train))} }")

    print_header("STEP 2/3 — Self-check: 5x5 stratified CV, scored by macro F1")
    diag = diagnostics.self_check_cv(X_train, y_train)
    print(f"  Per-repeat macro F1: {[round(s, 4) for s in diag['repeat_scores']]}")
    print(f"  Mean macro F1 (the real competition metric): {diag['mean_f1']:.4f} +/- {diag['std_f1']:.4f}")
    print("\n  Per-class F1 (pooled over all repeats/folds):")
    for cls, f1 in diag["per_class_f1"].items():
        print(f"    {cls:<10} {f1:.4f}")
    print("\n  Confusion matrix (rows=true, cols=pred), pooled:")
    print(diag["confusion_matrix"].to_string())
    print("\n  Classification report:")
    print(diag["report"])

    print_header("STEP 3/3 — Fitting final ensemble on all training files")
    ensemble = model_mod.RailEnsemble().fit(X_train, y_train)
    print(f"  Fitted CatBoost, XGBoost, and LogReg on all {X_train.shape[0]} training files.")
    ensemble.save(weights_dir)
    print(f"  Saved weights to {weights_dir}")
    return ensemble


def main() -> None:
    args = parse_args()
    weights_dir = (args.weights_dir or DEFAULT_WEIGHTS_DIR).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    if args.retrain:
        if data_dir is None:
            sys.exit("[ERROR] --retrain requires --data-dir.")
        ensemble = train_ensemble(data_dir, weights_dir)
    elif model_mod.RailEnsemble.weights_exist(weights_dir):
        print_header(f"Loading pre-trained weights from {weights_dir}")
        print("  (pass --retrain --data-dir <dir> to retrain instead)")
        ensemble = model_mod.RailEnsemble.load(weights_dir)
    elif data_dir is not None:
        print_header(f"No saved weights at {weights_dir} — training from --data-dir")
        ensemble = train_ensemble(data_dir, weights_dir)
    else:
        sys.exit(
            f"[ERROR] No saved weights found at {weights_dir}, and no --data-dir given to train "
            f"from. Either point --weights-dir at an existing weights/ folder, or pass --data-dir "
            f"(add --retrain to force training even if weights already exist)."
        )

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
    X_input = feat_mod.build_feature_matrix(files).values
    predictions = ensemble.predict(X_input)
    pred_df = pd.DataFrame({"file_id": [f.name for f in files], "prediction": predictions})
    print(pred_df.to_string(index=False))
    print(f"\n  Prediction distribution: {pred_df['prediction'].value_counts().to_dict()}")

    pred_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(pred_df)} predictions to {args.output}")
    print(f"  Columns: {list(pred_df.columns)}  (matches 04_Example_Submission/rail_predictions.csv)")

    if args.diagnostics_output:
        import json
        proba = ensemble.predict_proba(X_input)
        classes = ensemble.label_encoder.classes_.tolist()
        diag = []
        for i, f in enumerate(files):
            _, sensor_matrix = feat_mod.load_file(f)
            diag.append({
                "file_id": f.name,
                "probabilities": {cls: float(proba[i, j]) for j, cls in enumerate(classes)},
                "speed_kmh": float(X_input[i, feat_mod.FEATURE_NAMES.index("speed_kmh")]),
                "worst_offender": feat_mod.worst_offender(sensor_matrix),
            })
        args.diagnostics_output.write_text(json.dumps(diag, indent=2))
        print(f"  Wrote per-file diagnostics to {args.diagnostics_output}")


if __name__ == "__main__":
    main()
