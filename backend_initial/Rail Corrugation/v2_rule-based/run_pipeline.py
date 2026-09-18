"""
Rail Corrugation Subsystem (rule-based) — Single-Command Pipeline
=======================================================================
Run this one file to go from raw sensor CSVs to a submission-ready
predictions file, with diagnostics printed along the way:

    python run_pipeline.py --data-dir /path/to/Rail_Corrugation

That path must contain Train/, Train_Labels.csv, and (by default) Test/.
Everything else is optional:

    python run_pipeline.py --data-dir /path/to/Rail_Corrugation \\
        --input /path/to/Rail_Corrugation/Test \\
        --output rail_predictions.csv

--input accepts either a directory (every *.csv inside it is predicted,
one row each) or a single CSV file.

What it does, in order
-----------------------
1. Extracts the 41-feature vector (features.py) for every labelled
   training file.
2. Runs the self-check: repeated stratified 5-fold cross-validation of
   the full CatBoost+XGBoost+LogReg soft-voting ensemble (model.py),
   scored with the real competition metric -- macro F1 -- plus a
   per-class F1 breakdown and confusion matrix (diagnostics.py), since
   this subsystem's severe class imbalance (Normal:Side I:Side II =
   234:14:24) makes the aggregate number alone misleading.
3. Fits the final ensemble on all 272 training files.
4. Predicts on every file under --input.
5. Writes rail_predictions.csv in the exact submission schema: file_id,
   prediction (string labels, not encoded integers).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import features as feat_mod
import model as model_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rail Corrugation subsystem — rule-based soft-voting ensemble, one-command run."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing Train/, Train_Labels.csv, and Test/.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="File or directory to predict on. Defaults to <data-dir>/Test/.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("rail_predictions.csv"),
        help="Where to write the submission CSV (default: ./rail_predictions.csv).",
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


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    input_path = (args.input or data_dir / "Test").resolve()

    labels_path = data_dir / "Train_Labels.csv"
    if not labels_path.exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")
    if not input_path.exists():
        sys.exit(f"[ERROR] Input path not found: {input_path}")

    labels = pd.read_csv(labels_path)

    # -----------------------------------------------------------------
    # 1. Feature extraction for every labelled training file
    # -----------------------------------------------------------------
    print_header(f"STEP 1/4 — Extracting features for {len(labels)} training files")
    train_paths = [data_dir / "Train" / fn for fn in labels["filename"]]
    X_train = feat_mod.build_feature_matrix(train_paths).values
    y_train = labels["label"].values
    print(f"  Feature matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")
    print(f"  Class counts: { {c: int((y_train == c).sum()) for c in sorted(set(y_train))} }")

    # -----------------------------------------------------------------
    # 2. Self-check: repeated stratified 5-fold CV vs. the real metric
    # -----------------------------------------------------------------
    print_header("STEP 2/4 — Self-check: 5x5 stratified CV, scored by macro F1")
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

    # -----------------------------------------------------------------
    # 3. Fit final ensemble on ALL training data
    # -----------------------------------------------------------------
    print_header("STEP 3/4 — Fitting final ensemble on all training files")
    ensemble = model_mod.RailEnsemble().fit(X_train, y_train)
    print(f"  Fitted CatBoost, XGBoost, and LogReg on all {X_train.shape[0]} training files.")

    # -----------------------------------------------------------------
    # 4. Predict on the target input file(s)
    # -----------------------------------------------------------------
    print_header(f"STEP 4/4 — Predicting on {input_path}")
    files = resolve_input_files(input_path)
    X_input = feat_mod.build_feature_matrix(files).values
    predictions = ensemble.predict(X_input)
    pred_df = pd.DataFrame({"file_id": [f.name for f in files], "prediction": predictions})
    print(pred_df.to_string(index=False))
    print(f"\n  Prediction distribution: {pred_df['prediction'].value_counts().to_dict()}")

    # -----------------------------------------------------------------
    # Write submission file
    # -----------------------------------------------------------------
    pred_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(pred_df)} predictions to {args.output}")
    print(f"  Columns: {list(pred_df.columns)}  (matches 04_Example_Submission/rail_predictions.csv)")


if __name__ == "__main__":
    main()
