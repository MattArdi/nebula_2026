"""
Rail Corrugation Subsystem — Training & Evaluation Script
==========================================================
Loads all 272 training files, extracts features, runs stratified 5-fold
cross-validation with the chosen method, and reports macro F1.

Scoring metric: macro-averaged F1 (equal weight per class regardless of
frequency — critical because Normal:Side I:Side II = 234:14:24).

Usage
-----
    python training.py --method xgboost          # XGBoost + SMOTE (best, default)
    python training.py --method random_forest    # Random Forest + SMOTE (baseline)

    # Custom data directory
    python training.py --method xgboost --data-dir /path/to/Rail_Corrugation

    # Write test-set predictions to CSV
    python training.py --method xgboost --predict-test --output predictions.csv
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Make sure sibling modules (features, method_*) are importable regardless
# of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))
import features as feat_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS      = ("xgboost", "random_forest")
N_FOLDS      = 5
RANDOM_STATE = 42
TRAIN_SUBDIR = "Train"
TEST_SUBDIR  = "Test"
LABEL_FILE   = "Train_Labels.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _file_number(fname: str) -> int:
    """Extract the integer from 'Train42.csv' or 'Test7.csv'."""
    stem = Path(fname).stem          # e.g. 'Train42'
    return int("".join(filter(str.isdigit, stem)))


def _train_folder(data_dir: Path, fname: str) -> Path:
    """Route a training filename to Train 1/ or Train 2/ subfolder."""
    num = _file_number(fname)
    subfolder = "Train 1" if num <= 136 else "Train 2"
    return data_dir / TRAIN_SUBDIR / subfolder / fname


def load_labels(data_dir: Path) -> pd.DataFrame:
    path = data_dir / LABEL_FILE
    if not path.exists():
        sys.exit(f"[ERROR] {LABEL_FILE} not found in {data_dir}")
    return pd.read_csv(path)


def build_feature_matrix(
    data_dir: Path, labels: pd.DataFrame, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load every training file, extract features, return (X, y, filenames).

    Parameters
    ----------
    data_dir : ACV dataset root (contains Train_Labels.csv + Train/).
    labels   : DataFrame with columns ['filename', 'label'].
    verbose  : Print progress.

    Returns
    -------
    X         : Feature matrix (n_samples, n_features).
    y_str     : String labels array (n_samples,).
    filenames : List of filenames in row order.
    """
    rows      = []
    y_str     = []
    filenames = []

    for i, (_, row) in enumerate(labels.iterrows()):
        fpath = _train_folder(data_dir, row["filename"])
        if not fpath.exists():
            print(f"[WARN] {fpath} not found — skipping.")
            continue

        df    = pd.read_csv(fpath)
        feats = feat_mod.extract(df)
        rows.append(feats)
        y_str.append(row["label"])
        filenames.append(row["filename"])

        if verbose and (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(labels)} files...")

    X = pd.DataFrame(rows).values.astype(float)
    return X, np.array(y_str), filenames


def build_test_features(data_dir: Path, verbose: bool = True) -> tuple[np.ndarray, list[str]]:
    """Load all test files and extract features."""
    test_dir = data_dir / TEST_SUBDIR
    fnames   = sorted(test_dir.glob("*.csv"), key=lambda p: _file_number(p.name))

    rows      = []
    filenames = []
    for i, fpath in enumerate(fnames):
        df    = pd.read_csv(fpath)
        feats = feat_mod.extract(df)
        rows.append(feats)
        filenames.append(fpath.name)
        if verbose and (i + 1) % 20 == 0:
            print(f"  Loaded test {i + 1}/{len(fnames)} files...")

    return pd.DataFrame(rows).values.astype(float), filenames


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """
    Compute macro F1 and per-class F1 from integer-encoded arrays.

    Parameters
    ----------
    y_true  : Ground-truth integer labels.
    y_pred  : Predicted integer labels.
    classes : Class names in label-encoder order.

    Returns
    -------
    metrics : Dict with macro_f1, per_class_f1, confusion_matrix, report.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class = {
        cls: float(f1_score(y_true, y_pred, labels=[i], average="macro"))
        for i, cls in enumerate(classes)
    }
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    return {
        "macro_f1":     macro_f1,
        "per_class_f1": per_class,
        "confusion_matrix": cm,
        "report":       report,
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate(
    X: np.ndarray,
    y_enc: np.ndarray,
    le: LabelEncoder,
    method_mod,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> dict:
    """
    Run stratified k-fold CV and return aggregated metrics.

    Parameters
    ----------
    X          : Feature matrix.
    y_enc      : Integer-encoded labels.
    le         : Fitted LabelEncoder (for class names).
    method_mod : Imported method module (must expose fit() and predict()).
    n_folds    : Number of folds.
    verbose    : Print per-fold scores.

    Returns
    -------
    cv_results : Dict with fold scores and aggregated metrics.
    """
    cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    classes = list(le.classes_)

    fold_scores = []
    all_true    = []
    all_pred    = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        pipeline = method_mod.fit(X_tr, y_tr)
        y_pred   = method_mod.predict(pipeline, X_val)

        fold_f1 = float(f1_score(y_val, y_pred, average="macro"))
        fold_scores.append(fold_f1)
        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

        if verbose:
            print(f"  Fold {fold + 1}/{n_folds}  macro F1 = {fold_f1:.4f}")

    metrics = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    metrics["fold_scores"] = fold_scores
    metrics["mean_f1"]     = float(np.mean(fold_scores))
    metrics["std_f1"]      = float(np.std(fold_scores))
    return metrics


# ---------------------------------------------------------------------------
# Method loading
# ---------------------------------------------------------------------------
def load_method(method_name: str):
    try:
        return importlib.import_module(f"method_{method_name}")
    except ModuleNotFoundError:
        sys.exit(f"[ERROR] Unknown method '{method_name}'. Choose from: {METHODS}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run(
    data_dir: Path,
    method: str,
    predict_test: bool,
    output: Path | None,
    verbose: bool,
) -> dict:
    method_mod = load_method(method)

    # --- Feature extraction ---
    print(f"\nLoading training data from {data_dir / TRAIN_SUBDIR} ...")
    labels = load_labels(data_dir)
    X, y_str, filenames = build_feature_matrix(data_dir, labels, verbose=verbose)
    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    print(f"Classes: {list(le.classes_)}")
    print(f"Class counts: { {c: int((y_str == c).sum()) for c in le.classes_} }")

    # --- Cross-validation ---
    print(f"\nRunning {N_FOLDS}-fold stratified CV  [method={method}] ...")
    cv_results = cross_validate(X, y_enc, le, method_mod, verbose=verbose)

    # --- Optionally predict on test set ---
    if predict_test:
        print(f"\nLoading test data from {data_dir / TEST_SUBDIR} ...")
        X_test, test_fnames = build_test_features(data_dir, verbose=verbose)

        print("Fitting final model on all training data ...")
        final_pipeline = method_mod.fit(X, y_enc)
        y_test_pred    = method_mod.predict(final_pipeline, X_test)
        y_test_labels  = le.inverse_transform(y_test_pred)

        pred_df = pd.DataFrame({"file_id": test_fnames, "prediction": y_test_labels})

        if output:
            pred_df.to_csv(output, index=False)
            print(f"[INFO] Predictions written to {output}")
        else:
            print("\nTest predictions:")
            print(pred_df.to_string(index=False))

        print(f"\nTest prediction distribution:")
        print(pred_df["prediction"].value_counts().to_string())

    return cv_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rail Corrugation subsystem — train and evaluate classifiers."
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="xgboost",
        help=(
            "Training method:\n"
            "  xgboost       — XGBoost + SMOTE (best, default)  CV macro F1 ≈ 0.850\n"
            "  random_forest — Random Forest + SMOTE (baseline) CV macro F1 ≈ 0.686"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "../../../02_Datasets/Rail_Corrugation",
        help="Path to Rail_Corrugation dataset directory.",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="After CV, fit on all training data and predict the test set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path for test predictions (requires --predict-test).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-fold and per-file output.",
    )
    return parser.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.exists():
        sys.exit(f"[ERROR] Data directory not found: {data_dir}")

    print("=" * 60)
    print(f"Rail Corrugation Training")
    print(f"  method   : {args.method}")
    print(f"  data_dir : {data_dir}")
    print("=" * 60)

    results = run(
        data_dir=data_dir,
        method=args.method,
        predict_test=args.predict_test,
        output=args.output,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)
    print(f"  Macro F1 (mean ± std) : {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"  Per-fold scores       : {[round(s, 4) for s in results['fold_scores']]}")
    print()
    print("  Per-class F1:")
    for cls, f1 in results["per_class_f1"].items():
        print(f"    {cls:<22} {f1:.4f}")
    print()
    print("  Classification Report (aggregated over all CV folds):")
    print(results["report"])
    print("  Confusion Matrix (rows=true, cols=pred):")
    classes = list(results["per_class_f1"].keys())
    cm_df   = pd.DataFrame(
        results["confusion_matrix"], index=classes, columns=classes
    )
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
