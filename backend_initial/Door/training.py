"""
Door Subsystem — Training & Evaluation Script
=============================================
Two-stage pipeline:
  Stage 1 — Segmentation: rule-based cycle boundary detection.
  Stage 2 — Classification: per-cycle Normal / Abnormal resistance.

Scoring metric
--------------
The competition uses IoU-weighted F1 — timing accuracy (segment overlap)
and label correctness both matter. For CV purposes we report macro F1
on the labelled segments (Stage 2 only), since Stage 1 achieves perfect
boundary recovery on training data (110/110, 0 false positives).

An additional segmentation validation step checks the detected boundaries
against the answer file and reports any mismatches.

Usage
-----
    python training.py --method random_forest       # RF classifier (best, default)
    python training.py --method gradient_boosting   # GBM classifier (baseline)

    python training.py --method random_forest --data-dir /path/to/Door
    python training.py --method random_forest --predict-test --output predictions.csv
    python training.py --method random_forest --quiet
"""

import argparse
import importlib
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

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))
import features    as feat_mod  # noqa: E402
import segmentation as seg_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS      = ("random_forest", "gradient_boosting")
N_FOLDS      = 5
RANDOM_STATE = 42
TRAIN_FILE   = "Train.csv"
TEST_FILE    = "Test.csv"
ANSWER_FILE  = "Train_Segments_Answer.csv"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict:
    macro_f1  = float(f1_score(y_true, y_pred, average="macro"))
    per_class = {
        cls: float(f1_score(y_true, y_pred, labels=[i], average="macro"))
        for i, cls in enumerate(classes)
    }
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    return {
        "macro_f1":       macro_f1,
        "per_class_f1":   per_class,
        "confusion_matrix": cm,
        "report":         report,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_train(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sensor_df, answer_df)."""
    sensor_path = data_dir / TRAIN_FILE
    answer_path = data_dir / ANSWER_FILE

    if not sensor_path.exists():
        sys.exit(f"[ERROR] {TRAIN_FILE} not found in {data_dir}")
    if not answer_path.exists():
        sys.exit(f"[ERROR] {ANSWER_FILE} not found in {data_dir}")

    df  = seg_mod.add_timestamp_col(pd.read_csv(sensor_path))
    ans = pd.read_csv(answer_path)
    ans["start_ms"] = ans["start_time"].apply(seg_mod.parse_ts_ms)
    ans["end_ms"]   = ans["end_time"].apply(seg_mod.parse_ts_ms)
    return df, ans


def load_test(data_dir: Path) -> pd.DataFrame:
    path = data_dir / TEST_FILE
    if not path.exists():
        sys.exit(f"[ERROR] {TEST_FILE} not found in {data_dir}")
    return seg_mod.add_timestamp_col(pd.read_csv(path))


# ---------------------------------------------------------------------------
# Feature matrix from labelled segments
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    ans: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features from every labelled training segment.

    Parameters
    ----------
    df  : Full sensor DataFrame with 'ts_ms' column.
    ans : Answer DataFrame with 'start_ms', 'end_ms', 'operation', 'status'.

    Returns
    -------
    X     : Feature matrix (n_segments, n_features).
    y_str : String label array (n_segments,).
    """
    rows  = []
    y_str = []

    for _, seg in ans.iterrows():
        mask = (df["ts_ms"] >= seg["start_ms"]) & (df["ts_ms"] <= seg["end_ms"])
        d    = df[mask]
        feats = feat_mod.extract(d, seg["operation"])
        rows.append(feats)
        y_str.append(seg["status"])

    X = pd.DataFrame(rows).values.astype(float)
    return X, np.array(y_str)


# ---------------------------------------------------------------------------
# Segmentation validation
# ---------------------------------------------------------------------------
def validate_segmentation(
    df: pd.DataFrame,
    ans: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Compare rule-based segment boundaries to the ground-truth answer file.

    Parameters
    ----------
    df      : Sensor DataFrame with 'ts_ms'.
    ans     : Answer DataFrame with 'start_ms'.
    verbose : Print mismatch details.

    Returns
    -------
    seg_report : Dict with n_detected, n_expected, n_correct, n_fp, n_fn.
    """
    detected = seg_mod.detect_segments(df)
    detected_starts = {s for s, _ in detected}

    # Ground-truth start indices
    gt_starts = set()
    for _, seg in ans.iterrows():
        matches = df[df["ts_ms"] == seg["start_ms"]].index
        if len(matches):
            gt_starts.add(matches[0])

    tp = len(detected_starts & gt_starts)
    fp = len(detected_starts - gt_starts)
    fn = len(gt_starts - detected_starts)

    if verbose:
        print(f"  Segmentation: detected={len(detected)}  expected={len(ans)}"
              f"  correct={tp}  FP={fp}  FN={fn}")
        if fp or fn:
            print("  [WARN] Boundary mismatches detected — check segmentation rules.")

    return {
        "n_detected": len(detected),
        "n_expected": len(ans),
        "n_correct":  tp,
        "n_fp":       fp,
        "n_fn":       fn,
    }


# ---------------------------------------------------------------------------
# Method loading
# ---------------------------------------------------------------------------
def load_method(method_name: str):
    try:
        return importlib.import_module(f"method_{method_name}")
    except ModuleNotFoundError:
        sys.exit(f"[ERROR] Unknown method '{method_name}'. Choose from: {METHODS}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate(
    X: np.ndarray,
    y_str: np.ndarray,
    method_mod,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> dict:
    """
    Run stratified k-fold CV over the labelled segments.

    Note: Stratified split is important because Abnormal resistance is only
    ~27% of segments (30/110). Without stratification some folds may contain
    very few abnormal examples.

    Parameters
    ----------
    X          : Feature matrix (n_segments, n_features).
    y_str      : String labels.
    method_mod : Imported method module (exposes fit() and predict()).

    Returns
    -------
    cv_results : Dict with fold scores and aggregated metrics.
    """
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    classes = list(le.classes_)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []
    all_true    = []
    all_pred    = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        X_tr, X_val   = X[train_idx], X[val_idx]
        y_tr_s        = y_str[train_idx]
        y_val_enc     = y_enc[val_idx]

        model, fold_le = method_mod.fit(X_tr, y_tr_s)
        y_pred_s       = method_mod.predict(model, fold_le, X_val)
        y_pred_enc     = le.transform(y_pred_s)

        fold_f1 = float(f1_score(y_val_enc, y_pred_enc, average="macro"))
        fold_scores.append(fold_f1)
        all_true.extend(y_val_enc.tolist())
        all_pred.extend(y_pred_enc.tolist())

        if verbose:
            print(f"  Fold {fold + 1}/{n_folds}  macro F1 = {fold_f1:.4f}")

    metrics = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    metrics["fold_scores"] = fold_scores
    metrics["mean_f1"]     = float(np.mean(fold_scores))
    metrics["std_f1"]      = float(np.std(fold_scores))
    return metrics


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

    # --- Load training data ---
    print(f"\nLoading training data from {data_dir} ...")
    df_train, ans = load_train(data_dir)
    print(f"  Sensor stream : {len(df_train):,} rows")
    print(f"  Segments      : {len(ans)} labelled cycles")
    print(f"  Labels        : {dict(ans['status'].value_counts())}")

    # --- Validate segmentation ---
    print("\nValidating segmentation on training stream ...")
    seg_report = validate_segmentation(df_train, ans, verbose=verbose)

    # --- Build feature matrix from labelled segments ---
    X, y_str = build_feature_matrix(df_train, ans)
    print(f"\nFeature matrix : {X.shape[0]} segments × {X.shape[1]} features")

    # --- Cross-validation ---
    print(f"\nRunning {N_FOLDS}-fold stratified CV  [method={method}] ...")
    cv_results = cross_validate(X, y_str, method_mod, verbose=verbose)

    # --- Optionally predict on test set ---
    if predict_test:
        print(f"\nLoading test data from {data_dir / TEST_FILE} ...")
        df_test = load_test(data_dir)
        print(f"  Test stream : {len(df_test):,} rows")

        # Segment the test stream
        test_segments = seg_mod.detect_segments(df_test)
        print(f"  Detected {len(test_segments)} cycles in test stream")

        # Fit final model on all training segments
        print("Fitting final model on all training segments ...")
        le    = LabelEncoder()
        y_enc = le.fit_transform(y_str)
        final_model, final_le = method_mod.fit(X, y_str)

        # Extract features and classify each test cycle
        results = []
        for start_idx, end_idx in test_segments:
            d         = df_test.iloc[start_idx : end_idx + 1]
            operation = seg_mod.segment_operation(d)
            feats     = feat_mod.extract(d, operation)
            X_seg     = pd.DataFrame([feats]).values.astype(float)
            pred      = method_mod.predict(final_model, final_le, X_seg)[0]

            results.append({
                "start_time": df_test.iloc[start_idx]["Datetime"],
                "end_time":   df_test.iloc[end_idx]["Datetime"],
                "prediction": pred,
            })

        pred_df = pd.DataFrame(results)

        if output:
            pred_df.to_csv(output, index=False)
            print(f"[INFO] Predictions written to {output}")
        else:
            print("\nTest predictions:")
            print(pred_df.to_string(index=False))

        print(f"\nTest label distribution:")
        print(pred_df["prediction"].value_counts().to_string())

    cv_results["segmentation"] = seg_report
    return cv_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door subsystem — segment sensor stream and classify door cycles."
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="random_forest",
        help=(
            "Classification method:\n"
            "  random_forest      — RF balanced (best, default)  CV macro F1 = 1.000\n"
            "  gradient_boosting  — GBM (baseline)               CV macro F1 = 0.988"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "../../02_Datasets/Door",
        help="Path to Door dataset directory (must contain Train.csv, Test.csv, "
             "Train_Segments_Answer.csv).",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="After CV, segment the test stream and predict each cycle.",
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
        help="Suppress per-fold output.",
    )
    return parser.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.exists():
        sys.exit(f"[ERROR] Data directory not found: {data_dir}")

    print("=" * 60)
    print("Door Subsystem Training")
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

    seg = results["segmentation"]
    print(f"  Segmentation  : {seg['n_correct']}/{seg['n_expected']} correct "
          f"  FP={seg['n_fp']}  FN={seg['n_fn']}")
    print()
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
