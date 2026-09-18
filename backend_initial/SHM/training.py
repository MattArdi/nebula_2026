"""
SHM Subsystem — Training & Evaluation Script
=============================================
Loads all 64 training files, extracts features, runs 5-fold CV with the
chosen method, and reports the competition scoring metric.

Scoring metric
--------------
    score = max(0, 1 − MAPE)

where MAPE = mean(|y_true − y_pred| / |y_true|).
A score of 1.0 is perfect; 0.0 means MAPE ≥ 100%.

Note: MAPE heavily penalises files with very small true damage values
(< 0.05). Those files are the hardest to predict accurately and form
the main bottleneck for this subsystem.

Usage
-----
    python training.py --method ensemble       # Ridge+XGB ensemble (best, default)
    python training.py --method ridge          # Ridge only
    python training.py --method xgboost        # XGBoost only

    python training.py --method ensemble --data-dir /path/to/SHM
    python training.py --method ensemble --predict-test --output predictions.csv
    python training.py --method ensemble --quiet
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))
import features as feat_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS      = ("ensemble", "ridge", "xgboost")
N_FOLDS      = 5
RANDOM_STATE = 42
TRAIN_SUBDIR = "Train"
TEST_SUBDIR  = "Test"
LABEL_FILE   = "Train_Labels.csv"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return max(0.0, 1.0 - mape(y_true, y_pred))


def per_file_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filenames: list[str],
) -> pd.DataFrame:
    pct_err = np.abs(y_true - y_pred) / np.abs(y_true) * 100
    return pd.DataFrame({
        "filename":  filenames,
        "true":      np.round(y_true, 4),
        "pred":      np.round(y_pred, 4),
        "pct_error": np.round(pct_err, 2),
    }).sort_values("pct_error", ascending=False)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filenames: list[str],
) -> dict:
    return {
        "mape":             mape(y_true, y_pred),
        "score":            competition_score(y_true, y_pred),
        "per_file":         per_file_errors(y_true, y_pred, filenames),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels(data_dir: Path) -> pd.DataFrame:
    path = data_dir / LABEL_FILE
    if not path.exists():
        sys.exit(f"[ERROR] {LABEL_FILE} not found in {data_dir}")
    return pd.read_csv(path)


def build_feature_matrix(
    data_dir: Path,
    labels: pd.DataFrame,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load every training signal, extract features, return (X, y, filenames).

    Returns
    -------
    X         : Feature matrix (n_samples, n_features).
    y         : Damage values array (n_samples,), dtype float.
    filenames : List of filenames in row order.
    """
    rows      = []
    y_vals    = []
    filenames = []

    for i, (_, row) in enumerate(labels.iterrows()):
        fpath = data_dir / TRAIN_SUBDIR / row["filename"]
        if not fpath.exists():
            print(f"[WARN] {fpath} not found — skipping.")
            continue

        sig   = feat_mod.load_signal(fpath)
        feats = feat_mod.extract(sig)
        rows.append(feats)
        y_vals.append(float(row["damage"]))
        filenames.append(row["filename"])

        if verbose and (i + 1) % 16 == 0:
            print(f"  Loaded {i + 1}/{len(labels)} files...")

    X = pd.DataFrame(rows).values.astype(float)
    y = np.array(y_vals, dtype=float)
    return X, y, filenames


def build_test_features(
    data_dir: Path,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Load all test signals and extract features."""
    test_dir = data_dir / TEST_SUBDIR
    fnames   = sorted(
        test_dir.glob("*.csv"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)))
    )

    rows      = []
    filenames = []
    for i, fpath in enumerate(fnames):
        sig   = feat_mod.load_signal(fpath)
        feats = feat_mod.extract(sig)
        rows.append(feats)
        filenames.append(fpath.name)
        if verbose and (i + 1) % 8 == 0:
            print(f"  Loaded test {i + 1}/{len(fnames)} files...")

    return pd.DataFrame(rows).values.astype(float), filenames


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
    y: np.ndarray,
    filenames: list[str],
    method_mod,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> dict:
    """
    Run k-fold CV and return aggregated metrics.

    Note: KFold (not stratified) is used because this is a regression task.
    The damage distribution is right-skewed, so folds are shuffled to avoid
    ordering artefacts.

    Parameters
    ----------
    X          : Feature matrix.
    y          : Damage values.
    filenames  : Filenames matching X rows (for error reporting).
    method_mod : Imported method module (must expose fit() and predict()).

    Returns
    -------
    cv_results : Dict with fold scores, mean/std, and per-file errors.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []
    all_true    = []
    all_pred    = []
    all_files   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model  = method_mod.fit(X_tr, y_tr)
        y_pred = method_mod.predict(model, X_val)

        fold_score = competition_score(y_val, y_pred)
        fold_mape  = mape(y_val, y_pred)
        fold_scores.append(fold_score)

        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())
        all_files.extend([filenames[i] for i in val_idx])

        if verbose:
            print(
                f"  Fold {fold + 1}/{n_folds}  "
                f"MAPE={fold_mape:.4f}  score={fold_score:.4f}"
            )

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)

    metrics             = compute_metrics(all_true, all_pred, all_files)
    metrics["fold_scores"] = fold_scores
    metrics["mean_score"]  = float(np.mean(fold_scores))
    metrics["std_score"]   = float(np.std(fold_scores))
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

    # --- Feature extraction ---
    print(f"\nLoading training data from {data_dir / TRAIN_SUBDIR} ...")
    labels = load_labels(data_dir)
    X, y, filenames = build_feature_matrix(data_dir, labels, verbose=verbose)
    print(f"Feature matrix : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Damage range   : [{y.min():.4f}, {y.max():.4f}]")
    print(f"  < 0.05 : {(y < 0.05).sum()} files  (hardest for MAPE)")
    print(f"  ≥ 0.50 : {(y >= 0.5).sum()} files")

    # --- Cross-validation ---
    print(f"\nRunning {N_FOLDS}-fold CV  [method={method}] ...")
    cv_results = cross_validate(X, y, filenames, method_mod, verbose=verbose)

    # --- Optionally predict on test set ---
    if predict_test:
        print(f"\nLoading test data from {data_dir / TEST_SUBDIR} ...")
        X_test, test_fnames = build_test_features(data_dir, verbose=verbose)

        print("Fitting final model on all training data ...")
        final_model = method_mod.fit(X, y)
        y_test_pred = method_mod.predict(final_model, X_test)

        pred_df = pd.DataFrame({
            "file_id":    test_fnames,
            "prediction": np.round(y_test_pred, 4),
        })

        if output:
            pred_df.to_csv(output, index=False)
            print(f"[INFO] Predictions written to {output}")
        else:
            print("\nTest predictions:")
            print(pred_df.to_string(index=False))

    return cv_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHM subsystem — train and evaluate fatigue damage regression models."
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="ensemble",
        help=(
            "Regression method:\n"
            "  ensemble — Ridge(70%%) + XGBoost(30%%) (best, default)  score ≈ 0.840\n"
            "  ridge    — Ridge regression, log target               score ≈ 0.829\n"
            "  xgboost  — XGBoost, log target                        score ≈ 0.817"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "../../02_Datasets/SHM",
        help="Path to SHM dataset directory (must contain Train/ and Train_Labels.csv).",
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
    print("SHM Training")
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
    print(f"  Score  (mean ± std) : {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"  MAPE                : {results['mape']:.4f}")
    print(f"  Per-fold scores     : {[round(s, 4) for s in results['fold_scores']]}")
    print()
    print("  Worst predictions (by % error):")
    print(results["per_file"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
