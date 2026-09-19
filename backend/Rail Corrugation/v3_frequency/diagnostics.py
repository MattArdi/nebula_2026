"""
Rail Corrugation Subsystem (v3, frequency-augmented) — Diagnostics
========================================================================
Identical protocol to v2 -- this is deliberate: every number in
algorithm.md's Section 5 comparison table came from running this exact
function on v2's features and on v3's, nothing else changed, so the
delta between them is attributable to the added feature alone.

The self-check: repeated stratified k-fold cross-validation of the full
ensemble (features.py + model.py), scored with the real competition
metric -- macro F1, the unweighted mean of each class's F1 -- not
accuracy, which is a badly misleading number for this subsystem given
Normal:Side I:Side II are represented 234:14:24 in training.

Following the same protocol used to select the ensemble in the first
place: for each of N_REPEATS independent 5-fold StratifiedKFold splits,
predictions are pooled across all 5 folds of that repeat before scoring,
then the per-repeat scores are averaged. Pooling before scoring (rather
than averaging 5 small per-fold macro F1 numbers) avoids a fold with only
2-3 Side I examples producing a noisy, unstable per-fold F1.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import model as model_mod

N_REPEATS = 5
N_FOLDS = 5


def self_check_cv(X: np.ndarray, y_str: np.ndarray) -> dict:
    """
    Repeated stratified 5-fold CV of the full CatBoost+XGBoost+LogReg
    soft-voting ensemble. Returns per-repeat macro F1, their mean/std, and
    a pooled classification report + confusion matrix (pooled across every
    repeat and fold, for a stable per-class breakdown despite Side I/Side
    II's small sample sizes).
    """
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)

    repeat_scores = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        rep_true: list[int] = []
        rep_pred: list[int] = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

            cat = model_mod.make_catboost().fit(X_tr, y_tr)
            xgb = model_mod.make_xgboost().fit(X_tr, y_tr)
            log = model_mod.make_logreg().fit(X_tr_s, y_tr)

            proba = (
                model_mod.CATBOOST_WEIGHT * cat.predict_proba(X_val)
                + model_mod.XGBOOST_WEIGHT * xgb.predict_proba(X_val)
                + model_mod.LOGREG_WEIGHT * log.predict_proba(X_val_s)
            )
            pred = np.argmax(proba, axis=1)

            rep_true.extend(y_val.tolist())
            rep_pred.extend(pred.tolist())

        repeat_scores.append(f1_score(rep_true, rep_pred, average="macro"))
        all_true.extend(rep_true)
        all_pred.extend(rep_pred)

    per_class_f1 = {
        cls: float(f1_score(all_true, all_pred, labels=[i], average="macro"))
        for i, cls in enumerate(classes)
    }
    cm = pd.DataFrame(
        confusion_matrix(all_true, all_pred, labels=list(range(len(classes)))),
        index=classes, columns=classes,
    )
    return {
        "classes": classes,
        "repeat_scores": repeat_scores,
        "mean_f1": float(np.mean(repeat_scores)),
        "std_f1": float(np.std(repeat_scores)),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "report": classification_report(all_true, all_pred, target_names=classes),
    }
