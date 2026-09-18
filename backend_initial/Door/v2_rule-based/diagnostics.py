"""
Door Subsystem (rule-based) — Diagnostics
============================================
Three independent checks, all run before predictions on a new file are
trusted:

1. iou_weighted_f1()   — the ACTUAL competition metric (Door info kit,
   Section 4), computed against Train_Segments_Answer.csv, so there is a
   local number that means what the leaderboard means. Nothing else in
   this pipeline reports a metric that isn't this one.

2. self_check_on_train() — runs the FULL pipeline (segmentation.py +
   rules.py) on the raw Train.csv stream and scores it with (1) against
   the answer file. This is deliberately not the same as re-deriving
   features from the answer file's own start/end times — it measures
   what actually happens end-to-end, including any segmentation error.

3. flag_out_of_range() — Test has no labels, so it can't be scored. What
   it CAN be checked for is whether each segment's decision-relevant
   features (the ones rules.classify() actually reads) fall inside the
   range Train demonstrated for that operation. This does not tell you a
   prediction is wrong — it tells you the prediction is an extrapolation,
   which is a materially different and weaker kind of confidence.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import rules
import segmentation as seg_mod

DECISION_FEATURES = ["cur_max", "cur_mean", "cur_early_mean", "pos_max"]


# ---------------------------------------------------------------------------
# 1. The real competition metric
# ---------------------------------------------------------------------------
def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def iou_weighted_f1(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """
    Door info kit Section 4: IoU-weighted F1.

    Parameters
    ----------
    true_df, pred_df : Each needs 'start_s', 'end_s' (seconds, any common
        epoch) and 'label' columns.

    Returns
    -------
    dict with score, soft_recall, soft_precision, n_true, n_pred, n_matches.
    """
    candidates = []
    for ti, t in true_df.iterrows():
        for pi, p in pred_df.iterrows():
            if t["label"] != p["label"]:
                continue
            iou = _iou(t["start_s"], t["end_s"], p["start_s"], p["end_s"])
            if iou > 0:
                candidates.append((iou, ti, pi))
    candidates.sort(reverse=True)  # highest IoU first

    used_true, used_pred = set(), set()
    matched_iou_sum = 0.0
    n_matches = 0
    for iou, ti, pi in candidates:
        if ti in used_true or pi in used_pred:
            continue
        used_true.add(ti)
        used_pred.add(pi)
        matched_iou_sum += iou
        n_matches += 1

    n_true, n_pred = len(true_df), len(pred_df)
    soft_recall = matched_iou_sum / n_true if n_true else 0.0
    soft_precision = matched_iou_sum / n_pred if n_pred else 0.0
    denom = soft_recall + soft_precision
    score = 2 * soft_recall * soft_precision / denom if denom > 0 else 0.0

    return {
        "score": score,
        "soft_recall": soft_recall,
        "soft_precision": soft_precision,
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matches": n_matches,
    }


# ---------------------------------------------------------------------------
# 2. Full-pipeline self-check on Train.csv
# ---------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run segmentation + classification on a raw sensor DataFrame end to end.

    Returns
    -------
    DataFrame with one row per detected cycle: start_time, end_time,
    start_s, end_s, operation, prediction, plus every feature from
    rules.extract_features().
    """
    cycles = seg_mod.detect_cycles(df)
    plausible, implausible = seg_mod.flag_implausible(cycles)

    rows = []
    for start_idx, end_idx in plausible:
        seg = df.iloc[start_idx : end_idx + 1]
        operation = rules.get_operation(seg)
        label, feats = rules.classify(seg, operation)
        row = {
            "start_time": df.iloc[start_idx]["Datetime"],
            "end_time": df.iloc[end_idx]["Datetime"],
            "start_s": seg["ts"].iloc[0].timestamp(),
            "end_s": seg["ts"].iloc[-1].timestamp(),
            "operation": operation,
            "prediction": label,
        }
        row.update(feats)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["n_implausible"] = len(implausible)
    result.attrs["implausible_segments"] = implausible
    return result


def self_check_on_train(data_dir: Path) -> dict:
    """
    Run the full pipeline on Train.csv and score it against
    Train_Segments_Answer.csv using the real competition metric.
    """
    df = seg_mod.load_sensor_csv(data_dir / "Train.csv")
    ans = pd.read_csv(data_dir / "Train_Segments_Answer.csv")
    ans["start_s"] = ans["start_time"].apply(seg_mod.parse_ts).apply(lambda t: t.timestamp())
    ans["end_s"] = ans["end_time"].apply(seg_mod.parse_ts).apply(lambda t: t.timestamp())
    ans = ans.rename(columns={"status": "label"})

    pred = run_pipeline(df)
    pred_for_scoring = pred.rename(columns={"prediction": "label"})

    metrics = iou_weighted_f1(ans, pred_for_scoring)
    metrics["n_implausible"] = pred.attrs.get("n_implausible", 0)

    # Classification-only accuracy, conditional on segmentation being right,
    # to separate "did we find the cycle" from "did we label it right".
    correct_labels = 0
    for _, t in ans.iterrows():
        best = None
        for _, p in pred_for_scoring.iterrows():
            iou = _iou(t["start_s"], t["end_s"], p["start_s"], p["end_s"])
            if iou > 0.5 and (best is None or iou > best[0]):
                best = (iou, p["label"])
        if best is not None and best[1] == t["label"]:
            correct_labels += 1
    metrics["classification_accuracy_given_matched_segment"] = (
        correct_labels / len(ans) if len(ans) else 0.0
    )
    return metrics


# ---------------------------------------------------------------------------
# 3. Out-of-range flagging on a new (unlabeled) file
# ---------------------------------------------------------------------------
def build_train_ranges(data_dir: Path) -> pd.DataFrame:
    """Per-operation [min, max] of every decision-relevant feature, from Train."""
    df = seg_mod.load_sensor_csv(data_dir / "Train.csv")
    ans = pd.read_csv(data_dir / "Train_Segments_Answer.csv")

    seg_idx = np.zeros(len(df), dtype=int)
    cursor = 0
    for i, n in enumerate(ans["n_rows"]):
        seg_idx[cursor : cursor + n] = i
        cursor += n
    df = df.copy()
    df["seg_idx"] = seg_idx

    rows = []
    for i, row in ans.iterrows():
        seg = df[df["seg_idx"] == i]
        feats = rules.extract_features(seg)
        feats["operation"] = row["operation"]
        rows.append(feats)
    feat_df = pd.DataFrame(rows)

    ranges = feat_df.groupby("operation")[DECISION_FEATURES].agg(["min", "max"])
    return ranges


def flag_out_of_range(pred_df: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """
    For every predicted segment, list which decision-relevant features
    (if any) fall outside the range Train demonstrated for that operation.

    Returns
    -------
    DataFrame, one row per FLAGGED segment (empty if none), with the
    violating feature name, its value, and Train's [min, max] for it.
    """
    flags = []
    for idx, row in pred_df.iterrows():
        op = row["operation"]
        if op not in ranges.index:
            continue
        for feat in DECISION_FEATURES:
            lo, hi = ranges.loc[op, (feat, "min")], ranges.loc[op, (feat, "max")]
            val = row[feat]
            if val < lo or val > hi:
                flags.append(
                    {
                        "segment_row": idx,
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "operation": op,
                        "prediction": row["prediction"],
                        "feature": feat,
                        "value": val,
                        "train_min": lo,
                        "train_max": hi,
                    }
                )
    return pd.DataFrame(flags)
