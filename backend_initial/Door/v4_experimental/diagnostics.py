"""
Door Subsystem (v4, EXPERIMENTAL) — Diagnostics
====================================================
Same real-metric discipline as v2/v3 (iou_weighted_f1, self_check_on_train),
plus a diff-against-v3 report -- since v4's entire purpose is to be judged
against v3's already-validated output, not scored in isolation.
"""

from pathlib import Path

import pandas as pd

import rules
import segmentation as seg_mod


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def iou_weighted_f1(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    candidates = []
    for ti, t in true_df.iterrows():
        for pi, p in pred_df.iterrows():
            if t["label"] != p["label"]:
                continue
            iou = _iou(t["start_s"], t["end_s"], p["start_s"], p["end_s"])
            if iou > 0:
                candidates.append((iou, ti, pi))
    candidates.sort(reverse=True)

    used_true, used_pred = set(), set()
    matched_iou_sum, n_matches = 0.0, 0
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
        "score": score, "soft_recall": soft_recall, "soft_precision": soft_precision,
        "n_true": n_true, "n_pred": n_pred, "n_matches": n_matches,
    }


def run_pipeline(df: pd.DataFrame, open_class_stats: dict) -> pd.DataFrame:
    cycles = seg_mod.detect_cycles(df)
    plausible, implausible = seg_mod.flag_implausible(cycles)

    rows = []
    for start_idx, end_idx in plausible:
        seg = df.iloc[start_idx: end_idx + 1]
        operation = rules.get_operation(seg)
        feats = rules.extract_features(seg)
        label, meta = rules.classify(feats, operation, open_class_stats)
        row = {
            "start_time": df.iloc[start_idx]["Datetime"],
            "end_time": df.iloc[end_idx]["Datetime"],
            "start_s": seg["ts"].iloc[0].timestamp(),
            "end_s": seg["ts"].iloc[-1].timestamp(),
            "operation": operation,
            "prediction": label,
        }
        row.update(feats)
        row.update(meta)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["n_implausible"] = len(implausible)
    return result


def self_check_on_train(data_dir: Path, open_class_stats: dict) -> dict:
    df = seg_mod.load_sensor_csv(data_dir / "Train.csv")
    ans = pd.read_csv(data_dir / "Train_Segments_Answer.csv")
    ans["start_s"] = ans["start_time"].apply(seg_mod.parse_ts).apply(lambda t: t.timestamp())
    ans["end_s"] = ans["end_time"].apply(seg_mod.parse_ts).apply(lambda t: t.timestamp())
    ans = ans.rename(columns={"status": "label"})

    pred = run_pipeline(df, open_class_stats)
    pred_for_scoring = pred.rename(columns={"prediction": "label"})
    metrics = iou_weighted_f1(ans, pred_for_scoring)
    metrics["n_implausible"] = pred.attrs.get("n_implausible", 0)
    return metrics


def diff_against_v3(pred_v4: pd.DataFrame, v3_predictions_path: Path) -> pd.DataFrame:
    """Row-align by start_time and report every segment where v4 disagrees with
    v3's already-validated Test.csv predictions."""
    v3 = pd.read_csv(v3_predictions_path)
    merged = pred_v4[["start_time", "operation", "prediction"]].merge(
        v3[["start_time", "prediction"]], on="start_time", suffixes=("_v4", "_v3")
    )
    return merged[merged["prediction_v4"] != merged["prediction_v3"]]
