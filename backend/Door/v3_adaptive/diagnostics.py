"""
Door Subsystem (v3, adaptive) — Diagnostics
================================================
Same real-metric discipline as v2 (iou_weighted_f1, self_check_on_train),
plus one new check this version needs that v2 didn't:

flag_low_confidence() -- for every prediction, is the decision-relevant
value within the bootstrap confidence half-width of the threshold it was
judged against? If so, a different (equally legitimate) resample of the
same 15/40 training examples could plausibly have drawn the boundary on
the other side of this exact cycle. That's a materially weaker kind of
confidence than "outside the training range entirely" (v2's OOD flag,
still run here too) -- it means the call is close, not that it's
unprecedented.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import rules
import segmentation as seg_mod
from calibration import DECISION_FEATURE

DECISION_FEATURES = ["cur_max", "cur_mean", "cur_early_mean", "pos_max"]


# ---------------------------------------------------------------------------
# 1. The real competition metric (identical to v2)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Full-pipeline run, in stream order, updating rolling baselines as it goes
# ---------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, calib: dict, seed_windows: dict[str, list] | None = None) -> pd.DataFrame:
    """
    Run segmentation + adaptive classification on a raw sensor DataFrame,
    in stream order -- required, since each AdaptiveThreshold's rolling
    baseline depends on every prior cycle's predicted label.
    """
    cycles = seg_mod.detect_cycles(df)
    plausible, implausible = seg_mod.flag_implausible(cycles)

    thresholds = {
        op: rules.AdaptiveThreshold.from_calibration(
            calib[op], seed=(seed_windows or {}).get(op)
        )
        for op in ("Close", "Open")
    }
    # Purely observational -- what THIS run's own Normal-predicted cycles
    # looked like, for the end-of-run report. Never fed back into the
    # thresholds used above; see rules.classify_one's docstring for why.
    observed_normal = {"Close": [], "Open": []}

    rows = []
    for start_idx, end_idx in plausible:
        seg = df.iloc[start_idx: end_idx + 1]
        operation = rules.get_operation(seg)
        feats = rules.extract_features(seg)
        label, meta = rules.classify_one(feats, operation, thresholds)

        value = feats["cur_max"] if operation == "Close" else feats["cur_mean"]
        ci_half_width = calib[operation]["ci_half_width"]
        low_confidence = abs(value - meta["threshold_used"]) <= ci_half_width
        if label == rules.NORMAL:
            observed_normal[operation].append(value)

        row = {
            "start_time": df.iloc[start_idx]["Datetime"],
            "end_time": df.iloc[end_idx]["Datetime"],
            "start_s": seg["ts"].iloc[0].timestamp(),
            "end_s": seg["ts"].iloc[-1].timestamp(),
            "operation": operation,
            "prediction": label,
            "threshold_used": meta["threshold_used"],
            "baseline_used": meta["baseline_used"],
            "low_confidence": low_confidence,
        }
        row.update(feats)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["n_implausible"] = len(implausible)
    result.attrs["implausible_segments"] = implausible
    result.attrs["thresholds_used"] = {op: t.adjusted_threshold for op, t in thresholds.items()}
    result.attrs["observed_normal"] = observed_normal
    return result


def self_check_on_train(data_dir: Path, calib: dict) -> dict:
    """
    Run the full v3 pipeline on Train.csv, with EACH operation's rolling
    baseline starting from Train's own seed window (the same one shipped in
    the calibration artifact) -- so this measures the pipeline exactly as
    it will actually run, not a version with foreknowledge of Train.
    """
    df = seg_mod.load_sensor_csv(data_dir / "Train.csv")
    ans = pd.read_csv(data_dir / "Train_Segments_Answer.csv")
    ans["start_s"] = ans["start_time"].apply(seg_mod.parse_ts).apply(lambda t: t.timestamp())
    ans["end_s"] = ans["end_time"].apply(seg_mod.parse_ts).apply(lambda t: t.timestamp())
    ans = ans.rename(columns={"status": "label"})

    seed_windows = {op: calib[op]["seed_window"] for op in ("Close", "Open")}
    pred = run_pipeline(df, calib, seed_windows=seed_windows)
    pred_for_scoring = pred.rename(columns={"prediction": "label"})

    metrics = iou_weighted_f1(ans, pred_for_scoring)
    metrics["n_implausible"] = pred.attrs.get("n_implausible", 0)
    metrics["n_low_confidence"] = int(pred["low_confidence"].sum())

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
# 3. Out-of-range flagging (identical in spirit to v2, ranges shipped in calib)
# ---------------------------------------------------------------------------
def build_train_ranges(data_dir: Path) -> pd.DataFrame:
    from calibration import build_features_df
    fd = build_features_df(data_dir)
    ranges = fd.groupby("operation")[DECISION_FEATURES].agg(["min", "max"])
    return ranges


def save_train_ranges(path: Path, ranges: pd.DataFrame) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nested = {
        op: {feat: [float(ranges.loc[op, (feat, "min")]), float(ranges.loc[op, (feat, "max")])]
             for feat in DECISION_FEATURES}
        for op in ranges.index
    }
    path.write_text(json.dumps(nested, indent=2))


def load_train_ranges(path: Path) -> pd.DataFrame:
    import json
    nested = json.loads(Path(path).read_text())
    rows = {}
    for op, feats in nested.items():
        row = {}
        for feat, (lo, hi) in feats.items():
            row[(feat, "min")] = lo
            row[(feat, "max")] = hi
        rows[op] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def flag_out_of_range(pred_df: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for idx, row in pred_df.iterrows():
        op = row["operation"]
        if op not in ranges.index:
            continue
        for feat in DECISION_FEATURES:
            lo, hi = ranges.loc[op, (feat, "min")], ranges.loc[op, (feat, "max")]
            val = row[feat]
            if val < lo or val > hi:
                flags.append({
                    "segment_row": idx, "start_time": row["start_time"], "end_time": row["end_time"],
                    "operation": op, "prediction": row["prediction"], "feature": feat,
                    "value": val, "train_min": lo, "train_max": hi,
                })
    return pd.DataFrame(flags)
