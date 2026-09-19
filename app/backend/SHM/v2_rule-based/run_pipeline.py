"""
SHM Subsystem (rule-based) — Single-Command Pipeline
=========================================================
Run this one file to go from raw stress CSVs to a submission-ready
predictions file:

    python run_pipeline.py --input /path/to/Test

By default this LOADS the pre-fitted calibration constant already
committed in artifacts/calibration.json (C, the fixed exponent m, and
the training proxy range used for extrapolation flagging) and predicts
immediately -- no --data-dir, no training data, no refitting. This is
what makes "clone the repo, run on real data" work with nothing else on
disk.

--input accepts either a directory (every *.csv inside it is predicted,
one row each -- the normal case, since SHM's Test/ holds 16 separate
files) or a single CSV file.

To refit instead of loading the shipped calibration (e.g. after changing
physics.py or model.py, or to verify the shipped constant still
reproduces), pass --retrain together with --data-dir:

    python run_pipeline.py --retrain --data-dir /path/to/SHM \\
        --input /path/to/SHM/Test --output shm_predictions.csv

That path must contain Train/ and Train_Labels.csv. Retraining does, in
order:
1. Runs the full pipeline (physics.py + model.py) through leave-one-out
   cross-validation against Train_Labels.csv, scored with the real
   competition metric (max(0, 1-MAPE)) -- the one trustworthy estimate
   of how the pipeline performs on unseen files.
2. Refits C on all 64 training files and OVERWRITES artifacts/ with the
   result -- --retrain is destructive to the shipped calibration by
   design, so future default runs pick up the refitted constant.

Either way, the final step is the same: predicts on every file under
--input, flags any prediction whose Miner's-rule proxy falls outside the
range Train ever demonstrated, and writes shm_predictions.csv in the
exact submission schema: file_id, prediction.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import model as model_mod
import physics

DEFAULT_ARTIFACTS_PATH = Path(__file__).parent / "artifacts" / "calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHM subsystem — rule-based Miner's-rule damage prediction, one-command run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="File or directory to predict on. Required unless --data-dir sets a default via Test/.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("shm_predictions.csv"),
        help="Where to write the submission CSV (default: ./shm_predictions.csv).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing Train/ and Train_Labels.csv. Only needed with --retrain "
             "(or if --input is omitted, to default it to <data-dir>/Test).",
    )
    parser.add_argument(
        "--artifacts-path", type=Path, default=None,
        help=f"Where the calibration JSON is loaded from / saved to (default: {DEFAULT_ARTIFACTS_PATH}).",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Refit C from --data-dir instead of loading the shipped calibration, and overwrite "
             "--artifacts-path with the result. Requires --data-dir.",
    )
    parser.add_argument(
        "--diagnostics-output", type=Path, default=None,
        help="Optional: write a JSON array alongside --output, one object per file (same order), "
             "with values already computed but that don't belong in the submission CSV -- "
             "remaining fatigue life (1 - damage, as a %%), an error band from the shipped "
             "calibration's leave-one-out MAPE (null if the shipped calibration predates that "
             "field), the extrapolation flag/reason, and a damage-build-up curve recomputed on "
             "growing prefixes of the file via real rainflow counting. Purely additive: "
             "--output's contents are identical whether or not this is passed.",
    )
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def refit_calibration(data_dir: Path, artifacts_path: Path) -> dict:
    if not (data_dir / "Train_Labels.csv").exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")

    proxy_df = physics.compute_train_proxies(data_dir)

    print_header("STEP 1/2 — Self-check: leave-one-out CV vs. Train_Labels.csv")
    loo = diagnostics.self_check_loo(proxy_df)
    print(loo.sort_values("rel_err_pct", ascending=False).head(10).to_string(index=False))
    loo_mape_pct = float(loo["rel_err_pct"].mean())
    print(f"\n  Mean LOO score (max(0, 1-MAPE)): {loo.attrs['mean_score']:.4f}  (mean MAPE {loo_mape_pct:.2f}%)")

    print_header("STEP 2/2 — Fitting calibration constant C on all Train files")
    C = model_mod.fit_calibration_constant(proxy_df)
    train_range = diagnostics.build_train_proxy_range(proxy_df)
    print(f"  C = {C:.6e}   (m = {physics.M_EXPONENT})")
    print(f"  Train proxy range: [{train_range[0]:.3e}, {train_range[1]:.3e}]")

    model_mod.save_calibration(artifacts_path, C, physics.M_EXPONENT, train_range, loo_mape_pct=loo_mape_pct)
    print(f"  Saved calibration to {artifacts_path}")
    return {
        "C": C, "m": physics.M_EXPONENT,
        "train_proxy_min": train_range[0], "train_proxy_max": train_range[1],
        "loo_mape_pct": loo_mape_pct,
    }


def resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return [input_path]


def main() -> None:
    args = parse_args()
    artifacts_path = (args.artifacts_path or DEFAULT_ARTIFACTS_PATH).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None

    if args.retrain:
        if data_dir is None:
            sys.exit("[ERROR] --retrain requires --data-dir.")
        calib = refit_calibration(data_dir, artifacts_path)
    elif artifacts_path.exists():
        print_header(f"Loading pre-fitted calibration from {artifacts_path}")
        print("  (pass --retrain --data-dir <dir> to refit instead)")
        calib = model_mod.load_calibration(artifacts_path)
        print(f"  C = {calib['C']:.6e}   (m = {calib['m']})")
    elif data_dir is not None:
        print_header(f"No saved calibration at {artifacts_path} — fitting from --data-dir")
        calib = refit_calibration(data_dir, artifacts_path)
    else:
        sys.exit(
            f"[ERROR] No saved calibration found at {artifacts_path}, and no --data-dir given to "
            f"fit from. Either point --artifacts-path at an existing calibration.json, or pass "
            f"--data-dir (add --retrain to force refitting even if a calibration already exists)."
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

    train_range = (calib["train_proxy_min"], calib["train_proxy_max"])
    results = []
    series_by_file = {}
    for f in files:
        x = physics.load_stress_series(f)
        series_by_file[f.name] = x
        proxy = physics.miner_proxy(x, calib["m"])
        pred = proxy / calib["C"]
        flag = diagnostics.flag_extrapolation(proxy, train_range)
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

    if args.diagnostics_output:
        import json
        loo_mape_pct = calib.get("loo_mape_pct")
        diag = []
        for _, row in pred_df.iterrows():
            damage = float(row["prediction"])
            diag.append({
                "file_id": row["file_id"],
                "damage": damage,
                "remaining_life_pct": max(0.0, (1.0 - damage) * 100.0),
                "error_band_pct": loo_mape_pct,
                "extrapolation_flagged": bool(row["flagged"]),
                "extrapolation_reason": row["reason"],
                "damage_progress": physics.damage_progress(series_by_file[row["file_id"]], calib["C"], calib["m"]),
            })
        args.diagnostics_output.write_text(json.dumps(diag, indent=2))
        print(f"  Wrote per-file diagnostics to {args.diagnostics_output}")


if __name__ == "__main__":
    main()
