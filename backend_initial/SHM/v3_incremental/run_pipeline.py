"""
SHM Subsystem (v3, incremental) — Single-Command Pipeline
=============================================================
Same one-command contract as v2 (`v2_rule-based/run_pipeline.py`):

    python run_pipeline.py --input /path/to/Test

loads the shipped calibration and predicts immediately, no training data
needed. What's new in v3, on top of v2's damage prediction:

1. Each file is fed through the streaming rainflow engine
   (`physics.RainflowState`) in chunks (`--chunk-size`, default 4096
   rows) rather than handed to a single whole-array call. This is what
   "online" means concretely here: at no point does the pipeline need
   the full file in memory at once to keep the running Miner's-rule
   proxy up to date -- see v3_incremental/physics.py and algorithm.md for
   why this reproduces v2's numbers exactly rather than approximating them.

2. Alongside the damage prediction, each file also gets a cycles-to-
   failure estimate and a confidence interval on it (see model.py for
   the method and its citations, algorithm.md for the caveats). These
   are written to a separate "extended" CSV so the primary submission
   file keeps the exact file_id,prediction schema v2 already validated
   against 04_Example_Submission.

To refit (e.g. after changing physics.py or model.py):

    python run_pipeline.py --retrain --data-dir /path/to/SHM \\
        --input /path/to/SHM/Test --output shm_predictions.csv
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
DEFAULT_CHUNK_SIZE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHM subsystem v3 — incremental Miner's-rule damage prediction "
                    "plus cycles-to-failure with a confidence interval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None,
        help="File or directory to predict on. Required unless --data-dir sets a default via Test/.")
    parser.add_argument("--output", type=Path, default=Path("shm_predictions.csv"),
        help="Where to write the submission CSV, file_id+prediction only (default: ./shm_predictions.csv).")
    parser.add_argument("--output-extended", type=Path, default=None,
        help="Where to write the extended CSV (damage + cycles-to-failure + CI). "
             "Default: <output stem>_extended.csv next to --output.")
    parser.add_argument("--data-dir", type=Path, default=None,
        help="Directory containing Train/ and Train_Labels.csv. Only needed with --retrain "
             "(or if --input is omitted, to default it to <data-dir>/Test).")
    parser.add_argument("--artifacts-path", type=Path, default=None,
        help=f"Where the calibration JSON is loaded from / saved to (default: {DEFAULT_ARTIFACTS_PATH}).")
    parser.add_argument("--retrain", action="store_true",
        help="Refit C and sigma_log from --data-dir instead of loading the shipped calibration, "
             "and overwrite --artifacts-path with the result. Requires --data-dir.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Simulated arrival chunk size for the streaming rainflow engine (default: {DEFAULT_CHUNK_SIZE}). "
             "Result is identical for any chunk size -- this only controls how 'online' the demo looks.")
    parser.add_argument("--confidence", type=float, default=model_mod.DEFAULT_CONFIDENCE,
        help=f"Two-sided confidence level for the cycles-to-failure interval (default: {model_mod.DEFAULT_CONFIDENCE}).")
    parser.add_argument("--d-fail", type=float, default=model_mod.DEFAULT_D_FAIL,
        help=f"Miner's-rule damage sum treated as failure (default: {model_mod.DEFAULT_D_FAIL}; "
             "real materials scatter roughly 0.7-2.2, see model.py docstring).")
    return parser.parse_args()


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def refit_calibration(data_dir: Path, artifacts_path: Path) -> dict:
    if not (data_dir / "Train_Labels.csv").exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")

    proxy_df = physics.compute_train_proxies(data_dir)

    print_header("STEP 1/3 — Self-check: leave-one-out CV vs. Train_Labels.csv")
    loo = diagnostics.self_check_loo(proxy_df)
    print(loo.sort_values("rel_err_pct", ascending=False).head(10).to_string(index=False))
    print(f"\n  Mean LOO score (max(0, 1-MAPE)): {loo.attrs['mean_score']:.4f}")

    print_header("STEP 2/3 — Fitting calibration constant C on all Train files")
    C = model_mod.fit_calibration_constant(proxy_df)
    train_range = diagnostics.build_train_proxy_range(proxy_df)
    print(f"  C = {C:.6e}   (m = {physics.M_EXPONENT})")
    print(f"  Train proxy range: [{train_range[0]:.3e}, {train_range[1]:.3e}]")

    print_header("STEP 3/3 — Fitting log-scatter sigma for the cycles-to-failure interval")
    sigma_log = diagnostics.fit_log_residual_sigma(loo)
    print(f"  sigma_log = {sigma_log:.4f}  (from LOO residuals, see diagnostics.fit_log_residual_sigma)")

    model_mod.save_calibration(artifacts_path, C, physics.M_EXPONENT, train_range, sigma_log)
    print(f"  Saved calibration to {artifacts_path}")
    return {
        "C": C, "m": physics.M_EXPONENT,
        "train_proxy_min": train_range[0], "train_proxy_max": train_range[1],
        "sigma_log": sigma_log,
    }


def resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return [input_path]


def predict_one(fpath: Path, calib: dict, chunk_size: int, confidence: float, d_fail: float) -> dict:
    """Feed one file through the streaming engine in chunks (the 'online'
    part) and compute damage, cycles-to-failure, and its interval."""
    x = physics.load_stress_series(fpath)
    state = physics.RainflowState(m=calib["m"])
    for start in range(0, len(x), chunk_size):
        state.feed(x[start:start + chunk_size])
    state.close()

    damage = model_mod.predict_damage(state.proxy, calib["C"])
    n_to_fail = model_mod.cycles_to_failure(state.n_cycles, damage, d_fail)
    n_remaining = model_mod.cycles_remaining(state.n_cycles, damage, d_fail)
    life_lo, life_hi = model_mod.cycles_to_failure_interval(
        state.n_cycles, damage, calib["sigma_log"], d_fail, confidence,
    )
    flag = diagnostics.flag_extrapolation(state.proxy, (calib["train_proxy_min"], calib["train_proxy_max"]))

    return {
        "file_id": fpath.name,
        "prediction": damage,
        "n_cycles_observed": state.n_cycles,
        "cycles_to_failure": n_to_fail,
        "cycles_remaining": n_remaining,
        f"cycles_to_failure_lo_{int(confidence * 100)}pct": life_lo,
        f"cycles_to_failure_hi_{int(confidence * 100)}pct": life_hi,
        **flag,
    }


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
        print(f"  C = {calib['C']:.6e}   (m = {calib['m']})   sigma_log = {calib['sigma_log']:.4f}")
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

    print_header(f"Predicting on {input_path}  (streamed in chunks of {args.chunk_size} rows)")
    files = resolve_input_files(input_path)
    if not files:
        sys.exit(f"[ERROR] No .csv files found under {input_path}")

    results = [predict_one(f, calib, args.chunk_size, args.confidence, args.d_fail) for f in files]
    pred_df = pd.DataFrame(results)

    pct = int(args.confidence * 100)
    display_cols = [
        "file_id", "prediction", "cycles_to_failure",
        f"cycles_to_failure_lo_{pct}pct", f"cycles_to_failure_hi_{pct}pct",
    ]
    print(pred_df[display_cols].to_string(index=False))

    flagged = pred_df[pred_df["flagged"]]
    if flagged.empty:
        print("\n  No prediction's proxy falls outside Train's demonstrated range.")
    else:
        print(f"\n  [WARN] {len(flagged)}/{len(pred_df)} file(s) flagged:")
        for _, row in flagged.iterrows():
            print(f"    {row['file_id']}: {row['reason']}")

    out_df = pred_df[["file_id", "prediction"]]
    out_df.to_csv(args.output, index=False)

    extended_path = args.output_extended or args.output.with_name(f"{args.output.stem}_extended.csv")
    extended_cols = [c for c in pred_df.columns if c != "reason"] + ["reason"]
    pred_df[extended_cols].to_csv(extended_path, index=False)

    print_header("DONE")
    print(f"  Wrote {len(out_df)} predictions to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/shm_predictions.csv)")
    print(f"  Wrote extended output (cycles-to-failure + {pct}% CI) to {extended_path}")


if __name__ == "__main__":
    main()
