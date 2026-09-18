"""
SHM Subsystem (rule-based) — Single-Command Pipeline
=========================================================
Run this one file to go from raw stress CSVs to a submission-ready
predictions file, with diagnostics printed along the way:

    python run_pipeline.py --data-dir /path/to/SHM

That path must contain Train/, Train_Labels.csv, and (by default) Test/.
Everything else is optional:

    python run_pipeline.py --data-dir /path/to/SHM \\
        --input /path/to/SHM/Test \\
        --output shm_predictions.csv

--input accepts either a directory (every *.csv inside it is predicted,
one row each -- the normal case, since SHM's Test/ holds 16 separate
files) or a single CSV file.

What it does, in order
-----------------------
1. Runs the full pipeline (physics.py + model.py) through leave-one-out
   cross-validation against Train_Labels.csv, scored with the real
   competition metric (max(0, 1-MAPE)). This is the one trustworthy
   estimate of how the pipeline performs on unseen files.
2. Fits the final calibration constant C on all 64 training files.
3. Predicts damage for every file under --input.
4. Flags any prediction whose Miner's-rule proxy falls outside the range
   Train ever demonstrated -- an extrapolation, not an interpolation.
5. Writes shm_predictions.csv in the exact submission schema: file_id,
   prediction.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics
import model as model_mod
import physics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHM subsystem — rule-based Miner's-rule damage prediction, one-command run."
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
        "--output", type=Path, default=Path("shm_predictions.csv"),
        help="Where to write the submission CSV (default: ./shm_predictions.csv).",
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

    if not (data_dir / "Train_Labels.csv").exists():
        sys.exit(f"[ERROR] Train_Labels.csv not found in {data_dir}")
    if not input_path.exists():
        sys.exit(f"[ERROR] Input path not found: {input_path}")

    # -----------------------------------------------------------------
    # Compute rainflow proxies for all Train files ONCE, shared by both
    # the self-check and the final calibration fit below (rainflow
    # counting is the only real computational cost in this pipeline, so
    # this halves total runtime versus recomputing it twice).
    # -----------------------------------------------------------------
    proxy_df = physics.compute_train_proxies(data_dir)

    # -----------------------------------------------------------------
    # 1. Self-check: leave-one-out CV against Train_Labels.csv
    # -----------------------------------------------------------------
    print_header("STEP 1/4 — Self-check: leave-one-out CV vs. Train_Labels.csv")
    loo = diagnostics.self_check_loo(proxy_df)
    print(loo.sort_values("rel_err_pct", ascending=False).head(10).to_string(index=False))
    print(f"\n  Mean LOO score (max(0, 1-MAPE)): {loo.attrs['mean_score']:.4f}")

    # -----------------------------------------------------------------
    # 2. Fit final calibration constant on all training data
    # -----------------------------------------------------------------
    print_header("STEP 2/4 — Fitting calibration constant C on all Train files")
    C = model_mod.fit_calibration_constant(proxy_df)
    print(f"  C = {C:.6e}   (m = {physics.M_EXPONENT})")
    train_range = diagnostics.build_train_proxy_range(proxy_df)
    print(f"  Train proxy range: [{train_range[0]:.3e}, {train_range[1]:.3e}]")

    # -----------------------------------------------------------------
    # 3. Predict on the target input file(s)
    # -----------------------------------------------------------------
    print_header(f"STEP 3/4 — Predicting on {input_path}")
    files = resolve_input_files(input_path)
    results = []
    for f in files:
        x = physics.load_stress_series(f)
        proxy = physics.miner_proxy(x)
        pred = proxy / C
        flag = diagnostics.flag_extrapolation(proxy, train_range)
        results.append({"file_id": f.name, "prediction": pred, **flag})
    pred_df = pd.DataFrame(results)
    print(pred_df[["file_id", "prediction"]].to_string(index=False))

    # -----------------------------------------------------------------
    # 4. Extrapolation diagnostics
    # -----------------------------------------------------------------
    print_header("STEP 4/4 — Extrapolation check")
    flagged = pred_df[pred_df["flagged"]]
    if flagged.empty:
        print("  No prediction's proxy falls outside Train's demonstrated range.")
    else:
        print(f"  [WARN] {len(flagged)}/{len(pred_df)} file(s) flagged:")
        for _, row in flagged.iterrows():
            print(f"    {row['file_id']}: {row['reason']}")

    # -----------------------------------------------------------------
    # Write submission file
    # -----------------------------------------------------------------
    out_df = pred_df[["file_id", "prediction"]]
    out_df.to_csv(args.output, index=False)
    print_header("DONE")
    print(f"  Wrote {len(out_df)} predictions to {args.output}")
    print(f"  Columns: {list(out_df.columns)}  (matches 04_Example_Submission/shm_predictions.csv)")


if __name__ == "__main__":
    main()
