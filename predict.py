#!/usr/bin/env python3
"""
Root pipeline — run all four subsystems' predictions from one command.

    python predict.py <input.zip>

<input.zip> must contain up to four top-level folders (any subset is
fine — attempt whichever subsystems you like), named ACV, Door,
Rail_Corrugation, and SHM, each holding the data to predict on for that
subsystem — exactly what each subsystem's own v2_rule-based/run_pipeline.py
expects as --input:

    Door/              one continuous sensor .csv stream (like Test.csv)
    ACV/               one .xlsx case file (like acv_test_case.xlsx)
    Rail_Corrugation/  one or more .csv files, one per recording
    SHM/               one or more .csv files, one per recording

A subsystem folder you omit from the zip is simply skipped.

Each subsystem pipeline also needs its own labelled TRAINING data
(Train.csv/Train_Labels.csv/etc, in the same layout as the competition's
02_Datasets/<Subsystem>/) to fit itself and self-check. That is NOT
inside the zip — the zip is prediction input only — so it must already
exist on this machine. Point --data-root at a folder containing
ACV/, Door/, Rail_Corrugation/, SHM/ subfolders in that layout, or set
the NEBULA_DATA_ROOT environment variable. Without either, this script
tries a few conventional relative locations next to the repo and fails
loudly if none exist — it will not silently guess wrong.

Output: writes door_predictions.csv, acv_predictions.csv,
rail_predictions.csv, shm_predictions.csv into --output-dir (default:
current directory), one per subsystem folder actually found in the zip.

    python predict.py held_out_test.zip --data-root /path/to/02_Datasets
    python predict.py held_out_test.zip --output-dir ./predictions
"""

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# subsystem key -> (folder under app/backend/, which pre-built version of that
# subsystem's code to run, output filename, input kind). "version" names a
# subfolder inside app/backend/<pipeline_dir>/ that contains a run_pipeline.py
# following the --data-dir/--input/--output convention (every subsystem's
# v2_rule-based/ does) -- swap it here to point at a different pre-built
# pipeline without touching any orchestration logic below.
SUBSYSTEMS = {
    # Door runs v3_adaptive, not v2_rule-based: verified byte-identical to
    # v2 on real Test.csv (see app/backend/Door/v3_adaptive/algorithm.md
    # Section 3.2), so this changes nothing about the submitted labels
    # while additionally computing the low_confidence/out_of_range signals
    # v2 doesn't have. Kept in sync with app/backend/api/main.py's SUBSYSTEMS.
    "Door":             {"pipeline_dir": "Door",            "version": "v3_adaptive",   "output": "door_predictions.csv", "input_kind": "single_csv"},
    "ACV":              {"pipeline_dir": "ACV",              "version": "v2_rule-based", "output": "acv_predictions.csv",  "input_kind": "single_xlsx"},
    # Rail runs v4_class_weighted, not v2_ensemble: a real, validated
    # improvement (macro F1 0.8809 -> 0.8940 in repeated CV, confirmed on a
    # real held-out score 0.8552 -> 0.8877 -- see app/backend/Rail Corrugation/
    # v4_class_weighted/algorithm.md Section 5), not a byte-identical swap
    # like Door's above -- some Test predictions genuinely differ.
    "Rail_Corrugation": {"pipeline_dir": "Rail Corrugation", "version": "v4_class_weighted", "output": "rail_predictions.csv", "input_kind": "dir"},
    # SHM runs v3_ensemble_blend, not the pure-physics v2_rule-based: a
    # small, real held-out improvement (0.9652 -> 0.9668), though the
    # underlying LOO gain that motivated it (+0.0006) is noise-level and
    # unconfirmed by an independent check -- see app/backend/SHM/
    # v3_ensemble_blend/algorithm.md Section 5 for the full, honest account
    # of why this shipped anyway.
    "SHM":              {"pipeline_dir": "SHM",              "version": "v3_ensemble_blend", "output": "shm_predictions.csv",  "input_kind": "dir"},
}

# Accepted spellings for each subsystem's folder, inside the zip and under --data-root.
NAME_ALIASES = {
    "Door": ["door"],
    "ACV": ["acv"],
    "Rail_Corrugation": ["rail_corrugation", "rail corrugation"],
    "SHM": ["shm"],
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _is_junk(path: Path) -> bool:
    """Zip-metadata cruft (macOS resource forks, .DS_Store) that should never match real data."""
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


def find_subsystem_dir(root: Path | None, subsystem: str) -> Path | None:
    """Find a directory under `root` matching one of `subsystem`'s aliases (any depth, shallowest wins)."""
    if root is None:
        return None
    aliases = {_normalize(a) for a in NAME_ALIASES[subsystem]}
    candidates = [
        p for p in root.rglob("*")
        if p.is_dir() and _normalize(p.name) in aliases and not _is_junk(p)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.relative_to(root).parts))
    return candidates[0]


def _descend_if_single_child(d: Path, glob_pattern: str, max_depth: int = 3) -> Path:
    """If `d` has no matching files directly but wraps a single subfolder, descend into it."""
    for _ in range(max_depth):
        if any(not _is_junk(f) for f in d.glob(glob_pattern)):
            return d
        subdirs = [c for c in d.iterdir() if c.is_dir() and not _is_junk(c)]
        if len(subdirs) != 1:
            break
        d = subdirs[0]
    return d


def resolve_data_root(explicit: Path | None) -> Path | None:
    """
    Best-effort only -- returns None (not a hard error) if nothing is found.
    Not every subsystem needs training data to run (Rail Corrugation's
    v2_ensemble ships pre-trained weights and works with no data root at
    all); whether its absence is actually fatal is decided per-subsystem
    in run_subsystem(), once each pipeline script has had a chance to say
    whether it can proceed without it.
    """
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get("NEBULA_DATA_ROOT")
    if env:
        return Path(env).resolve()
    for candidate in [
        REPO_ROOT / "02_Datasets",
        REPO_ROOT / "PS3" / "02_Datasets",
        REPO_ROOT.parent / "02_Datasets",
        REPO_ROOT.parent / "PS3" / "02_Datasets",
    ]:
        if candidate.exists():
            return candidate
    return None


def resolve_input(subsystem_dir: Path, kind: str, subsystem: str) -> Path:
    if kind == "dir":
        return _descend_if_single_child(subsystem_dir, "*.csv")

    pattern = "*.csv" if kind == "single_csv" else "*.xlsx"
    matches = sorted(p for p in subsystem_dir.rglob(pattern) if not _is_junk(p))
    if len(matches) == 0:
        sys.exit(f"[ERROR] {subsystem}: no {pattern} file found under {subsystem_dir}")
    if len(matches) > 1:
        sys.exit(
            f"[ERROR] {subsystem}: expected exactly one {pattern} file under {subsystem_dir}, "
            f"found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]


def run_subsystem(subsystem: str, cfg: dict, zip_root: Path, data_root: Path, output_dir: Path):
    """Returns (status, message). status is True/False/None (None = skipped)."""
    subsystem_input_dir = find_subsystem_dir(zip_root, subsystem)
    if subsystem_input_dir is None:
        return None, "skipped -- no matching folder found in the zip"

    # Training data is only strictly required by subsystems that don't ship
    # pre-trained weights (Rail Corrugation's v2_ensemble does -- see its
    # run_pipeline.py). So a missing --data-root match is NOT fatal here:
    # just omit --data-dir and let that subsystem's own run_pipeline.py
    # decide whether it can proceed (it errors clearly if it truly needs it).
    subsystem_data_dir = find_subsystem_dir(data_root, subsystem)

    try:
        input_path = resolve_input(subsystem_input_dir, cfg["input_kind"], subsystem)
    except SystemExit as e:
        return False, str(e)

    pipeline_script = REPO_ROOT / "app" / "backend" / cfg["pipeline_dir"] / cfg["version"] / "run_pipeline.py"
    if not pipeline_script.exists():
        return False, f"FAILED -- pipeline script not found: {pipeline_script}"

    output_path = (output_dir / cfg["output"]).resolve()

    cmd = [sys.executable, str(pipeline_script)]
    if subsystem_data_dir is not None:
        cmd += ["--data-dir", str(subsystem_data_dir.resolve())]
    cmd += ["--input", str(input_path.resolve()), "--output", str(output_path)]

    print(f"\n{'=' * 70}\nRunning {subsystem}\n{'=' * 70}")
    print(f"  data-dir : {subsystem_data_dir if subsystem_data_dir else '(none found -- relying on pre-trained weights, if any)'}")
    print(f"  input    : {input_path}")
    print(f"  output   : {output_path}")

    result = subprocess.run(cmd, cwd=str(pipeline_script.parent))
    if result.returncode != 0:
        return False, f"FAILED -- run_pipeline.py exited with code {result.returncode}"
    return True, f"OK -- wrote {output_path}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("zipfile", type=Path, help="Zip file containing subsystem prediction-input folders.")
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Folder containing each subsystem's Train data (layout of 02_Datasets/). "
             "Falls back to $NEBULA_DATA_ROOT, then a few conventional relative paths.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Where to write the *_predictions.csv files (default: current directory).",
    )
    args = parser.parse_args()

    if not args.zipfile.exists():
        sys.exit(f"[ERROR] Zip file not found: {args.zipfile}")
    if not zipfile.is_zipfile(args.zipfile):
        sys.exit(f"[ERROR] Not a valid zip file: {args.zipfile}")

    data_root = resolve_data_root(args.data_root)
    print(f"[INFO] Training-data root : {data_root if data_root else '(none found -- subsystems without pre-trained weights will fail)'}")
    print(f"[INFO] Output directory   : {args.output_dir.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nebula_predict_") as tmp:
        tmp_path = Path(tmp)
        print(f"[INFO] Extracting {args.zipfile} ...")
        with zipfile.ZipFile(args.zipfile) as zf:
            zf.extractall(tmp_path)

        results = {}
        for subsystem, cfg in SUBSYSTEMS.items():
            results[subsystem] = run_subsystem(subsystem, cfg, tmp_path, data_root, args.output_dir)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    any_failed = False
    for subsystem, (ok, msg) in results.items():
        tag = "SKIP" if ok is None else ("OK" if ok else "FAIL")
        any_failed = any_failed or (ok is False)
        print(f"  [{tag:4}] {subsystem:18} {msg}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
