"""
Local prediction API for the PS3 dashboard.

Wraps each subsystem's already-validated run_pipeline.py
(backend/<Subsystem>/<version>/run_pipeline.py) via subprocess -- the same
approach root predict.py uses, and for the same reason: each subsystem's
modules (rules.py, diagnostics.py, model.py, ...) share names across
subsystems, so importing more than one into a single long-running Python
process would collide. Subprocess isolation sidesteps that and reuses the
exact, already-verified CLI behavior (including "predict with no --data-dir"
-- every pipeline here loads its shipped artifacts/weights, never touches
training data).

Run:
    uvicorn backend.api.main:app --reload --port 8000
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Only present in the deployed container (see Dockerfile: the frontend build
# output is copied here) -- absent in local dev, where the dashboard runs
# under `npm run dev` on its own port instead. Mounted, if present, after
# every /api/* route below, so it never shadows them.
STATIC_DIR = REPO_ROOT / "static"

# Mirrors predict.py's SUBSYSTEMS config, so the dashboard and the CLI
# submission path (predict.py) always run the same pipeline version --
# except Door, deliberately: v3_adaptive is byte-identical to v2_rule-based
# on real Test.csv (verified; see backend/Door/v3_adaptive/algorithm.md
# Section 3.2), so running it here costs nothing and unlocks the
# low_confidence/out_of_range diagnostics v2 doesn't compute. predict.py's
# own SUBSYSTEMS config was updated to match, for the same reason.
SUBSYSTEMS = {
    "door": {"pipeline_dir": "Door", "version": "v3_adaptive",
              "output": "door_predictions.csv", "input_kind": "single"},
    "acv": {"pipeline_dir": "ACV", "version": "v2_rule-based",
             "output": "acv_predictions.csv", "input_kind": "single"},
    "rail": {"pipeline_dir": "Rail Corrugation", "version": "v2_ensemble",
              "output": "rail_predictions.csv", "input_kind": "multi"},
    "shm": {"pipeline_dir": "SHM", "version": "v2_rule-based",
             "output": "shm_predictions.csv", "input_kind": "multi"},
}

app = FastAPI(title="PS3 Condition Monitoring API")
app.add_middleware(
    CORSMiddleware,
    # Wildcard is harmless here: the deployed frontend calls this API
    # same-origin (see STATIC_DIR below), so CORS only matters for someone
    # calling the API directly from a browser on another origin, which
    # carries no more exposure than the API already has with no auth of
    # its own -- the dashboard's login gate is client-side only.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _run_subsystem(key: str, files: list[UploadFile]) -> dict:
    cfg = SUBSYSTEMS[key]
    pipeline_script = REPO_ROOT / "backend" / cfg["pipeline_dir"] / cfg["version"] / "run_pipeline.py"
    if not pipeline_script.exists():
        raise HTTPException(500, f"Pipeline script not found: {pipeline_script}")

    if cfg["input_kind"] == "single" and len(files) != 1:
        raise HTTPException(400, f"'{key}' expects exactly one file, got {len(files)}")
    if not files:
        raise HTTPException(400, "No file(s) uploaded")

    with tempfile.TemporaryDirectory(prefix=f"ps3_{key}_") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for f in files:
            dest = input_dir / Path(f.filename).name
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)

        input_path = (input_dir / Path(files[0].filename).name) if cfg["input_kind"] == "single" else input_dir
        output_path = tmp_path / cfg["output"]
        diagnostics_path = tmp_path / "diagnostics.json"

        cmd = [
            sys.executable, str(pipeline_script),
            "--input", str(input_path), "--output", str(output_path),
            "--diagnostics-output", str(diagnostics_path),
        ]
        # subprocess.run blocks its calling thread for the pipeline's full
        # runtime (interpreter startup + heavy imports + inference). Run it
        # in a worker thread so concurrent requests -- e.g. all four
        # subsystem pages auto-loading their sample file at once on page
        # load -- actually run in parallel instead of queueing behind each
        # other on FastAPI's single asyncio event loop.
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, cwd=str(pipeline_script.parent)
        )

        if result.returncode != 0 or not output_path.exists():
            raise HTTPException(status_code=500, detail={
                "message": f"'{key}' pipeline failed (exit {result.returncode})",
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            })

        df = pd.read_csv(output_path)
        csv_text = output_path.read_text()
        # Diagnostic fields the pipeline already computes but that don't
        # belong in the submission CSV (see each subsystem's algorithm.md
        # Section 4/5, "Explainability") -- never affects `csv`/`rows`
        # above, which are exactly what predict.py would produce.
        diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else None

    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records"),
        "csv": csv_text,
        "filename": cfg["output"],
        "diagnostics": diagnostics,
        "log_tail": result.stdout[-2000:],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/door/predict")
async def predict_door(file: UploadFile = File(...)):
    return await _run_subsystem("door", [file])


@app.post("/api/acv/predict")
async def predict_acv(file: UploadFile = File(...)):
    return await _run_subsystem("acv", [file])


@app.post("/api/rail/predict")
async def predict_rail(files: list[UploadFile] = File(...)):
    return await _run_subsystem("rail", files)


@app.post("/api/shm/predict")
async def predict_shm(files: list[UploadFile] = File(...)):
    return await _run_subsystem("shm", files)


# Registered last: this app has no client-side routing (no react-router),
# so serving the built dashboard is just `index.html` + its assets -- no
# catch-all/SPA-fallback route needed. html=True serves index.html at "/".
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
