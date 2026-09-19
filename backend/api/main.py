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

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Mirrors predict.py's SUBSYSTEMS config exactly, so the dashboard and the
# CLI submission path (predict.py) always run the same pipeline version.
SUBSYSTEMS = {
    "door": {"pipeline_dir": "Door", "version": "v2_rule-based",
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
    allow_origins=["*"],  # local-only dev server, not deployed publicly
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_subsystem(key: str, files: list[UploadFile]) -> dict:
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

        cmd = [sys.executable, str(pipeline_script), "--input", str(input_path), "--output", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(pipeline_script.parent))

        if result.returncode != 0 or not output_path.exists():
            raise HTTPException(status_code=500, detail={
                "message": f"'{key}' pipeline failed (exit {result.returncode})",
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            })

        df = pd.read_csv(output_path)
        csv_text = output_path.read_text()

    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records"),
        "csv": csv_text,
        "filename": cfg["output"],
        "log_tail": result.stdout[-2000:],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/door/predict")
async def predict_door(file: UploadFile = File(...)):
    return _run_subsystem("door", [file])


@app.post("/api/acv/predict")
async def predict_acv(file: UploadFile = File(...)):
    return _run_subsystem("acv", [file])


@app.post("/api/rail/predict")
async def predict_rail(files: list[UploadFile] = File(...)):
    return _run_subsystem("rail", files)


@app.post("/api/shm/predict")
async def predict_shm(files: list[UploadFile] = File(...)):
    return _run_subsystem("shm", files)
