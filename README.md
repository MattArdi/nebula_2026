# Nebula 2026 — PS3 Condition Monitoring

A condition-monitoring system for four independent subsystems on a rail
vehicle fleet: **ACV** (air-conditioning refrigerant-leak localization),
**Door** (door motor cycle segmentation and resistance classification),
**Rail Corrugation** (rail-side fault classification from axle-box
vibration), and **SHM** (structural health monitoring — cumulative
fatigue-damage estimation). Each subsystem is a self-contained,
versioned prediction pipeline; a FastAPI backend wraps all four for a
React dashboard, and a root CLI script (`predict.py`) runs all four
in batch over a zip of input files for competition submission.

This repo holds three things: the raw competition datasets
(`Datasets/`), everything needed to run the app — the prediction
pipelines, the FastAPI orchestration layer, and the web dashboard,
all under `app/` — and committed submission snapshots
(`predictions/`, `predictions.zip`), plus container deployment config
for running the app as one Cloud Run service.

## Repository layout

```
nebula_2026/
├── Datasets/              Raw competition data (Train/Test) per subsystem
├── app/                   Everything needed to run the app
│   ├── backend/             Prediction pipelines + FastAPI orchestration
│   │   ├── ACV/
│   │   ├── Door/
│   │   ├── Rail Corrugation/
│   │   ├── SHM/
│   │   └── api/                FastAPI service wrapping all four
│   └── dashboard/            React + Vite frontend
├── predictions/            Committed submission snapshots (v2, v3, v4)
├── predictions.zip         Zipped copy of predictions/v4
├── predict.py              CLI: run all four pipelines over a zip of inputs
├── Dockerfile              Two-stage build: dashboard → static/, then API
├── cloudbuild.yaml         Google Cloud Build config for the same image
└── .dockerignore / .gitignore
```

## `Datasets/`

One folder per subsystem (`ACV/`, `Door/`, `Rail_Corrugation/`, `SHM/`),
each with `Train/` (raw sensor files), `Train_Labels.csv` (or
equivalent — see each subsystem below), and `Test/` (unlabelled files
to predict on, matching the competition's held-out set). Rail
Corrugation is the largest by file count (341 files: 272 labelled
training + 68 test, each a 10,000-row × 129-column sensor CSV); SHM has
64 training + 16 test files, each a ~580K-row single-column stress
series; ACV has 6 labelled training cases plus test `.xlsx` files; Door
is a single continuous multi-day sensor stream (`Train.csv` +
`Train_Segments_Answer.csv`) rather than one-file-per-case.

## `app/` — everything needed to run the app

`app/backend/` and `app/dashboard/` together are the app: the
Dockerfile builds a single image from just this folder (see
Deployment, below) — `Datasets/`, `predictions/`, and `predict.py` are
repo-level, not part of the served app, and aren't copied into the
image.

### `app/backend/` — prediction pipelines

#### The versioned-pipeline convention

Every subsystem folder (`app/backend/ACV/`, `app/backend/Door/`,
`app/backend/Rail Corrugation/`, `app/backend/SHM/`) contains one
subdirectory per model version (`v1_testing`, `v2_rule-based`, `v3_...`,
etc.). `v1_testing/` is always exploratory scratch work (feature/method
scripts, not a runnable pipeline). Every version from `v2` onward
follows the same shape:

- `run_pipeline.py` — single-command CLI: `--input`, `--output`,
  `--data-dir`, `--retrain`, `--diagnostics-output`. By default it
  loads the version's shipped, pre-fitted weights/calibration and
  predicts immediately — no training data needed. Passing
  `--retrain --data-dir <path>` refits from scratch and overwrites the
  shipped artifacts.
- A model/rules module (`model.py`, `rules.py`, `ranking.py`, or
  `physics.py` depending on the subsystem) implementing the actual
  prediction logic.
- `diagnostics.py` — the honest validation protocol for that task
  (leave-one-out CV for the small-sample subsystems, repeated
  stratified k-fold for Rail's larger set), scored with the real
  competition metric, not a proxy.
- `weights/` or `artifacts/` — the version's fitted parameters,
  committed to the repo (model files, calibration constants,
  thresholds), loaded by default so no pipeline ever retrains on a
  normal prediction call.
- `algorithm.md` — full write-up: input/output format, how the
  algorithm works section by section, what was tried and discarded,
  and the real train/test evidence behind every design choice. This is
  the primary source of truth for each pipeline's rationale — read it
  before modifying a pipeline.

Only one version per subsystem is "shipped" (wired into
`app/backend/api/main.py` and `predict.py`'s `SUBSYSTEMS` configs) at a
time; older versions stay in the repo for comparison and are still
independently runnable.

#### Subsystem-by-subsystem

- **`app/backend/ACV/`** — `v2_rule-based` (shipped): a
  zero-fitted-parameter ranking algorithm (cross-car median-differencing
  → locally-standardized → decayed CUSUM) that orders a train's 8 cars
  from most- to least-likely to have the refrigerant leak. No ML — every
  constant comes from SPC theory, since only 6 labelled cases exist.
- **`app/backend/Door/`** — `v3_adaptive` (shipped): timestamp-gap
  segmentation to find cycle boundaries in a continuous stream, then a
  recalibratable-baseline current threshold per operation (Open/Close)
  to classify each cycle Normal or Abnormal resistance, with a bootstrap
  confidence half-width on the threshold itself. `v2_rule-based` is the
  non-adaptive predecessor (byte-identical predictions on the real Test
  set); `v3_adaptive` adds the recalibration and confidence machinery.
- **`app/backend/Rail Corrugation/`** — `v4_class_weighted` (shipped): a
  3-model soft-voting ensemble (CatBoost + XGBoost + Logistic
  Regression) on 49 engineered time- and frequency-domain features,
  combined via a per-class (not per-model) weight matrix. `v2_ensemble`
  is the original 41-feature, scalar-weighted ensemble; `v3_frequency`
  added the 8 FFT features; `v4_class_weighted` changed only how the
  three models' votes combine.
- **`app/backend/SHM/`** — `v3_ensemble_blend` (shipped): a weighted
  blend of a rainflow/Miner's-rule physics formula (dominant, weight
  7/8) and an Extra Trees regressor on 47 engineered features (weight
  1/8). `v2_rule-based` is the pure-physics predecessor — one fitted
  parameter, no ML — which the blend's own `algorithm.md` documents as
  outperforming every individual ML model tried.

#### `app/backend/api/`

`main.py` — a FastAPI service (`uvicorn backend.api.main:app`) backing
the dashboard. It does not import the pipelines directly; each
subsystem's modules share names across subsystems (`model.py`,
`diagnostics.py`, ...), so it shells out to each version's
`run_pipeline.py` as a subprocess (run in a worker thread, so
concurrent requests to different subsystems run in parallel instead of
queueing behind each other on FastAPI's single event loop). Its
`SUBSYSTEMS` dict names which version of each pipeline is live, kept in
sync with `predict.py`'s config at the repo root. `requirements.txt`
pins the Python dependencies for the whole backend (FastAPI, pandas,
scikit-learn, xgboost, catboost, rainflow, etc.).

### `app/dashboard/`

A React + Vite single-page app (`npm run dev` for local development,
`npm run build` for a production build). `src/subsystems/` holds one
page per subsystem plus a fleet-wide ranking view; `src/components/`
holds shared chart and layout components; `public/sample-data/` bundles
example input files so each subsystem page can demo a prediction
without the user uploading anything. Talks to the FastAPI backend over
`/api/*`; in production the backend serves the built dashboard directly
(see `Dockerfile`), so API calls are same-origin.

## `predictions/` and `predictions.zip`

`predictions/` holds committed submission snapshots, one folder per
round (`v2`, `v3`, `v4`), each with `acv_predictions.csv`,
`door_predictions.csv`, `rail_predictions.csv`, `shm_predictions.csv`
in the exact competition submission schema. These are point-in-time
outputs of whichever pipeline versions were shipped when that snapshot
was generated — not regenerated automatically, and not guaranteed to
match the *currently* shipped versions in `app/backend/`.
`predictions.zip` is a zipped copy of `predictions/v4`, kept at the
repo root for easy download. Regenerate a fresh submission at any time
with `predict.py` (below) rather than relying on an older snapshot.

## Running it

**One-off batch submission** (all four subsystems, matching the
competition's expected zip layout):

```
python predict.py held_out_test.zip --data-root /path/to/02_Datasets --output-dir ./predictions
```

**Local API + dashboard**, two processes:

```
uvicorn app.backend.api.main:app --reload --port 8000    # backend
cd app/dashboard && npm run dev                            # frontend (separate terminal)
```

**A single subsystem pipeline directly**, e.g. Rail Corrugation:

```
cd "app/backend/Rail Corrugation/v4_class_weighted"
python run_pipeline.py --input /path/to/Test
```

**Deployment**: `Dockerfile` builds `app/dashboard/`, then copies it
and `app/backend/` into a single Python image that serves both the API
and the static frontend from one process (`cloudbuild.yaml` wires this
into Google Cloud Build for Cloud Run). Note: the Cloud Build trigger
described in `cloudbuild.yaml`'s setup comments fires on pushes to
`app_working`, not `main` — check which branch your trigger actually
watches before expecting a push here to auto-deploy.
