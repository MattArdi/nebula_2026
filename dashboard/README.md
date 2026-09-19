# Train Condition Monitoring — PS3

A single web app covering all four PS3 subsystems (Door, ACV, Rail Corrugation, SHM).
Every subsystem's dashboard **opens already populated** with real, bundled PS3
held-out test data — stat cards + a results table, no upload required just to look
around — and dropping in your own file(s) re-runs the same pipeline and replaces
what's shown. Everything runs client-side in the browser — no backend, no upload to
a server.

## Quick start

```bash
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5174`). Each subsystem in the
sidebar loads its own dashboard automatically. Drop in file(s) to swap in different
data (e.g. the official held-out test set at submission time) and get an updated
prediction table + download button.

## Bundled sample data

`public/sample-data/` holds a representative subset of the real PS3 `Test/` files,
copied from `PS3/02_Datasets/`, auto-fetched on each page's first load (see
`src/lib/sampleManifest.js` for exactly which files):

- **Door**: the full `Test.csv` stream.
- **ACV**: the full `acv_test_case.xlsx`.
- **Rail Corrugation**: 12 files (4 predicted Normal, 4 Side I, 4 Side II) out of the
  official 68, chosen for a demo mix — the other 56 are correctly predicted Normal
  too, just not bundled to keep the repo size reasonable (68 files is ~1.1GB).
- **SHM**: all 16 official test files.

This is a demo convenience, not a substitute for the real submission — for
`predictions.zip`, drop the **full, official** Test folder for each subsystem in to
regenerate complete predictions before downloading.

## The four models

Every model was fit and/or validated against the real Train data in
`PS3/02_Datasets/`, using each subsystem's **actual competition metric** (not a
generic proxy) — see each subsystem's Info Kit for the metric definitions.

| Subsystem | Approach | Validated score |
|---|---|---|
| **Door** (`src/subsystems/door/doorModel.js`) | Segmentation by timestamp gaps (>100ms — the file has zero rows during the idle gap between cycles, confirmed against Train.csv), then a 4-feature logistic regression (motor current mean/std/max, back-EMF mean) | 5-fold CV F1 = **0.99** |
| **ACV** (`src/subsystems/acv/acvModel.js`) | Unsupervised cross-car anomaly scoring — for every numeric per-car telemetry parameter, each car's z-score vs. the other 7 at each timestamp, averaged per-parameter then across parameters, ranked descending. No fitted parameters, so it can't overfit Train, and generalizes across the wildly different 8-param vs. 60+-param file schemas without any hardcoded column names. | Average rank-decay score = **0.979** (5/6 Train cases ranked the true faulty car 1st, 1/6 ranked it 2nd) |
| **Rail Corrugation** (`src/subsystems/rail/railModel.js`) | 25 hand-engineered time-domain vibration/shock features (RMS, kurtosis, crest factor, zero-crossing rate, p95, computed separately for Side I vs Side II axle-box channels), fed into a standardized multinomial logistic regression | 5-fold CV macro F1 = **0.770** (vs. 0.33 for a trivial "always Normal" baseline) |
| **SHM** (`src/subsystems/shm/shmModel.js`) | Real ASTM E1049 rainflow cycle counting + Miner's linear damage rule (the literal domain-standard method the reference damage values were computed with) — S-N exponent m=5 (grid-searched), single calibration constant fit as the median ratio of true/proxy damage across the 64 Train files | Leave-one-out CV `max(0, 1-MAPE)` = **0.974** |

Door and Rail Corrugation's fitted parameters, and ACV/SHM's derivation, are
documented inline at the top of each model file — including the honest
cross-validation protocol used and any baseline approaches that were tried and
scored worse.

## Architecture

```
src/
  App.jsx                     subsystem picker + page routing
  components/
    Layout.jsx                 sidebar / mobile nav / top bar
    ui.jsx                     FileDrop, LabelBadge, ProgressBar, etc.
    BatchSubsystemPage.jsx      shared "drop many files -> one row per file" page
                                 (used by Rail Corrugation and SHM, which share that shape)
  lib/
    csvExport.js                builds + triggers the prediction CSV download
    parseNumericCsv.js           headerless numeric-CSV parsing (Rail Corrugation, SHM)
  subsystems/
    door/     DoorPage.jsx + doorModel.js     (single continuous-stream file)
    acv/      AcvPage.jsx + acvModel.js       (single .xlsx file)
    rail/     RailPage.jsx + railModel.js     (many files, one row each)
    shm/      ShmPage.jsx + shmModel.js       (many files, one row each)
```

Door and ACV have bespoke pages (their output isn't "one row per file" — Door has
no `file_id` at all, ACV has a `ranked_cars` list). Rail Corrugation and SHM share
`BatchSubsystemPage` since both score one prediction row per uploaded file.

## Producing `predictions.zip`

1. Go to each subsystem you're submitting, drop in the **held-out test files**
   (`02_Datasets/<Subsystem>/Test/`), and click download once processing finishes.
2. Zip the resulting `*_predictions.csv` files together — flat, no subfolders,
   named `predictions.zip` — per the top-level PS3 spec's submission structure.

## A note on the ACV/Rail Corrugation validation

Both models were also each re-validated by an independent second pass (fresh code,
not reusing the first attempt's intermediate work) specifically to catch mistakes
the first pass might have carried through unnoticed — the ACV port initially had a
real bug (a "None" text value used for a missing sensor reading was wrongly
disqualifying the entire telemetry parameter, and a weighting mismatch between
JS and the validated Python reference was silently changing rankings) caught this
way and fixed; see the comments in `acvModel.js` for what happened and why the
final approach was chosen over the alternatives that were tried.
