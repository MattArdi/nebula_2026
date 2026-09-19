# Rail Corrugation Subsystem — v3, Frequency-Augmented Feature Engineering + Soft-Voting Ensemble

Rail Corrugation is a **3-class classification** task: for each file, decide
whether the rail is `Normal`, or corrugated on `Side I` (odd bearing
positions 1, 3, 5, 7 across all 8 cars) or `Side II` (even bearing
positions 2, 4, 6, 8). One file goes in, one label comes out.

This version adds one thing to v2: 8 frequency-domain features (FFT energy
in the 0-100 Hz band) alongside v2's 41 time-domain features. Nothing else
changed — same ensemble architecture, same hyperparameters, same training
protocol — so the entire improvement (macro F1 0.8411 → 0.8808 in repeated
CV, confirmed on two independent sets of CV seeds) is attributable to that
one addition. Section 5 has the full evidence trail.

## 1. Required data input and format

### Input

A sensor CSV: one header row, 10,000 rows, 129 columns. Column 0 is
`Rotating speed` (a 0/1 pulse train, not a continuous speed reading).
Columns 1-128 are `Vibration of bearing in position P of car C` / `Shock
of bearing in position P of car C`, for `P` in 1..8 and `C` in 1..8,
interleaved (vib, shock, vib, shock, ...) so that all 16 readings for one
car appear together before the next car's. Per the info kit
(`Rail_Corrugation_Info_Kit.md` Section 2.1): sampling frequency 10,000 Hz,
1 second per file, units m/s².

### Labels (for retraining only, not needed to run predictions)

`filename,label` — one row per training file, label one of `Normal`,
`Side I`, `Side II`. The fitted ensemble is already shipped in `weights/`
(see Section 3.3), so cloning this repo and running `run_pipeline.py
--input <files>` needs none of this — no `Train/`, no `Train_Labels.csv`,
nothing beyond the files to predict on.

### Output

One row per file, identical schema to v2:

```
file_id,prediction
Test1.csv,Normal
Test13.csv,Side I
Test22.csv,Side I
```

## 2. How this addresses the info kit's 3 pain points

| # | Pain point | Status | Mechanism |
|---|---|---|---|
| 1 | *"The corrugation formation mechanism is influenced by many confounding factors (sleeper spacing, bogie natural frequencies, curve geometry, track elasticity), so simple threshold-based detection on raw vibration amplitude is unreliable — the characteristic signature must be separated from normal speed- and ballast-dependent vibration."* | **Addressed further than v2, still partial** | Not a raw-amplitude threshold — a 49-feature statistical ensemble. Speed confound: every RMS feature has a speed-normalized variant (Section 3.1). Bogie/resonance frequency: **now represented** — 8 new FFT energy features in the 0-100 Hz band, chosen empirically (Section 5) and independently corroborated by the info kit's own physical description of corrugation as producing "low-frequency rumbling noise" from "self-excited coupled vibration feedback... within specific frequency bands." This is a real, validated gain (+0.05 macro F1), not a token addition — see Section 5's full ablation. Sleeper spacing / curve geometry / track elasticity: **still not addressable from this data** — no positional or track metadata exists in the raw files, a dataset ceiling no feature engineering can cross. |
| 2 | *"Side I and Side II rails must be judged independently from the same recording..."* | **Unchanged from v2 — the localization half is solved; independent-per-side judgment is capped by the label schema, not the model** | Every feature (time-domain and the new frequency-domain ones alike) is computed entirely separately for Side I and Side II (Section 3.1). `Train_Labels.csv` still only ever has one label per file — no "both sides faulty" label exists anywhere in the data, so this cap is inherited from the task's own label design, unaffected by anything in v3. |
| 3 | *"The dataset is class-imbalanced..."* | **Addressed, unchanged from v2** | Training: `class_weight="balanced"` (LogReg), `auto_class_weights="Balanced"` (CatBoost) — identical to v2 (Section 3.2). Evaluation: scored by macro F1 throughout. v3 does not touch imbalance handling at all — the gain here comes entirely from giving the same imbalance-aware ensemble a feature that actually separates the classes better, not from a different imbalance strategy. |

## 3. How the algorithm works

### 3.1 Feature extraction (49 features per file)

The first 41 features are byte-for-byte identical to v2's: the 128 sensor
columns reshaped to `(10000, 8 car, 8 pos, 2 kind)` (kind 0 = vibration,
kind 1 = shock), then for each of {vibration, shock} x {RMS, kurtosis
(Fisher, bias-corrected), crest factor, peak-to-peak}, computed per (car,
position) and aggregated two ways over each side's four positions:

```
side1_{kind}_{stat}_mean = mean(matrix[:, [0,2,4,6]])   # positions 1,3,5,7
side1_{kind}_{stat}_max  = max (matrix[:, [0,2,4,6]])
side2_{kind}_{stat}_mean = mean(matrix[:, [1,3,5,7]])   # positions 2,4,6,8
side2_{kind}_{stat}_max  = max (matrix[:, [1,3,5,7]])
```

giving 32 features, plus `speed_kmh` (from the column-0 pulse train: 90
pulses/revolution, 0.85 m wheel diameter) and 8 speed-normalized RMS
variants (`{name}_norm = value / (speed_kmh + 1)`) — 41 total.

**New in v3** — 8 more features, same aggregation pattern applied to a
different statistic: for each of {vibration, shock}, the FFT power
spectrum is computed per (car, position) channel over the file's 10,000
samples (10,000 Hz sampling, confirmed against the info kit), summed over
the 0-100 Hz band, then aggregated exactly like every time-domain stat
above:

```
side1_{kind}_fft_low_mean = mean(low_band_energy[:, [0,2,4,6]])
side1_{kind}_fft_low_max  = max (low_band_energy[:, [0,2,4,6]])
side2_{kind}_fft_low_mean = mean(low_band_energy[:, [1,3,5,7]])
side2_{kind}_fft_low_max  = max (low_band_energy[:, [1,3,5,7]])
```

Why 0-100 Hz specifically, and not the other bands tried (100-500,
500-2000, 2000-5000 Hz, per-car spread, or cross-side ratios): Section 5
has the full empirical comparison. In short, 0-100 Hz was the only
addition that improved macro F1 on its own — the other bands individually
scored *below* the 41-feature baseline — and it lines up with the info
kit's own description of corrugation's physical signature as low-frequency.
Total: 41 + 8 = 49 features.

### 3.2 Soft-voting ensemble

Unchanged from v2 — see `model.py`, byte-for-byte identical file. Three
classifiers, weighted soft voting on `predict_proba` (weighted average of
class probabilities, then argmax):

| Model | Config | Weight |
|---|---|---|
| CatBoost | `iterations=200, depth=4, learning_rate=0.1, auto_class_weights="Balanced"` | 2 |
| XGBoost | `n_estimators=200, max_depth=4, learning_rate=0.1` | 2 |
| Logistic Regression | `penalty="l1", solver="saga", class_weight="balanced"`, on standardized features | 1 |

Kept unchanged deliberately: v2's own architecture search (documented in
v2's algorithm.md) already established this as the strongest combination
on the 41-feature set, and re-running that whole search on 49 features
would have muddied whether any gain came from the new feature or from a
re-tuned ensemble. Isolating the one variable was the point (Section 5).

### 3.3 Shipped weights, not a from-scratch fit every run

The fitted ensemble (CatBoost's own `.cbm`, XGBoost's own `.json`, and the
label encoder + scaler + logistic-regression coefficients in one
`sklearn_components.joblib`) lives in `weights/` and is committed to the
repo, fit on all 272 training files with the 49-feature set. By default
`run_pipeline.py` loads these and predicts immediately — no training data,
no `--data-dir`. Passing `--retrain` together with `--data-dir` reruns the
full fit (feature extraction, the 5x5 CV self-check, then a final fit) and
overwrites `weights/` — needed only after changing `features.py` or
`model.py`, or to verify the shipped weights still reproduce.

## 4. Explainability

Same two layers as v2, now covering 49 features instead of 41:

**Feature-level (always available, no model needed)**: every feature has a
stated physical meaning (Section 3.1) — which side, which signal kind,
which statistic (including the two new ones: FFT low-band energy, mean or
max across that side's 4 positions). For any file, the 8 new features
alone tell you whether there's excess low-frequency vibration/shock energy
concentrated on one side — the physical signature the info kit describes
corrugation as producing — independent of what the ensemble predicts.

**Model-level**: CatBoost and XGBoost both expose built-in feature
importances. Refitting on the full 272-file training set and inspecting
them: the new low-band FFT features rank alongside (not above)
`side1_vib_rms_max` / `side2_vib_rms_max` among the top features — the
model leans on both the original time-domain signal and the new
frequency-domain one, not one replacing the other. `predict_proba()` is
available on the fitted ensemble exactly as in v2 (Section 3.2).

**Current limitation, unchanged from v2**: feature importance as computed
here is *global*, not *local* (per-prediction), and `run_pipeline.py`
calls `.predict()`, discarding the probability distribution
`predict_proba()` already computes. Same fix available as in v2: exposing
that distribution (already done via `--diagnostics-output`, see below) or
adding per-prediction SHAP values is a small addition on top of an
already-fitted model.

## 5. What was tried and did not work well

### The real evidence: a head-to-head comparison, same protocol as v2

Every number below comes from running v2's own `diagnostics.py`
(`self_check_cv` — 5 repeats x 5-fold `StratifiedKFold`, macro F1, `random_state=0..4`),
unmodified, on different feature sets built from the same 272 training
files. The 41-feature baseline reproduces v2's shipped `0.8411 ± 0.0150`
exactly, confirming the comparison is apples-to-apples.

| Feature set | n features | Mean macro F1 | Side I F1 | Side II F1 |
|---|---|---|---|---|
| A: baseline (v2's 41) | 41 | 0.8411 ± 0.0150 | 0.676 | 0.866 |
| B: baseline + FFT, all 4 bands (0-100, 100-500, 500-2000, 2000-5000 Hz) | 73 | **0.8913 ± 0.0173** | 0.752 | 0.935 |
| C: baseline + cross-side ratios (vib/shock RMS, kurtosis, p2p ratios) | 45 | 0.8349 ± 0.0136 | 0.657 | 0.866 |
| D: baseline + per-car concentration spread (std of each side's per-car max) | 43 | 0.8271 ± 0.0271 | 0.657 | 0.843 |
| E: baseline + everything above combined | 79 | 0.8746 ± 0.0288 | 0.711 | 0.931 |
| **F: baseline + 0-100 Hz FFT band only (shipped)** | **49** | **0.8808 ± 0.0087** | **0.726** | **0.931** |

Two things stand out. First, **cross-side ratios (C) and per-car spread
(D) individually hurt** — both score below the 41-feature baseline. On
272 training files, extra features that don't carry real signal cost more
in variance than they give back, exactly the failure mode v2's own
"what was tried" section already flagged for other additions. Second,
**combining everything (E) is worse than the FFT band alone (B or F)** —
0.8746 vs. 0.8913/0.8808 — because C and D's noise dilutes the FFT
signal's benefit rather than complementing it.

### Which frequency band, and why 0-100 Hz was shipped over the full 4-band set

Isolating each FFT band individually (baseline + that one band's 8
features, nothing else):

| Band | Mean macro F1 | Side I F1 | Side II F1 |
|---|---|---|---|
| **0-100 Hz (low, shipped)** | **0.8808 ± 0.0087** | **0.726** | **0.931** |
| 100-500 Hz (mid) | 0.8327 ± 0.0195 | 0.652 | 0.865 |
| 500-2000 Hz (high) | 0.8268 ± 0.0277 | 0.636 | 0.865 |
| 2000-5000 Hz (vhigh) | 0.8311 ± 0.0220 | 0.652 | 0.864 |

Only the 0-100 Hz band improves on the baseline on its own — mid, high,
and vhigh each score *below* 0.8411 alone, meaning they carry mostly
noise for this task at this sample size, not complementary signal. This
matches the info kit's own account of corrugation's physical mechanism
("self-excited coupled vibration feedback," "low-frequency rumbling
noise") — the discriminative signal genuinely lives in the low-frequency
range the physics predicts it should, not spread evenly across the
spectrum.

The full 4-band set (B, 0.8913 ± 0.0173) does score marginally higher
than the low-band-only set (F, 0.8808 ± 0.0087) — but it does so by adding
24 features from three bands that, individually, actively hurt
performance, nearly doubling the feature count on a 272-row training set,
and with roughly double F's variance across CV repeats. That combination
of a small mean gain, higher variance, and three of its four ingredients
failing their own individual test reads as a less robust bet than it
looks on the headline number alone. **F (0-100 Hz only, 49 features
total) is what's shipped** — the more parsimonious, more stable, and more
physically-grounded choice. Both were also re-run on an independent
second set of CV seeds (`random_state=100..104`, never used in any of the
comparisons above) as a robustness check: baseline 0.8409 (matches the
first run's 0.8411), low-band-only 0.8935 (matches, in fact slightly
exceeds, the first run's 0.8808) — the gain is not an artifact of one
particular fold split.

### What v3 inherits unchanged from v2

**A 12-model comparison, SMOTE, Balanced Random Forest, dropping Random
Forest from the ensemble, and stacking** were all v2's own experiments,
run on the 41-feature set, before v3 existed — see v2's algorithm.md
Section 5 for the full account. None of them were re-run on 49 features:
the whole point of v3's comparison (above) was to change exactly one
thing at a time against the already-selected v2 ensemble, not to
re-open every prior decision at once. Nothing about the frequency feature
addition interacts with those conclusions in an obvious way (the ensemble
architecture and the imbalance handling are both feature-set-agnostic
choices), but they were not re-verified on 49 features and should be
treated as inherited, not re-confirmed.

**A "info kit had it right all along" note**: `v1_testing/features.py`
assumed a 10,000 Hz sampling rate for its own (much larger, 58-feature,
never-shipped) FFT feature set. That assumption turned out to be exactly
correct — confirmed against `Rail_Corrugation_Info_Kit.md` Section 2.1
("The sampling frequency is 10,000 Hz") — so the frequency axis under
every FFT feature in this file is on firm ground, not an inherited guess
carried forward uncritically.
