# Rail Corrugation Subsystem — Rule-Based Feature Engineering + Soft-Voting Ensemble

Rail Corrugation is a **3-class classification** task: for each file, decide
whether the rail is `Normal`, or corrugated on `Side I` (odd bearing
positions 1, 3, 5, 7 across all 8 cars) or `Side II` (even bearing
positions 2, 4, 6, 8). One file goes in, one label comes out.

## 1. Required data input and format

### Input

A sensor CSV: one header row, 10,000 rows, 129 columns. Column 0 is
`Rotating speed` (despite the name, a 0/1 pulse train, not a continuous
speed reading — verified by inspection: every value in the column is
exactly 0 or 1). Columns 1-128 are `Vibration of bearing in position P of
car C` / `Shock of bearing in position P of car C`, for `P` in 1..8 and
`C` in 1..8, interleaved (vib, shock, vib, shock, ...) so that all 16
readings for one car appear together before the next car's.

### Labels (for retraining only, not needed to run predictions)

`filename,label` — one row per training file, label one of `Normal`,
`Side I`, `Side II`. The fitted ensemble is already shipped in
`weights/` (see Section 2), so cloning this repo and running
`run_pipeline.py --input <files>` needs none of this — no `Train/`, no
`Train_Labels.csv`, nothing beyond the files to predict on.

### Output

One row per file:

```
file_id,prediction
Test1.csv,Normal
Test13.csv,Side II
Test22.csv,Side I
```

## 2. How the algorithm works

### Step 1 — feature extraction (41 features per file)

The 128 sensor columns are reshaped to `(10000, 8 car, 8 pos, 2 kind)`
(kind 0 = vibration, kind 1 = shock). For each of {vibration, shock} x
{RMS, kurtosis (Fisher, bias-corrected), crest factor (peak/RMS),
peak-to-peak}, computed per (car, position) over the 10,000 rows, the
resulting `(8 car, 8 pos)` matrix is aggregated two ways over each side's
four positions per car:

```
side1_{kind}_{stat}_mean = mean(matrix[:, [0,2,4,6]])   # positions 1,3,5,7
side1_{kind}_{stat}_max  = max (matrix[:, [0,2,4,6]])
side2_{kind}_{stat}_mean = mean(matrix[:, [1,3,5,7]])   # positions 2,4,6,8
side2_{kind}_{stat}_max  = max (matrix[:, [1,3,5,7]])
```

giving 2 sides x 2 kinds x 4 stats x 2 aggregations = 32 features. Mean
and max are both kept per side because a single badly corrugated bearing
can sit under a side average that still looks unremarkable — max catches
the worst offender, mean gives the side-wide baseline.

A speed feature is derived from column 0's pulse train:

```
transitions = count(|diff(pulse)| > 0.5)
speed_kmh   = (transitions / 2 / 90) * pi * 0.85 * 3.6
```

(90 pulses/revolution encoder, 0.85 m wheel diameter — both fixed
constants reverse-engineered during EDA from the pulse frequency and a
plausible wheel size, not re-derived per file.) Every `*_rms_mean` and
`*_rms_max` feature (8 of them) gets a speed-normalized variant,
`{name}_norm = {value} / (speed_kmh + 1)`, since vibration/shock RMS
scales with how fast the train is moving and a raw RMS threshold would
conflate speed with corrugation severity. Total: 32 + 8 speed-normalized
+ 1 `speed_kmh` = 41 features.

### Step 2 — soft-voting ensemble

Three classifiers are trained on the same 41 features and combined by
weighted soft voting (weighted average of `predict_proba`, then argmax):

| Model | Config | Weight |
|---|---|---|
| CatBoost | `iterations=200, depth=4, learning_rate=0.1, auto_class_weights="Balanced"` | 2 |
| XGBoost | `n_estimators=200, max_depth=4, learning_rate=0.1` | 2 |
| Logistic Regression | `penalty="l1", solver="saga", class_weight="balanced"`, on standardized features | 1 |

CatBoost and XGBoost train on raw (unscaled) features; Logistic
Regression trains on features standardized with a `StandardScaler` fit on
the training data only. Final prediction is
`argmax(2*P_cat + 2*P_xgb + 1*P_logreg)`.

The two gradient-boosted trees get equal, double weight because they are
individually the two strongest models and roughly comparable in
strength; the linear model gets a single, smaller weight because on its
own it is markedly weaker (see Section 3), but it makes decision
boundaries the two tree models cannot (see Section 3's CatBoost/XGBoost
asymmetry note) so it still moves the vote in cases the trees agree on
incorrectly.

### Step 3 — shipped weights, not a from-scratch fit every run

The fitted ensemble (CatBoost's own `.cbm`, XGBoost's own `.json`, and
the label encoder + scaler + logistic-regression coefficients in one
`sklearn_components.joblib`) lives in `weights/` and is committed to the
repo. By default `run_pipeline.py` loads these and predicts immediately
— no training data, no `--data-dir`, no refitting. Passing `--retrain`
together with `--data-dir` reruns the full fit described above (feature
extraction on all 272 training files, the 5x5 CV self-check, then a
final fit on all of them) and overwrites `weights/` with the result —
needed only after changing `features.py` or `model.py`, or to verify the
shipped weights still reproduce.

## 3. What was tried and did not work well

**A 12-model comparison** (Random Forest, Balanced Random Forest, plain
and SMOTE-augmented XGBoost/LightGBM/CatBoost, Logistic Regression, and
several voting/stacking combinations of these) was run under the same
5x5 repeated `StratifiedKFold` protocol, scored by macro F1, before
settling on the final ensemble. Individual-model results:

| Model | Macro F1 (mean ± std) |
|---|---|
| XGBoost (solo) | 0.8126 ± 0.0255 |
| CatBoost (solo, `auto_class_weights="Balanced"`) | 0.8250 ± 0.0169 |
| LightGBM (solo, `class_weight="balanced"`) | 0.8065 ± 0.0401 |
| Random Forest (solo, `class_weight="balanced"`) | 0.7685 ± 0.0119 |
| Logistic Regression (solo, standardized) | 0.7570 ± 0.0247 |
| XGBoost + SMOTE oversampling | 0.7745 ± 0.0434 |
| Balanced Random Forest (`imblearn`) | 0.6760 ± 0.0123 |
| **CatBoost(2) + XGBoost(2) + LogReg(1) soft vote (final)** | **0.8411 ± 0.0150** |

The best individual model was XGBoost alone at ~0.81; every combination
tried scored below the final CatBoost+XGBoost+LogReg vote.

**SMOTE oversampling** was tried on every boosted-tree model, not just
XGBoost, and consistently made things worse (XGBoost 0.8126 -> 0.7745
with SMOTE). With only 14 Side I examples across the whole training set,
SMOTE's nearest-neighbor interpolation has almost nothing real to
interpolate between — the synthetic minority points it manufactures
amplify whatever noise sits between those 14 points rather than
reinforcing real signal, and the resulting decision boundary overfits to
that noise.

**Balanced Random Forest** (`imbalanced-learn`'s per-tree
undersample-then-bag classifier) was tried as a more principled
imbalance-handling alternative to plain class weighting, and scored worst
of everything tried (0.6760 ± 0.0123) — undersampling the majority class
per tree throws away most of the 234 Normal examples' signal on every
tree, which costs far more than the imbalance handling gains back on this
dataset's class ratio.

**Random Forest was dropped from the final ensemble.** A plain
`class_weight="balanced"` Random Forest scored 0.7685 solo — clearly
weaker than either boosted-tree model — and adding it into the ensemble
made results worse, not better: CatBoost(2)+XGBoost(2)+LogReg(1) at
0.8411 dropped to 0.8351 with Random Forest(1) added, and only recovered
to 0.8391 (still below the RF-free ensemble) even after doubling RF's
weight. An XGBoost+RandomForest-only vote scored 0.7959, actually *below*
XGBoost solo's 0.8126 — Random Forest is redundant with XGBoost's own
tree-based decision boundaries rather than complementary to them, so
including it just dilutes the vote with a weaker, correlated opinion.

**Stacking** (CatBoost + XGBoost out-of-fold predictions feeding a
logistic-regression meta-learner, `StackingClassifier` with internal
3-fold CV to generate out-of-fold features) scored 0.8066 ± 0.0182,
clearly below the same two models combined by soft voting. With only 14
Side I examples, the meta-learner's internal CV splits leave as few as
3-4 Side I examples per training fold to learn a meta-decision boundary
from — not enough to learn a reliable combination rule, whereas voting's
fixed weights need no additional data to fit at all.

**CatBoost's `auto_class_weights="Balanced"` vs. XGBoost's unweighted
training** — this is not an incidental detail but the reason the ensemble
beats either model alone. Run solo and compared class-by-class (pooled
across the same 5x5 CV protocol):

| Model | Side II recall | Side II F1 | Normal precision |
|---|---|---|---|
| CatBoost (balanced) | 90.8% | 0.829 | 98.3% |
| XGBoost (unweighted) | 80.8% | 0.836 | 97.3% |

CatBoost's class weighting pushes it to catch far more true Side II cases
(90.8% recall vs. 80.8%), at a small cost to how cleanly it separates
Normal (98.3% vs. 97.3% precision — CatBoost is slightly more willing to
call a borderline file Side II instead of Normal). XGBoost's unweighted
training does the opposite trade-off. Because the two models err in
different, complementary directions rather than making the same
mistakes, averaging their probabilities recovers cases either one alone
would miss — which is why the ensemble's 0.8411 clears both solo scores
by a wider margin than either model's own run-to-run noise (±0.015-0.025)
would explain by chance.
