# SHM Subsystem — Rule-Based Cumulative Damage Prediction Algorithm

SHM is a **regression** task: predict a single cumulative fatigue damage
value for each dynamic-stress file. There is no segmentation, no
classification threshold, and no ranking or tiebreak step — one file goes
in, one number comes out, from a single calibrated physical formula.

## 1. Required data input and format

### Input

A single-column CSV with no header: one raw dynamic-stress reading per
row. Every file (training and test alike) has the same row count.

### Labels (for calibration only, not needed for prediction)

`filename,damage` — one row per training file.

### Output

One row per file:

```
file_id,prediction
test01.csv,0.032277
```

## 2. How the algorithm works

### Step 1 — rainflow cycle extraction

The raw stress series is reduced to a set of stress cycles via rainflow
counting (the ASTM E1049 standard algorithm), each with a range, a mean,
and a count. This is the same method the info kit names as the standard
preprocessing step for fatigue analysis on this kind of data.

### Step 2 — Miner's-rule proxy

```
proxy = Σ (count_i · range_i^m),   m = 5.0
```

This is proportional to Miner's cumulative damage rule under a power-law
S-N curve, `D = Σ(n_i/N_i)` with `N_i = C/σ_i^m` — up to the one unknown
material constant `C`. `m = 5.0` is fixed; only mean (static-offset)
stress, not the amplitude exponent, was found to matter for anything
beyond this.

### Step 3 — calibration and prediction

`C` is fit once, from the labelled training files, as the median of
`proxy / damage` across them (a median rather than a least-squares fit,
so a handful of large-damage files don't dominate the calibration — see
Section 3). Prediction for any new file is then just:

```
damage_predicted = proxy / C
```

One fitted parameter, total. There is no feature vector, no model
architecture, and no training loop beyond that single division.

## 3. What was tried and did not work well

**Generic time-domain summary statistics** (peak-to-peak range, RMS,
standard deviation, percentiles) correlate with damage on their own, but
not precisely enough to build a competitive predictor — the strongest
single generic statistic falls well short of what the physics
reconstruction below achieves.

**A generic ML regressor on engineered features** (Ridge regression, and
a Ridge+XGBoost ensemble, each on time-domain/frequency-domain feature
sets) scores dramatically worse than the physics model under the same
kind of cross-validation: roughly 0.83–0.84 mean fold score (MAPE
16–17%), against the physics model's 0.974 (MAPE 2.6%). With only 64
labelled files, a model with many learned parameters has nowhere near
enough data to outperform a model with one, even though it's given the
same underlying signal to work with.

**Sweeping the S-N exponent by correlation** rather than assuming a
textbook value found a sharp, unambiguous optimum at `m = 5.0` (not a
broad plateau) — this is what fixed the exponent used above, rather than
picking a round number from a materials handbook.

**Mean-stress correction (Walker model)**, using each rainflow cycle's
mean stress in addition to its range, via `σ_eq = σ_max^γ · σ_a^(1−γ)`
swept jointly with the exponent. The optimum sits at `γ = 0` — pure
stress amplitude, i.e. no benefit from incorporating mean stress at all.
This rules out mean-stress dependence as an explanation for the model's
residual error, rather than leaving it as an untested possibility.

**A flexible, non-parametric "learned S-N shape"** — binning cycle ranges
into buckets and fitting a non-negative weight per bucket, instead of
committing to a single power-law exponent — collapses badly under
leave-one-out validation (score fell to roughly 0.62, using only a
handful of the available bins in a full-data fit). Structurally similar
to Miner's rule (additive in cycle counts) but with far more free
parameters than 64 examples can support.

**An ML residual-correction model on top of the physics prediction**
(gradient boosting on secondary summary features, predicting the leftover
log-ratio between true and physics-predicted damage) gave at best a
marginal improvement under leave-one-out validation, and two of the four
feature choices tried made performance *worse*, not better — a sign of
overfitting rather than a real, exploitable pattern at this sample size.

**Manufacturing more training examples by sub-windowing files** (splitting
each file into quarters with proportionally scaled damage targets) was
checked for validity before being used: rainflow-on-quarters-summed
should reproduce rainflow-on-the-whole-file if the additivity assumption
behind this held. It didn't — a consistent 3–14% discrepancy across the
files checked, caused by large stress cycles that span a quarter boundary
getting truncated when the segments are processed independently. That
distortion is larger than the physics model's entire error budget, so
this was not pursued as a way to get more effective training data.
