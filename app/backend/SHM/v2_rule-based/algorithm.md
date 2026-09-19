# SHM Subsystem — Rule-Based Cumulative Damage Prediction Algorithm

SHM is a **regression** task: predict a single cumulative fatigue damage
value for each dynamic-stress file. There is no segmentation, no
classification threshold, and no ranking or tiebreak step — one file goes
in, one number comes out, from a single calibrated physical formula.

## 1. Required data input and format

### Input

A single-column CSV with no header: one raw dynamic-stress reading per
row. Every file (training and test alike) has the same row count.

### Labels (for refitting the calibration constant only, not needed to predict)

`filename,damage` — one row per training file. The fitted constant `C`
is already shipped in `artifacts/calibration.json` (see Section 3), so
cloning this repo and running `run_pipeline.py --input <files>` needs
nothing beyond the files to predict on.

### Output

One row per file:

```
file_id,prediction
test01.csv,0.032277
```

## 2. How this addresses the info kit's 2 pain points

| # | Pain point | Status | Mechanism |
|---|---|---|---|
| 1 | *"Traditional Miner's linear damage rule relies on manual rainflow counting and time-series statistics, lacking automation, computational efficiency, and real-time capability."* | **Addressed on automation and efficiency; partially on real-time** | Rainflow counting is fully automated via the `rainflow` package's ASTM E1049 implementation — no manual counting anywhere. Measured, not assumed: a full rainflow + Miner's-proxy pass over one complete 581,120-row file takes ~0.58s; a single-file prediction end-to-end (loading the shipped calibration, no `--data-dir`) took 1.32s wall time in a fresh run, most of it Python/pandas import overhead. What's not solved: this is a **batch** algorithm — every prediction reprocesses a complete raw series from scratch. Nothing here maintains rainflow's cycle-counting state incrementally as new samples arrive, which is what genuine real-time / streaming capability would require. |
| 2 | *"It lacks machine-learning-based time-series intelligence, making it difficult to perform rapid, online fatigue damage and remaining-life assessments on massive dynamic-stress time-series data, and cannot meet the demands of dynamic, real-time intelligent evaluation."* | **Not addressed, on both halves** | ML-based intelligence is deliberately **not** used in the shipped model — tested, not skipped: Ridge and Ridge+XGBoost score 0.83–0.84 against the physics model's 0.974 under the same validation (Section 5), so adopting ML here would make predictions worse, not more intelligent, at this sample size (64 files). Remaining-life / future-damage assessment isn't built into the pipeline at all — a single static cumulative-damage number per file is all `run_pipeline.py` ever outputs. Separately explored: within-file short-horizon extrapolation degrades fast (backtested MAPE 11.8%, growing to 23.9% at the far end of the horizon), and genuine service-life forecasting across the vehicle's operating history isn't derivable from this dataset at all, since the info kit states file numbers are randomly assigned and uncorrelated with recording order or damage level — there's no time axis to forecast along. |

## 3. How the algorithm works

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
Section 5). Prediction for any new file is then just:

```
damage_predicted = proxy / C
```

One fitted parameter, total. There is no feature vector, no model
architecture, and no training loop beyond that single division.

### Shipped calibration, not a from-scratch fit every run

`C` (plus the fixed exponent `m` and the training proxy range used for
extrapolation flagging) is committed to the repo in
`artifacts/calibration.json`. By default `run_pipeline.py` loads this
and predicts immediately — no training data, no `--data-dir`. Passing
`--retrain` together with `--data-dir` reruns the leave-one-out
self-check and refits `C` on all 64 training files, overwriting
`artifacts/` with the result — needed only after changing `physics.py`
or `model.py`, or to verify the shipped constant still reproduces.

## 4. Explainability

This is the most transparent of the four subsystems, by construction: `damage_predicted = proxy /
C` is one division of two real, individually inspectable numbers, with no feature selection, no
hidden interaction terms, and nothing approximated.

`proxy` is not an abstract score — it's built directly from the rainflow-extracted cycle list
(`(range, mean, count)` per cycle, via the standard ASTM E1049 algorithm), so "why is this file's
damage high" is answerable by looking at which cycles have the largest `range` (since
`proxy = Σ(count_i · range_i^5)`, a cycle's contribution grows with the 5th power of its range —
a handful of large-range cycles can dominate the sum even if most cycles are small). `C` is a
single fitted scalar, its value and derivation (the median of `proxy/damage` across the 64
labelled files) fully stated in Section 3.3 — there's no second parameter, no per-file
adjustment, and no version of this model that isn't traceable to those two numbers.

**Honest limit, not a gap to fix**: explainability here is total but coarse. The model can say
*which cycles* drove the damage number (the largest-range ones, via the fixed exponent) but has
no per-cycle-cause explanation beyond that — there is no mechanism, by design, for attributing
damage to anything other than a cycle's own range and count. That's the appropriate level of
explanation for a single-parameter physics formula; a more granular story would require a model
this dataset's 64 labelled files can't support (see Section 5's ML-regressor results).

## 5. What was tried and did not work well

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
