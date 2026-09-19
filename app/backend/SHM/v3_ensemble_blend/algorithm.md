# SHM Subsystem — v3, Physics + ML Ensemble Blend

SHM is a **regression** task: predict a single cumulative fatigue damage
value for each dynamic-stress file. Same task, same input/output schema
as v2 — see v2's `algorithm.md` Section 1 for the full I/O spec (unchanged
here).

**This version was shipped on explicit instruction despite the evidence
against it.** Every individual ML model tried (19 of them, across
regression, bagging, boosting, and kernel-method families) scored below
the pure physics model under honest leave-one-out cross-validation — see
Section 5 for the full sweep. This module blends 4 members anyway
(physics + linear regression + bagged Ridge + Extra Trees), found to give
a +0.0006 LOO gain over pure physics (0.9744 → 0.9750) by a weight search
that has no independent confirmation set to check it against, unlike
every weight search done for Rail Corrugation this session (all of which
were validated on a second, disjoint set of CV seeds before being
trusted). This gain is the same size as several changes this session
already confirmed were pure noise on this exact dataset (SHM's own
calibration-loss fix and exponent re-sweep, both ±0.0005). **Treat this
version's improvement as unconfirmed; it exists because it was asked for,
not because the evidence supports it being better than v2.**

## 1. Required data input and format

Identical to v2 — see v2's `algorithm.md` Section 1.

## 2. How this addresses the info kit's 2 pain points

Unchanged from v2's assessment (this version doesn't change the
rainflow/Miner's-rule core, only adds 3 ML members alongside it) — see
v2's `algorithm.md` Section 2.

## 3. How the algorithm works

### 3.1 The 4 ensemble members

1. **Physics** (`physics.py`, unchanged from v2): rainflow cycle
   extraction (ASTM E1049) → Miner's-rule proxy (`Σ count_i · range_i^5`)
   → `damage = proxy / C`, `C` fit as the median of `proxy/damage` across
   training files.
2. **Linear regression**, fit on the physics proxy alone (1 feature,
   standardized).
3. **Bagged Ridge** (`BaggingRegressor(Ridge(alpha=0.1))`, 300
   estimators), also on the physics proxy alone.
4. **Extra Trees** (300 estimators), fit on a 47-feature set: 17
   time-domain stats, 11 rainflow-cycle-derived stats, 14 Miner's-proxy
   variants at 7 exponents (raw + log), 5 FFT-based stats — see
   `features.py` for the full list.

### 3.2 The blend weights

```python
PHYSICS_WEIGHT = 7
LINEAR_WEIGHT = 0
BAGGED_RIDGE_WEIGHT = 0
EXTRA_TREES_WEIGHT = 1
```

Final prediction = weighted average of the 4 members' predictions,
divided by the total weight (8). Found by grid-searching all
11×11×11×11 integer weight combinations (0–10 each) against the LOO
predictions of all 64 training files, all 4 members refit per fold. The
winning combination is effectively physics with a small Extra Trees
correction; linear regression and bagged Ridge contribute nothing at
their found weight of 0 (still fit and predicted-from at inference time,
kept swappable for a future retrain that might find a different optimum
on different data — see model.py's module docstring).

### 3.3 Shipped artifacts, not a from-scratch fit every run

`artifacts/physics_calibration.json` (C, m) and
`artifacts/sklearn_components.joblib` (the proxy scaler, and the 3 fitted
sklearn/imblearn-style regressors) are committed to the repo, fit on all
64 training files. `run_pipeline.py` loads these by default;
`--retrain --data-dir` reruns the full fit and overwrites `artifacts/`.

## 4. Explainability

Weaker than v2 by construction — v2's `damage = proxy/C` is one
inspectable division; this blend adds 3 more opaque members (a linear
model, a 300-estimator bagged ensemble, a 300-tree Extra Trees forest) on
top. The physics member still dominates the blend (weight 7 of 8), so
`proxy/C` remains the primary explanation for any prediction, with the
Extra Trees member as an unexplained ~12.5%-weight adjustment.

## 5. What was tried and did not work well

### The honest evidence: nothing beat physics on its own

Every model below was evaluated with the same honest leave-one-out
protocol as v2's physics model (refit on the other 63 files every fold,
scored with the real competition metric, `max(0, 1-MAPE)`).

| Model | LOO score | vs physics (0.9744) |
|---|---|---|
| **Physics (v2, unchanged)** | **0.9744** | — |
| Linear regression, physics-proxy-only feature | 0.9726 | -0.0018 |
| Ridge, full 47 features, log target | 0.9715 | -0.0029 |
| Bagged Ridge, physics-proxy-only feature | 0.9714 | -0.0030 |
| Ridge, physics-proxy-only feature | 0.9711 | -0.0033 |
| Extra Trees, full 47 features | 0.9581 | -0.0163 |
| Gradient Boosting, full 47 features | 0.9548 | -0.0196 |
| K-NN (k=3), physics-proxy-only feature | 0.9540 | -0.0204 |
| ElasticNet, full 47 features | 0.9519 | -0.0225 |
| Gaussian Process (RBF + white noise), full 47 features | 0.9514 | -0.0230 |
| Bagged linear regression, full 47 features | 0.9438 | -0.0306 |
| Random Forest, full 47 features | 0.9424 | -0.0320 |
| XGBoost, physics-proxy-only feature | 0.9384 | -0.0360 |
| Bagged decision tree, full 47 features | 0.9278 | -0.0466 |
| Ridge, full 47 features | 0.9270 | -0.0474 |
| Linear regression, full 47 features | 0.9236 | -0.0508 |
| XGBoost, full 47 features | 0.9110 | -0.0634 |
| Lasso, full 47 features | 0.9074 | -0.0671 |
| K-NN (k=5), full 47 features | 0.7648 | -0.2096 |
| Linear regression, physics-proxy-only, log target | 0.6008 | -0.3737 |
| SVR (RBF kernel), full 47 features | 0.5241 | -0.4503 |

Pattern: the closest ML results are the ones that collapse toward using
just the physics proxy as their dominant signal (linear/bagged-Ridge on
the proxy alone) — they approximate the physics formula, they don't beat
it. The full-47-feature versions of the same model families score
meaningfully worse than their proxy-only counterparts, the signature of
64 samples being too few to support that many learned parameters.

### Ensembling the 3 best ML candidates alone (no physics) — still lost

Averaging linear regression, bagged Ridge, and Extra Trees together
(equal weights): LOO score 0.9719, still below physics. Grid-searching
their 3-way weights: best found was 0.9731 (linear=9, bagged_ridge=0,
extra_trees=2), still below physics's 0.9744. Three models individually
weaker than physics, and not decorrelated enough from each other, don't
combine into something stronger than physics.

### Calibration-loss and exponent refinements (documented in v2, re-verified here)

Two more single-parameter refinements to the pure physics model were
tested and also landed at noise level, establishing the scale against
which this version's +0.0006 blend gain should be judged:

- Fitting `C` to directly minimize LOO mean APE (the actual scored loss)
  instead of the median-of-ratios estimator: LOO score 0.9746 (+0.0001).
- Fine-sweeping the S-N exponent `m` around the shipped 5.0: best found
  was `m=5.02` at 0.9749 (+0.0005); the exponent sits at a genuinely
  sharp peak (score falls to 0.93 by `m=4.75` or `m=5.25`).

### What v3 inherits unchanged from v2

`physics.py` (rainflow extraction, Miner's proxy, the calibration
formula for the physics member specifically) is untouched. The training
protocol (leave-one-out validation before any refit) is the same idea,
extended to cover all 4 members instead of just the physics constant.
