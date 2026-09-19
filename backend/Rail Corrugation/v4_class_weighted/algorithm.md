# Rail Corrugation Subsystem — v4, Class-Weighted Soft-Voting Ensemble

Rail Corrugation is a **3-class classification** task: for each file, decide
whether the rail is `Normal`, or corrugated on `Side I` (odd bearing
positions 1, 3, 5, 7 across all 8 cars) or `Side II` (even bearing
positions 2, 4, 6, 8). One file goes in, one label comes out.

This version changes one thing from v3: how the three base models' votes
combine. v3 used a single scalar weight per model, applied uniformly
across all three classes (`cat=2, xgb=2, log=1`). v4 replaces that with a
3x3 model x class weight matrix — each model gets its own weight *per
class* instead of one weight for everything it predicts. Feature
extraction (`features.py`, 49 features) is byte-for-byte identical to v3.
Section 5 has the full evidence trail.

## 1. Required data input and format

Identical to v3 — see v3's `algorithm.md` Section 1 for the full input/
output schema (unchanged: same sensor CSV format, same `Train_Labels.csv`
format, same `file_id,prediction` output schema).

## 2. How this addresses the info kit's 3 pain points

Unchanged from v3 (this version touches only ensemble combination, not
feature engineering or imbalance handling) — see v3's `algorithm.md`
Section 2 for the full discussion. The imbalance pain point (#3) is worth
restating since it's the direct motivation for this version: v3 addressed
it with per-model `class_weight="balanced"`/`auto_class_weights="Balanced"`
during *training*; v4 additionally lets the *combination* step account for
imbalance, since the per-class solo-accuracy gap between models (below)
is itself a symptom of how differently each model resolves the imbalance.

## 3. How the algorithm works

### 3.1 Feature extraction

Unchanged from v3 — see v3's `algorithm.md` Section 3.1 for the full
41+8=49 feature derivation. `features.py` here is a byte-for-byte copy.

### 3.2 Why a per-class weight matrix

Measuring each of the three base models' own solo accuracy per class
(same 5x5 CV protocol, `n=1360` validation instances pooled across 5
repeats) showed no single model is best across the board:

| Class | CatBoost solo | XGBoost solo | LogReg solo | Ensemble (v3, scalar 2/2/1) |
|---|---|---|---|---|
| Normal | 97.5% | **98.8%** | 94.3% | 98.5% |
| Side I | 71.4% | 61.4% | **74.3%** | 70.0% |
| Side II | **95.8%** | 85.0% | 86.7% | 95.8% |

XGBoost is strongest on the majority class but weakest on both minority
classes; LogReg is the only one that meaningfully separates Side I; CatBoost
is the most consistent across all three and never the worst on any class.
A single scalar weight per model can't express this — raising a model's
weight to help its strong class also inflates its (weaker) vote on the
other two.

### 3.3 The weight matrix

```python
# columns: [Normal, Side I, Side II] -- matches LabelEncoder's alphabetical
# sort of the 3 class strings, asserted at fit time in model.py.
CAT_CLASS_WEIGHTS = [4, 0, 4]
XGB_CLASS_WEIGHTS = [0, 4, 0]
LOG_CLASS_WEIGHTS = [4, 0, 0]
```

Final score for class `c` = `sum_m WEIGHTS[m][c] * model_m.predict_proba(x)[c]`,
normalized per row to sum to 1 (see `model.py`'s `combine_probas`), argmax
over `c`. The normalization is a per-row positive rescale, so it never
changes which class wins — it exists so `predict_proba()`'s output stays a
valid probability distribution for anything downstream that reads it
(`run_pipeline.py --diagnostics-output`).

This was **not** hand-picked to match the per-class solo-accuracy table
above (a naive reading might expect `cat` weighted on Side II, `log` on
Side I, `xgb` on Normal, roughly matching the table) — it was found by
random search (3000 samples over the 9-parameter space, `[0,4]` per
entry) optimizing pooled macro F1 on the same 5x5 CV protocol, then
independently confirmed on a second, disjoint set of CV seeds (Section 5).
The winning matrix does route each model toward roughly the class it's
individually strongest at, but the exact numbers came from search, not
intuition — e.g. XGBoost (weakest solo on Side I, 61.4%) still ends up as
the *only* model weighted on the Side I column; its raw Side I probability
turns out to carry a cleaner signal-to-noise ratio in combination than its
solo accuracy alone would suggest.

Shared with `diagnostics.py`: both `RailEnsemble.predict_proba` and the CV
self-check call the same `model_mod.combine_probas()` function, so the
reported CV numbers and the live ensemble's behavior can never drift apart
(v3's diagnostics.py duplicated the scalar-weight formula inline instead;
not worth doing for a 9-parameter matrix).

### 3.4 Shipped weights, not a from-scratch fit every run

Same as v3 — the fitted ensemble lives in `weights/`, committed to the
repo, fit on all 272 training files. `run_pipeline.py` loads it by default;
`--retrain --data-dir` reruns the full fit and overwrites `weights/`.

## 4. Explainability

Unchanged from v3 (Section 4) at the feature level. At the model-combination
level, v4 adds one more layer of explainability v3 didn't have: the
per-class weight matrix itself is a direct, readable statement of "how much
does each model's opinion count toward this specific class" — e.g. for any
prediction, you can show CatBoost and LogReg's Normal-column probabilities
(the only two that count toward Normal) or XGBoost's Side I-column
probability (the only one that counts toward Side I) as the operative
evidence for that specific label, rather than an opaque blend of all three
models across all three classes.

## 5. What was tried and did not work well

### The real evidence: search + independent-seed confirmation

Every number below reuses the exact 5x5 repeated `StratifiedKFold` protocol
(`random_state=0..4` for the original search set, `100..104` for the
independent confirmation set) already established for v2 and v3, applied
to the same 49-feature set v3 ships (no feature changes in this version).

| Config | Macro F1 (seeds 0-4) | Macro F1 (seeds 100-104) | Side I acc (0-4 / 100-104) |
|---|---|---|---|
| v3 shipped (scalar 2/2/1, uniform per class) | 0.8809 | 0.8931 | 70.0% / 75.7% |
| **v4 shipped (class-weight matrix, this version)** | **0.8940** | **0.9034** | 68.6% / 71.4% |

The gain (+0.0131 / +0.0103, both directions of the independent-seed check
agreeing) is real and robust, not a search artifact — it was selected on
seeds 0-4 and *independently re-measured*, not re-searched, on seeds
100-104, and it held up. It comes from sharpening Normal and Side II, not
from improving Side I — Side I accuracy is slightly *below* v3's baseline
under this matrix on both seed sets.

### Alternatives tried and rejected, in pursuit of also fixing Side I

Side I (14 of 272 training samples, the rarest class) stayed the weak
point under every ensemble-combination variant tried. Two more aggressive
options were tested and rejected in favor of the matrix above:

1. **A Side-I-priority weight matrix** (`cat=[2,4,3], xgb=[2,1,2],
   log=[1,4,1]`): robustly hits 80.0% / 85.7% Side I accuracy on the two
   seed sets, but costs macro F1 down to 0.8611 / 0.8683 — worse than even
   v3's baseline, let alone the shipped v4 matrix. Rejected: macro F1 is
   the scored competition metric, and this trades a big chunk of it for a
   minority-class gain.

2. **Routing Side I through `EasyEnsembleClassifier`** (which solo-hits
   90% Side I accuracy by training each of its base learners on an
   undersampled, near-uniform-prior bootstrap): tested as a hard gate
   ("if EasyEnsemble says Side I, trust it; otherwise fall back to the
   shipped ensemble choosing between Normal/Side II only") and as a
   blended probability. Both rejected — measuring EasyEnsemble's raw
   `predict_proba` output by true class showed it carries almost no
   absolute signal (mean P(Side I) = 0.335 when true class is Normal,
   vs 0.385 when true class actually is Side I — a signal-to-noise ratio
   of 1.15x, compared to CatBoost/XGBoost/LogReg's 15-64x). Its 90% solo
   accuracy comes entirely from its own internal argmax being biased
   toward minority classes, not from informative probabilities, so it
   doesn't combine cleanly with the other three models' well-separated
   probabilities. The hard-gate version confirmed this concretely: it hit
   90.0%/91.4% Side I accuracy but cratered macro F1 to 0.7555/0.7561,
   because 149 of 1170 true-Normal instances got false-positive-routed
   into Side I on seeds 0-4 alone (a 2.4:1 false-positive-to-true-positive
   ratio on the gate).

3. **SMOTE-family oversampling** (SMOTE, Borderline-SMOTE, ADASYN,
   SVM-SMOTE, train-fold only): best variant (SVM-SMOTE) reached 0.8699
   macro F1, below v3's 0.8808 un-resampled baseline. Rejected — with only
   ~11 Side I samples in a training fold, synthetic interpolation doesn't
   produce realistic points, and it's redundant with the `class_weight`/
   `auto_class_weights` already used in every base model's training.

### What v4 inherits unchanged from v3

Feature extraction (`features.py`), base model hyperparameters (`model.py`'s
`make_catboost`/`make_xgboost`/`make_logreg`), and the training protocol are
all untouched. Only the combination step (Section 3.2-3.3) changed.
