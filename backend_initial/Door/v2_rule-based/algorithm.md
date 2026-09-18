# Door Subsystem — Rule-Based Segmentation & Classification Algorithm

Door is a **segmentation + binary classification** task (find each door
cycle in a continuous stream, label it Normal or Abnormal resistance),.

## 1. Required data input and format

### Input

A sensor CSV: one header row, one row per 20ms sample (50 Hz), 17 columns.
`Datetime` uses the format `Year-Month-Day-Hour-Minute-Second-Millisecond`,
not zero-padded (e.g. `2023-7-5-0-11-17-664`). The algorithm reads three
columns: `Motor current(mA)`, `Door leaf position`, and `Close command`.

### Output

One row per **predicted segment** (not per file — the input is a single
continuous stream that may contain many cycles):

```
start_time,end_time,prediction
2023-7-5-0-0-0-0,2023-7-5-0-0-3-760,Normal
```

No `file_id` column and no `operation` column.

### Train data (for refreshing the OOD comparison set only, not needed to predict)

`Train.csv` and `Train_Segments_Answer.csv`, same format. The two
classification thresholds are fixed constants (see Section 2) that don't
need training data to use — the per-operation feature ranges used for
out-of-range flagging are already shipped in `artifacts/train_ranges.json`,
so cloning this repo and running `run_pipeline.py --input <file>` needs
nothing beyond the file to predict on.

## 2. How the algorithm works

### Step 1 — segmentation

The input is not a genuinely continuous recording of idle-plus-active time
— it is individual door cycles concatenated back-to-back with the time
between them removed. Because of that, every row-to-row timestamp delta
falls into one of two regimes: about 20ms within a cycle, or several
seconds or more between cycles. A cycle boundary is declared wherever the
gap between consecutive rows' timestamps exceeds `GAP_THRESHOLD_S = 0.1`
seconds.

Two guardrails then discard any candidate segment shorter than
`MIN_CYCLE_ROWS = 50` rows or longer than `MAX_CYCLE_ROWS = 250` rows, as
a safety margin against spurious micro-segments if the input stream ever
contains genuine idle-period noise.

### Step 2 — Open vs. Close

Read directly from `Close command` (mean > 0.5 → Close, otherwise Open).
This is a lookup, not a prediction — it only exists because Step 3
branches on it.

### Step 3 — classification

The decision branches by operation, because the physical signature of
resistance differs by direction:

**Close cycles** — decided by peak current (`cur_max`):

```
Abnormal resistance if cur_max < 2060 mA, else Normal
```

Abnormal Close cycles draw a *lower* peak current than Normal ones, not a
higher one — the controller's stall/torque-limit logic caps current once
it senses resistance while closing.

**Open cycles** — peak current does not separate the classes, so the rule
instead uses mean current over the whole cycle (`cur_mean`):

```
Abnormal resistance if cur_mean > 700 mA, else Normal
```

Abnormal Open cycles draw sustained excess current rather than a distinct
peak, because an Open stroke has to keep pushing against the obstruction
for longer, where a Close stroke's controller cuts power short instead.

These two thresholds, plus the two guardrail row-count bounds, are the
only constants the algorithm uses. There is no model fit and no ranking
or tiebreak step — each segment is assigned exactly one label
independently of every other segment.

### Shipped comparison set, not a from-scratch computation every run

The per-operation feature ranges `flag_out_of_range()` checks against
(`artifacts/train_ranges.json`) are committed to the repo. By default
`run_pipeline.py` loads this and predicts immediately — no training
data, no `--data-dir`. Passing `--retrain` together with `--data-dir`
reruns the self-check against `Train_Segments_Answer.csv` and rebuilds
the ranges from the 110 labelled cycles, overwriting `artifacts/` with
the result — needed only after changing `rules.py`, or to verify the
shipped ranges still reproduce.

## 3. What was tried and did not work well

**An elaborate position/current segmentation rule** (direction reversal
>200 position units; current dropping from >1500mA to <200mA, combined
with command-change or position-reset logic) was the original approach.
Cross-checking it against the timestamp-gap rule showed the two produce
**byte-identical boundaries** in every case — the elaborate rule was never
actually finding cycle edges from motor behaviour, it was coincidentally
firing at the same place a simple time discontinuity already marks.
Replaced with the gap rule: simpler, and it doesn't carry a false
impression that the model detects boundaries from motor dynamics when it
doesn't need to.

**Validating segmentation by matching only start-index sets** against a
labelled answer file reports a boundary-count match but never checks end
times, never computes IoU, and never implements the actual scoring
formula at all. Replaced with a proper IoU-weighted matching function that
implements the real metric, run against the pipeline's own detected
segments rather than against features pre-computed at known-correct
boundaries.

**A random forest / gradient boosting classifier** on a larger feature
vector (motor current/voltage/back-EMF statistics, a resistance proxy,
etc.) also reaches a strong classification result, but carries a concrete
robustness cost the two-threshold rule doesn't: `GradientBoostingClassifier`
hard-crashes (`ValueError: Input X contains NaN`) on a segment whose
resistance-proxy feature is undefined (an all-low-current segment produces
an all-NaN resistance value), a failure mode `RandomForestClassifier`
tolerates but GBM does not. A fitted model is also materially harder to
justify than "peak current below X, or mean current above Y," which lines
up with an intuitive physical story (stall/torque limiting on Close,
sustained resistance on Open) that a black-box decision boundary wouldn't
surface without extra explainability work.

**Scoring classification accuracy from features computed at the answer
file's known-correct boundaries**, rather than from the boundaries the
segmentation step itself detects, answers an easier question than "how
does the full pipeline do end to end" — the two only coincide if
segmentation happens to be perfect. Replaced by running segmentation and
classification together on the raw stream before scoring.

**Trusting a prediction without checking it against the range of values
the classifier's decision actually relies on** was the original approach
for new, unlabeled data. Replaced with a check that flags any predicted
segment whose decision-relevant features (peak current, mean current,
early-cycle mean current, position range) fall outside anything
previously seen for that operation — not because the prediction is
necessarily wrong, but because it's then an extrapolation rather than an
interpolation, which is a meaningfully different kind of confidence worth
surfacing rather than hiding behind a bare label.
