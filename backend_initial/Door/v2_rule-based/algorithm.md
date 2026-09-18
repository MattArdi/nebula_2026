# Door Subsystem — Rule-Based Segmentation & Classification Algorithm

This document describes the algorithm in `segmentation.py` / `rules.py` /
`diagnostics.py` / `run_pipeline.py`: what data it needs, how it actually
works, and what alternatives were tried and rejected along the way. All
numbers below are from re-running the pipeline against the real dataset
while writing this document — see the verification note at the end.

Note on scope: Door is a **segmentation + binary classification** task
(find each door cycle in a continuous stream, label it Normal or Abnormal
resistance), not a ranking task — there is no CUSUM and no "which of
several candidates" ranking step here, unlike the ACV subsystem. Section 2
below covers the two things this algorithm actually does: how a cycle
boundary is decided, and how the Normal/Abnormal decision is made,
including the one place a decision genuinely branches (Open vs. Close).

## 1. Required data input and format

### Directory layout

```
<data-dir>/
├── Train.csv                    # continuous sensor stream, many cycles back-to-back
├── Train_Segments_Answer.csv    # ground truth: one row per true cycle
└── Test.csv                     # continuous sensor stream, unlabeled
```

### Sensor CSV schema (`Train.csv` / `Test.csv`)

17 columns, one header row, one row per 20ms sample (50 Hz): `Datetime`
(format `Year-Month-Day-Hour-Minute-Second-Millisecond`, not zero-padded,
e.g. `2023-7-5-0-11-17-664`) plus 16 sensor/flag readings. The two this
algorithm actually reads are `Motor current(mA)` and `Door leaf position`;
`Close command` is read to determine Open vs. Close. Every row in these
files belongs to exactly one door cycle — confirmed directly:
`sum(n_rows)` across all 110 answer-file segments equals `len(Train.csv)`
exactly, so there is no idle/between-cycle telemetry recorded at all (see
Section 2.1).

### Answer file schema (`Train_Segments_Answer.csv`)

`segment_id, start_time, end_time, operation, status, n_rows` — one row
per true cycle. `status` (`Normal` / `Abnormal resistance`) is the label
this algorithm predicts; `operation` (`Open` / `Close`) is informational
only, not something the model needs to output.

### Output format

`run_pipeline.py` writes one row per **predicted segment** (not per file —
`Test.csv` is a single continuous stream):

```
start_time,end_time,prediction
2023-7-5-0-0-0-0,2023-7-5-0-0-3-760,Normal
```

No `file_id` column (Door has none) and no `operation` column (not
required). Verified byte-for-byte against the header and separator in
`04_Example_Submission/door_predictions.csv`.

## 2. How the algorithm works

### 2.1 Step 1 — segmentation: finding each cycle's boundaries

`Train.csv`/`Test.csv` are not genuinely continuous recordings — they are
110 (Train) individual door cycles concatenated back-to-back with the idle
time between them stripped out of the file entirely. Every row-to-row
timestamp delta is either exactly 20ms (inside a cycle) or 10+ seconds
(between cycles), with nothing in between — confirmed directly on both
files. A cycle boundary is detected the moment the timestamp gap exceeds
`GAP_THRESHOLD_S = 0.1` seconds (`segmentation.detect_cycles`); the actual
margin between the two regimes (20ms vs. 10s+) is enormous, so the exact
threshold value doesn't matter within a very wide range.

Two guardrails (`flag_implausible`, `MIN_CYCLE_ROWS=50`,
`MAX_CYCLE_ROWS=250`) discard any candidate segment whose row count falls
outside what a real cycle ever demonstrated in training (137–190 rows) —
insurance against a genuinely continuous stream (with real idle-period
noise) producing spurious micro-segments, even though nothing in the
current Train/Test files ever triggers this.

**This gap rule was cross-checked against a more elaborate position- and
current-based rule** (direction reversal, current dropping from >1500mA to
<200mA) and the two produce **identical boundaries** on both Train.csv and
Test.csv — the elaborate rule was implicitly keying off the same
timestamp discontinuity, not an independent sensor signature (Section 3).

### 2.2 Step 2 — determining Open vs. Close

Read directly off `Close command` (`rules.get_operation`) — not predicted,
since it's a perfect indicator already present in the data (matches the
per-operation row counts exactly: 10,225 Close-cycle rows vs. 7,811
Open-cycle rows in Train.csv). This isn't a modeling decision; it exists
only because the classification rule below branches on it.

### 2.3 Step 3 — classification: Normal vs. Abnormal resistance

This is the one place the algorithm makes a real decision, and it branches
by operation because the underlying physical signature is different in
each direction:

**Close cycles** — decided by **peak current** (`cur_max`):

```
Abnormal resistance if cur_max < 2060 mA, else Normal
```

In training data this threshold sits in a real gap with **zero overlap**:
Normal cycles ranged 2078–2266 mA, Abnormal cycles 1993–2046 mA. Abnormal
Close cycles show a *lower* peak current, not higher — this controller's
stall/torque-limit logic caps current lower once it senses resistance
while closing, which is the opposite of the naive intuition that a
struggling motor should draw more current.

**Open cycles** — peak current does **not** separate the classes (both
cluster 2490–2560 mA regardless of label), so the rule instead uses
**mean current over the whole cycle**:

```
Abnormal resistance if cur_mean > 700 mA, else Normal
```

Also a real, zero-overlap gap in training: Normal 630–666 mA, Abnormal
738–973 mA. Abnormal Open cycles draw sustained excess current, most
pronounced in the first third of the stroke (722–758 mA Normal vs.
941–1342 mA Abnormal) — physically, an Open stroke has to keep pushing
against the obstruction for longer, where a Close stroke's controller cuts
power short instead.

Both thresholds (`CLOSE_CUR_MAX_THRESHOLD = 2060.0`,
`OPEN_CUR_MEAN_THRESHOLD = 700.0`) sit at the midpoint of their respective
training gap. They are the only two numbers "learned" from data in this
whole algorithm — there is no model fit, no cross-validation split, and no
tunable hyperparameter beyond these two thresholds and the guardrail
row-count bounds above.

### 2.4 No ranking, no tiebreakers

Unlike ACV (which ranks 8 candidate cars against each other within one
file), Door assigns exactly one label to each independently-detected
segment — there is nothing to rank and no tie to break between segments.
The nearest equivalent design question is the Open/Close branch above,
which is a deterministic lookup (Section 2.2), not a contest between
competing evidence.

## 3. What was tried and did not work well

**An elaborate position/current segmentation rule** (direction reversal
>200 position units; current dropping from >1500mA to <200mA, combined
with command-change or position-reset logic) was the original approach,
and it does recover all 110 training boundaries with zero false positives.
But cross-checking it against the trivial timestamp-gap rule on both
Train.csv and Test.csv showed the two produce **byte-identical
boundaries** in every case — the elaborate rule was never actually finding
cycle edges from sensor *behaviour*, it was coincidentally firing at the
same place a simple time discontinuity already marks, because of how this
specific dataset was constructed (cycles concatenated with idle time
removed, not a genuinely continuous stream). Kept the gap rule instead:
it's simpler, it's what's actually doing the work, and it doesn't carry a
false impression that the model detects boundaries from motor behaviour
when it doesn't need to.

**Validating segmentation by matching only start-index sets against the
answer file** (the original approach in the v1 ML pipeline) reports "110/110
correct" but never checks end times, never computes IoU, and never
implements the real scoring metric at all — replaced by
`diagnostics.iou_weighted_f1()`, the actual Door info kit Section 4
formula, run against the pipeline's own detected segments rather than
against features pre-computed from the answer file's boundaries (see
`self_check_on_train`'s docstring: "deliberately not the same as
re-deriving features from the answer file's own start/end times").

**A random forest / gradient boosting classifier on a 24-feature vector**
(the v1 ML pipeline, `backend_initial/Door/v1_testing/`) also reaches a
perfect training score, but it carries costs the two-threshold rule
doesn't: `GradientBoostingClassifier` hard-crashes on a segment whose
resistance-proxy feature is all-NaN (confirmed by direct test — an
all-low-current segment produces `ValueError: Input X contains NaN` for
that method specifically, though not for the Random Forest method), a
failure mode that never surfaces in training data but is a real risk on
an unseen segment; and a fitted model is materially harder to justify in
a write-up than "peak current below X, or mean current above Y" — which
also happens to line up with an intuitive physical story (stall/torque
limiting on Close, sustained resistance on Open) that a black-box
classifier's decision boundary would not make visible without extra
explainability work. The rule-based version was kept as the primary
approach for this reason, not because the ML version scores worse.

**Trusting the classification score without checking it against the
pipeline's own segmentation output.** The v1 pipeline computed training
accuracy from features re-derived at the answer file's exact boundaries,
which is a different (and easier) question than "how does the full
pipeline do end to end" — the two only coincide here because segmentation
happens to be perfect on this file. `diagnostics.self_check_on_train()`
fixes this by running `segmentation.py` and `rules.py` on the raw stream,
the same code path `run_pipeline.py` uses for Test.csv, so the reported
number can't silently diverge from what the pipeline actually does.

**Trusting a Test.csv prediction without checking it against the range
Train actually demonstrated.** Nothing in the original approach compared
Test predictions against training's feature ranges before writing them out.
`diagnostics.flag_out_of_range()` closes this gap, and it isn't
hypothetical: on the real Test.csv, it flags 7 of 38 predicted segments,
including one (`2023-7-5-0-20-55-731` to `...20-59-411`) whose door
travels to position 807 — beyond the 695–705 ceiling every training Open
cycle ever reached — with a current profile that also falls outside
training's range on two other features simultaneously. This is not
treated as an error (there is no ground truth to say the prediction is
wrong), but it is surfaced rather than silently trusted, since it's an
extrapolation rather than an interpolation.

## Verification note

Re-run while writing this document, against the real dataset:

```
python run_pipeline.py --data-dir <PS3>/02_Datasets/Door --output door_predictions.csv
```

Self-check (Step 1) result: **110/110 segments matched, IoU-weighted F1 =
1.0000, classification accuracy on matched segments = 1.0000, 0 implausible
candidates discarded.** Test.csv (Step 3): 38 cycles detected (20 Close /
18 Open), 29 Normal / 9 Abnormal resistance. Out-of-range check (Step 4):
7/38 segments flagged, matching the specific segments and feature values
described above. Output CSV columns and format verified against
`04_Example_Submission/door_predictions.csv`.
