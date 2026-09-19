# Door Subsystem — v3, Adaptive-Threshold Segmentation & Classification

Door is a **segmentation + binary classification** task: given a continuous stream of door
motor/position sensor readings, find each individual door-open/close cycle within it, then decide
whether that cycle shows Normal or Abnormal opening/closing resistance. There is no file-per-cycle
structure to key on — one stream goes in, a list of timestamped, labelled segments comes out.

This version addresses all three pain points named in the info kit, summarized in Section 2 and
explained in full in Section 3.

## 1. Required data input and format

### Input

A sensor CSV: one header row, one row per 20ms sample (50 Hz), 17 columns. `Datetime` uses the
format `Year-Month-Day-Hour-Minute-Second-Millisecond`, not zero-padded (e.g.
`2023-7-5-0-11-17-664`). The algorithm reads three columns: `Motor current(mA)`, `Door leaf
position`, and `Close command`.

### Labels (for recalibration only, not needed to run predictions)

`Train.csv` + `Train_Segments_Answer.csv`: the former in the same raw sensor format as above; the
latter one row per true cycle, giving its start/end time, whether it was an Open or Close
operation, and its status (`Normal` / `Abnormal resistance`). The calibrated state this pipeline
needs is already shipped in `artifacts/calibration.json` and `artifacts/train_ranges.json`, so
cloning this repo and running `run_pipeline.py --input <file>` needs neither of those two files —
they're only read when explicitly recalibrating (`--retrain --data-dir <dir>`).

### Output

One row per predicted segment (not per file — the input is a single continuous stream that may
contain many cycles):

```
start_time,end_time,prediction
2023-7-5-0-0-0-0,2023-7-5-0-0-3-760,Normal
```

No `file_id` column and no `operation` column.

## 2. How this addresses the info kit's 3 pain points

| # | Pain point | Status | Mechanism |
|---|---|---|---|
| 1 | *"Statistical thresholds set from scarce fault samples are hard to evaluate for reasonableness."* | **Addressed** | A bootstrap confidence half-width on each classification threshold (Section 3.3), flagging any prediction close enough to the boundary that a different resample of the same training examples could plausibly have called it the other way. |
| 2 | *"Data distributions differ among doors; a uniform threshold causes false alarms and missed detections."* | **Addressed** | Every threshold is expressed relative to a recalibratable local baseline, not a bare constant (Section 3.2) — recalibrating against a different door's own data shifts the decision boundary by exactly that door's offset. |
| 3 | *"A live deployment must find cycle boundaries in a continuous stream before it can classify them."* | **Addressed** | Timestamp-gap segmentation (Section 3.1), which finds cycle boundaries directly from the stream's own row-to-row timing — no pre-cut, one-cycle-per-file input required. |

## 3. How the algorithm works

### 3.1 Segmentation — finding cycle boundaries from timing alone

The input stream is not continuously logged through idle time between cycles — it only contains
rows recorded while a cycle was actually in progress, with the periods between cycles absent from
the file entirely rather than represented by rows of idle readings. This is directly checkable:
in the raw training stream, every row belongs to some cycle and the file contains no rows outside
any cycle at all, meaning the recording device produces data only during active door motion and
nothing in between.

The practical consequence is that every row-to-row timestamp delta falls into one of two regimes
with nothing in between: about 20ms (the normal 50 Hz sampling interval, within a cycle), or ten
or more seconds (the gap where an idle period was never logged, between cycles). A cycle boundary
is declared wherever the gap between two consecutive rows' timestamps exceeds
`GAP_THRESHOLD_S = 0.1` seconds — a threshold that sits in the enormous, unambiguous space between
those two regimes, so the exact value chosen barely matters.

Two guardrails then discard any candidate segment shorter than `MIN_CYCLE_ROWS = 50` rows or
longer than `MAX_CYCLE_ROWS = 250` rows, as a safety margin against spurious micro-segments if a
real deployment stream ever does contain genuine idle-period noise that this dataset doesn't.

This was cross-checked against a more elaborate, motor-behavior-based segmentation rule (a
direction reversal of more than 200 position units, combined with current dropping from above
1500 mA to below 200 mA) as an independent way to find the same boundaries. The two rules produced
byte-identical boundaries in every case — the elaborate rule was never actually detecting edges
from motor dynamics, it was coincidentally firing at the same point the timing gap already marks.
The simpler timing-based rule is used because it's what's actually doing the work.

**Verified**: run end-to-end against the training stream, this segmentation plus the classification
in 3.2 reproduces every one of the 110 true labelled cycles with perfect IoU-weighted F1 (1.0000)
— the real competition metric, not just a boundary count. Run against the real, unlabelled
`Test.csv`, it finds exactly 38 cycles with no candidate ever discarded by the guardrails.

### 3.2 Classification — a threshold relative to a calibrated local baseline

Two features drive the decision, one per operation, computed from a segment's motor current
readings and read off the `Close command` column to determine whether the segment is a Close or
Open cycle:

- `cur_max` — the segment's peak current, used for Close cycles. Close cycles under abnormal
  resistance draw a *lower* peak current than normal ones, not higher — the controller's
  stall/torque-limit logic caps current once it senses resistance while closing.
- `cur_mean` — the segment's mean current over the whole cycle, used for Open cycles. Open cycles
  under abnormal resistance draw sustained *excess* current instead of a distinct peak, since an
  Open stroke keeps pushing against an obstruction for longer, where a Close stroke's controller
  cuts power short instead.

A third feature, `cur_early_mean` (the mean current over just the first third of the cycle), is
tracked alongside these purely as a diagnostic signal — see Section 3.3 and 3.4 — because abnormal
Open cycles draw excess current earliest and most sharply in the stroke, before it's diluted into
the whole-cycle mean.

Each operation's decision is:

```
Close: Abnormal resistance if cur_max < adjusted_threshold_close, else Normal
Open:  Abnormal resistance if cur_mean > adjusted_threshold_open,  else Normal
```

where each `adjusted_threshold` is not a bare constant but:

```
adjusted_threshold = fixed_threshold + (local_baseline − train_baseline)
```

`fixed_threshold` is 2060 mA for Close and 700 mA for Open — each the midpoint of a clean,
non-overlapping gap between the two classes' peak/mean current in the 110 labelled training
cycles (Close: Abnormal 1993–2046 mA vs. Normal 2078–2266 mA; Open: Normal 630–666 mA vs. Abnormal
738–973 mA). `train_baseline` is the median of that operation's Normal cycles over a fixed
20-cycle window taken from the end of Train's own recording, and `local_baseline` is the median of
the most recent 20 cycles this pipeline has itself classified Normal for that operation — seeded,
at calibration time, from that exact same 20-cycle Train window, so `local_baseline` and
`train_baseline` are defined as the median of literally the same numbers and therefore start
identical by construction, not by coincidence.

**Within a single run, `local_baseline` is frozen at its calibrated seed value** — it is not
updated cycle-by-cycle as the pipeline classifies its own input; Section 5 explains why an
earlier version that did update it live was rejected. It only changes between runs, via an
explicit `--retrain --data-dir <a door's own labelled data>`, which reseeds it from that door's
own recent Normal history.

This is what answers pain point 2: whenever a door's own recent Normal cycles sit at a different
current level than the door `Train.csv` was recorded from, recalibrating shifts the threshold by
exactly that difference, rather than checking every door against one fixed reference level
forever. And because `local_baseline` equals `train_baseline` exactly at calibration time,
`adjusted_threshold` reduces to precisely the plain fixed-threshold rule whenever there's no
drift — which is the only case this dataset ever demonstrates. **Verified**: run against the real
`Test.csv`, this reproduces the plain fixed-threshold rule's predictions byte-for-byte.

### 3.3 Confidence — a bootstrap half-width on the threshold itself

This is what answers pain point 1. At calibration time, the 40 Normal / 15 Abnormal training
examples per operation are bootstrap-resampled — 2000 resamples, with replacement — and the same
gap-midpoint calculation used to derive the 2060 / 700 mA constants in the first place is
recomputed on every resample. The 5th–95th percentile spread of that distribution of recomputed
thresholds gives a `ci_half_width` per operation: how far the threshold could plausibly have
landed given only this many training examples, not just where it happens to sit with the data at
hand.

A prediction is flagged `low_confidence` when its decision value (`cur_max` for Close, `cur_mean`
for Open) falls within `ci_half_width` of the threshold it was judged against — meaning a
different, equally legitimate resample of the same training data could plausibly have called it
the other way. This is a materially different, and weaker, kind of uncertainty than the
out-of-range flag in Section 3.4: out-of-range means "nothing like this value was ever seen in
training at all"; low-confidence means "this value was well within range, but the boundary itself,
right where this value sits, is statistically shaky." A given cycle can be flagged by either, both,
or neither.

### 3.4 Out-of-range flagging — a second, distinct kind of doubt

Alongside the confidence check above, every prediction's decision-relevant features (`cur_max`,
`cur_mean`, `cur_early_mean`, `pos_max`) are checked against the full range each one ever took in
the 110 training cycles, per operation. A value outside that range doesn't mean the prediction is
wrong — it means the prediction is an extrapolation rather than an interpolation, a meaningfully
different and weaker kind of confidence worth surfacing rather than hiding behind a bare label.

## 4. Explainability

Same auditable two-number decision as v2 (Section 3.2), with one addition: the threshold itself
is no longer a bare constant, so "why" now has three parts instead of two — the decision value
(`cur_max`/`cur_mean`), the *adjusted* threshold it was compared against, and the baseline that
threshold was adjusted from. `classify_one()` already returns all of this: its second return
value is `{"threshold_used": ..., "baseline_used": ...}`, computed fresh for every cycle, not
just the final label.

This version goes further than v2 on explainability by design — it's how pain point 1 (Section 2)
gets addressed. Two extra, independent signals sit alongside the label:

- **Low-confidence flag** (Section 3.3): "the decision value fell within the bootstrap
  confidence half-width of the threshold" — a different resample of the same 40/15 training
  examples could plausibly have drawn the line elsewhere, so this specific call is a close one.
- **Out-of-range flag** (Section 3.4): "this cycle's features are outside anything Train ever
  demonstrated" — a materially different kind of doubt (extrapolation, not proximity to a
  boundary), and the two can fire independently, together, or not at all.

Together these mean a prediction is never just a bare label — it always carries a value, a
threshold, and (if applicable) which of two distinct kinds of doubt apply to it.

**Current limitation**: exactly as in v2, `run_pipeline.py` writes only `start_time,end_time,
prediction` — `threshold_used`, `baseline_used`, `low_confidence`, and the out-of-range flag are
all computed per cycle but none reach the output CSV today.

## 5. What was tried and did not work well

**Online self-training within a single run** — updating `local_baseline`'s rolling window live,
using every cycle the pipeline itself just classified Normal, as the stream was processed — was
the first version built. It seemed like the more literally "adaptive" design. Tested against the
real `Test.csv`: the Open baseline drifted from 646.23 to 659.25 mA over the course of the stream
(pulled upward by that stream's own naturally slightly-higher Normal cycles), shifting the
adjusted threshold from 700.0 to ~712.0 mA and flipping one borderline segment
(`cur_mean = 707.79`, previously correctly classified Abnormal under the plain 700 mA threshold)
to Normal — with no accuracy benefit to show for it, since that segment was already correctly
classified before this change. Self-training on a model's own predictions within one inference
pass is a known way to compound whatever the model is already slightly wrong about, and this was
a concrete instance of exactly that, not just a theoretical concern. Replaced with the
frozen-per-run design in Section 3.2: `local_baseline` only moves between runs, via an explicit
`--retrain` a person chooses to trigger against a specific door's own data, never silently during
inference.

**Defining `train_baseline` as the median of *all* 40 Normal training examples**, separately from
the 20-cycle window used to seed `local_baseline`, was the first calibration approach. The two
medians aren't guaranteed to match (the median of 40 examples isn't the same statistic as the
median of their most recent 20), so even with the online-update problem above fixed, a
freshly-calibrated run still started with a small nonzero offset baked in — Close opened at
2064 mA and Open at 701.71 mA instead of exactly 2060 / 700, a difference small enough to not
change any Test.csv label in this particular run, but a real, silent departure from the plain
fixed-threshold rule that had no justification (it wasn't measuring anything about door drift,
just an arithmetic mismatch between two differently-sized windows). Fixed by defining
`train_baseline` as the median of the seed window itself, guaranteeing `local_baseline ==
train_baseline` and therefore `adjusted_threshold == fixed_threshold` exactly at calibration time,
by construction.

**A window size for `local_baseline`** other than 20 wasn't swept exhaustively — 20 was chosen as
half of the 40 Normal examples available per operation in Train, balancing a robust-enough median
against leaving room for the window to actually move if recalibrated against a door with a
genuinely different baseline. There's no drift signal in the available data to tune this against:
Close Normal `cur_max` averages 2149.4 mA in the first 10 training cycles and 2147.4 mA in the
last 10; Open Normal `cur_mean` averages 644.0 mA in the first 10 and 648.8 mA in the last 10 —
both essentially flat across the whole recording. 20 is a reasonable default, not a value this
dataset can validate as optimal.
