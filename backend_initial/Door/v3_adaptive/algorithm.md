# Door Subsystem — v3, Adaptive-Threshold Segmentation & Classification

v3 keeps v2's segmentation entirely unchanged (see `v2_rule-based/algorithm.md` for that
derivation — issue 3 in the info kit, finding cycle boundaries in a continuous stream, was
already solved there and isn't touched here). What v3 adds is aimed at the two info-kit issues
v2 left unaddressed:

1. *"Statistical thresholds set from scarce fault samples are hard to evaluate for
   reasonableness."*
2. *"Data distributions differ among doors; a uniform threshold causes false alarms and missed
   detections."*

## 1. Required data input and format

### Input

Identical to v2: a sensor CSV, one header row, one row per 20ms sample (50 Hz), 17 columns. The
algorithm reads `Motor current(mA)`, `Door leaf position`, and `Close command`.

### Labels (for recalibration only, not needed to run predictions)

`Train.csv` + `Train_Segments_Answer.csv`, same format as v2. The calibrated threshold state is
already shipped in `artifacts/calibration.json` and `artifacts/train_ranges.json`, so cloning
this repo and running `run_pipeline.py --input <file>` needs neither of those files.

### Output

Same schema as v2 — one row per predicted segment, `start_time,end_time,prediction`. No `file_id`,
no `operation` column.

## 2. How this addresses the info kit's 3 pain points

| # | Pain point | Status | Mechanism |
|---|---|---|---|
| 1 | *"Statistical thresholds set from scarce fault samples are hard to evaluate for reasonableness."* | **Addressed in v3** | Bootstrap confidence half-width per threshold (Section 3, "Confidence"), flagging any prediction close enough to the boundary that a different resample of the same 15/40 training examples could plausibly have called it the other way. |
| 2 | *"Data distributions differ among doors; a uniform threshold causes false alarms and missed detections."* | **Addressed in v3** | Each threshold is expressed relative to a recalibratable local baseline (Section 3, "Classification"), not a bare constant — `--retrain` against a different door's own data shifts the boundary by that door's own offset. |
| 3 | *"A live deployment must find cycle boundaries in a continuous stream before it can classify them."* | **Already solved in v2**, unchanged here | Timestamp-gap segmentation — see `v2_rule-based/algorithm.md`. Not touched in v3: it was already validated (perfect IoU-weighted F1 on Train, exact segment count on the real `Test.csv`), and neither pain point above bears on segmentation. |

Issues 1 and 2 are the ones this version was built for; issue 3 is listed for completeness, not
because anything here changes it.

One explicit non-goal, worth stating plainly: none of this corrects any individual prediction
v2 already made, including whichever one caused the real `Test.csv` score to come back 37/38
instead of 38/38. v3 is verified to reproduce v2's predictions byte-for-byte on that file (Section
3). A separate attempt to specifically correct that one prediction was built, tested against the
real data, and found to trade one likely fix for several new, worse errors — see the project
history for that experiment's findings; it was not kept, since it made things worse, not better.

## 3. How the algorithm works

### Segmentation and feature extraction — unchanged from v2

Timestamp-gap cycle detection, then the same six features per cycle (`cur_max`, `cur_mean`,
`cur_early_mean`, `pos_min`, `pos_max`, `n_rows`).

### Classification — a threshold relative to a calibrated local baseline, not a bare constant

v2 decided Close cycles by `cur_max < 2060` and Open cycles by `cur_mean > 700`, both fixed
constants. v3 replaces each fixed constant with:

```
adjusted_threshold = fixed_threshold + (local_baseline − train_baseline)
```

where `train_baseline` is the median of that operation's Normal cycles in Train (specifically,
the median of the *same* 20-cycle window used to seed `local_baseline` below — see Section 4 for
why they have to be the same window), and `local_baseline` is the median of the most recent 20
cycles this pipeline has classified Normal for that operation, seeded from Train's own history at
calibration time.

**Within a single run, `local_baseline` is frozen at its calibrated seed value** — it is not
updated cycle-by-cycle as the pipeline classifies its own input (Section 4 explains why that was
tried and rejected). It only changes when the pipeline is explicitly recalibrated (`--retrain
--data-dir <a door's own labelled data>`), which reseeds it from that door's own recent Normal
history.

This directly answers issue 2: whenever a door's own recent Normal cycles sit at a different
current level than Train's door did, recalibrating there shifts the threshold by exactly that
difference, rather than checking every door against Train's specific current level forever. And
because `local_baseline` is defined to equal `train_baseline` exactly at calibration time (by
construction — see Section 4), `adjusted_threshold` reduces to precisely v2's fixed constant on
this dataset, since Train's own door shows no measurable drift. **Verified**: v3 reproduces v2's
predictions byte-for-byte on the real `Test.csv`.

### Confidence — a bootstrap half-width on the threshold itself

Directly answering issue 1: at calibration time, the 40 Normal / 15 Abnormal training examples
per operation are bootstrap-resampled (2000 resamples, with replacement), and the same
gap-midpoint derivation v2 used to pick 2060 / 700 in the first place is recomputed on each
resample. The 5th–95th percentile of that distribution gives a `ci_half_width` per operation —
how far the threshold could plausibly have landed given only this many examples, not just where
it happens to sit.

A prediction is flagged `low_confidence` when its decision value falls within `ci_half_width` of
the threshold it was judged against — a different, equally legitimate resample of the same
training data could plausibly have called it the other way. This is a materially different (and
weaker) kind of uncertainty than v2's out-of-range flag, which is kept unchanged alongside it: OOD
means "nothing like this was ever seen in training"; low-confidence means "this was seen, but the
boundary near it is itself shaky." A cycle can be flagged by either, both, or neither.

## 4. What was tried and did not work well

**Online self-training within a single run** — updating `local_baseline`'s rolling window live,
using every cycle the pipeline itself just classified Normal, as the stream was processed — was
the first version built. It seemed like the more literally "adaptive" design. Tested against the
real `Test.csv`: the Open baseline drifted from 646.23 to 659.25 mA over the course of the stream
(pulled upward by that stream's own naturally slightly-higher Normal cycles), shifting the
adjusted threshold from 700.0 to ~712.0 mA and flipping one borderline segment
(`cur_mean = 707.79`, previously correctly Abnormal under v2's fixed 700 mA line) to Normal — with
no accuracy benefit to show for it, since that segment was already correctly classified by v2.
Self-training on a model's own predictions within one inference pass is a known way to compound
whatever the model is already slightly wrong about, and this was a concrete instance of exactly
that, not just a theoretical concern. Replaced with the frozen-per-run design in Section 3:
`local_baseline` only moves between runs, via an explicit `--retrain` a person chooses to trigger
against a specific door's own data, never silently during inference.

**Defining `train_baseline` as the median of *all* 40 Normal training examples**, separately from
the 20-cycle `seed_window` used to initialize `local_baseline`, was the first calibration
approach. The two medians aren't guaranteed to match (median of 40 examples vs. median of their
most recent 20 isn't the same statistic), so even with the online-update problem above fixed, a
freshly-calibrated run still started with a small nonzero offset baked in — Close opened at
2064 mA and Open at 701.71 mA instead of exactly 2060 / 700, a difference small enough to not
change any Test.csv label in this particular run, but a real, silent departure from v2's exact
behavior that had no justification (it wasn't measuring anything about door drift, just an
arithmetic mismatch between two differently-sized windows). Fixed by defining `train_baseline` as
the median of the seed window itself, guaranteeing `local_baseline == train_baseline` and
therefore `adjusted_threshold == fixed_threshold` exactly at calibration time, by construction.

**A window size for `local_baseline`** other than 20 wasn't swept exhaustively — 20 was chosen as
half of the 40 Normal examples available per operation in Train, balancing a robust-enough median
against leaving room for the window to actually move if recalibrated against a door with a
genuinely different baseline. Given Train's own door shows no measurable drift across its own
recording (Close Normal `cur_max`: 2149.4 mean in the first 10 cycles vs. 2147.4 in the last 10;
Open Normal `cur_mean`: 644.0 vs. 648.8 — see `v2_rule-based/algorithm.md`), there is no drift
signal in the data available to tune this against; 20 is a reasonable default, not a value this
dataset can validate as optimal.
