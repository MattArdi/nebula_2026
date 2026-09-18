# ACV Subsystem — Rule-Based Ranking Algorithm

This document describes the fault-localisation algorithm in `schema.py` /
`ranking.py` / `diagnostics.py` / `run_pipeline.py`: what data it needs, how
the ranking is actually computed (including tiebreakers), and what
alternatives were tried and rejected along the way. All numbers below are
from re-running the pipeline against the real dataset while writing this
document — see the verification note at the end.

## 1. Required data input and format

### Directory layout

```
<data-dir>/
├── Train/
│   ├── acv_case_01.xlsx .. acv_case_06.xlsx
├── Train_Labels.csv          # columns: filename, faulty_car
└── Test/
    └── acv_test_case.xlsx
```

### Per-file schema

Every case file is one `.xlsx` with one row per timestamp (sampled every
30s) and a block of columns per car, named `Car <NN> - <parameter>`, where
`<NN>` is the two-digit car identifier **exactly as it appears in that
file's own headers** — this is also the identifier the submission's
`ranked_cars` field must use. The pipeline never assumes a fixed column
list; it parses each file's own headers at load time (`schema.py`).

Two schema variants exist in the real data, and the pipeline handles both:

| | Standard schema | Rich schema |
|---|---|---|
| Files | `case_01/02/03/05/06`, `acv_test_case.xlsx` | `case_04` only |
| Columns | 67 (8 params × 8 cars + 3 id columns) | 483 (60+ params × 8 cars) |
| Indoor reading used | `Indoor Average Temperature` | `Passenger Cabin Temperature Detected Value` |
| Target reading used | `ACV Control Temperature (Cooling)` | `Target Temperature Value` |

The pipeline tries the standard pair of column names first, falls back to
the rich pair, and if neither is present *or present but 100% empty*, marks
that car as having no usable data rather than fabricating a value. This is
not a hypothetical: cars 05–08 are **100% NaN across every one of ~60
parameters** in `acv_case_04.xlsx` — confirmed directly, not inferred.
Cars with no data are always ranked last, in car-identifier order, and are
excluded from the CUSUM/tiebreak computation entirely (see Section 2).

### Labels file

`Train_Labels.csv`: `filename,faulty_car`, one row per training case, e.g.
`acv_case_01.xlsx,01`. Used only by `diagnostics.self_check_on_train()`.

### Output format

`run_pipeline.py` writes one row per input file:

```
file_id,ranked_cars
acv_test_case.xlsx,01|08|04|03|02|07|05|06
```

`ranked_cars` lists every car in the file, most- to least-likely faulty,
pipe-separated, using each car's own header identifier — verified
byte-for-byte against `04_Example_Submission/acv_predictions.csv`'s
column names and separator.

## 2. How the algorithm works

### 2.1 The core idea: compare each car to its 7 siblings, not to a fixed number

A refrigerant leak degrades a car's cooling capacity, so that car can't
hold its target temperature as tightly as its healthy siblings. The
absolute size of that effect varies enormously file to file (confirmed:
real training deviations range from 0.67°C in one file to 2.69°C in
another, and in one file even the *healthy* cars deviate by over 1°C) — so
there is no fixed absolute threshold that works across files. Every
comparison in this pipeline is **relative, within one file, at one moment
in time**: a car is judged against the other cars in the *same file*, not
against a number carried over from training.

### 2.2 Step 1 — per-timestep gap from the fleet

For each car, at every timestep:

```
gap = (car's indoor − target) − median across all cars' (indoor − target) at that same timestep
```

Comparing against the cross-car median at the *same moment* cancels shared
environmental noise (outdoor temperature swings hit every car at once) —
this is a stronger baseline than comparing a car to its own fixed setpoint,
because in 2 of 5 training files the faulty car's own configured setpoint
differs slightly from its siblings', which would bias a same-car-only
comparison.

### 2.3 Step 2 — local standardisation ("small region" normalisation)

The raw gap is divided by a local spread estimate: at each timestep, pool
`|gap|` across all 8 cars over a **trailing 1-hour window** (120 rows at
30s sampling), take the median of that pool. This is local in time
(a 1-hour window, not the whole file and not a fixed constant carried
across files) and cross-sectional (drawn from all 8 cars, not one car's own
history), so a file whose cars are all inherently noisier (confirmed:
`acv_case_06`'s healthy cars deviate over 1°C from each other) doesn't get
judged by the same absolute yardstick as a quieter file.

Two alternatives were tried here and dropped:

- **A single global constant `σ`** (estimated from pooled healthy-car data
  across all 5 standard-schema training files) instead of a per-file/local
  one. Confirmed to still work for *ranking* but not for anything absolute
  (see Section 3).
- **A per-file constant `σ`**, estimated once from that file's own pooled
  data via median absolute deviation. Confirmed broken by data
  quantisation: it evaluated to exactly **0** for `acv_case_05.xlsx` and
  `acv_case_06.xlsx`, because more than half of the pooled residuals landed
  on the exact same discretised value. The local, rolling-window version
  used in the shipped code doesn't have this failure mode.

### 2.4 Step 3 — decayed CUSUM

The standardised gap series `z` is fed into a one-sided CUSUM with
exponential forgetting:

```
c_t = max(0, decay · c_{t-1} + (z_t − k))
```

- **`k = 0.5`** (in local-σ units) — the standard SPC slack: ordinary
  jitter smaller than this is absorbed every step and never accumulates.
- **`decay`** corresponds to a **4-hour half-life** (`decay = 0.5^(1/480)`
  at 30s sampling). Chosen from the observed physical timescale of a real
  onset, not an arbitrary round number: the two training files where the
  fault visibly emerges partway through the recording ramped from baseline
  to peak over roughly 4–8 hours (measured directly from the hourly
  deviation series), so a memory length on that same order gave the best
  worst-case margin in an empirical sweep from 30 minutes to infinite
  memory (Section 3 has the numbers).
- The car's score is **`c_t` at the end of the file** (the current decayed
  value), not the highest value it ever reached — using the all-time peak
  was tested and found to let old, no-longer-relevant noise from one car
  permanently compete against another car's real, recent signal (Section
  3).

The car with the highest score is the primary suspect.

### 2.5 Step 4 — tiebreaker (secondary signal), and why it's a tiebreak and not a vote

A second, independent signal is computed from a different data source: the
number of transitions in `ACV Running Mode` (control-mode switching, not
temperature) over the file. This exists to hedge against the primary
signal's occasional weak margin — but **it is only ever consulted when two
cars' primary CUSUM scores are within 10% of each other** (`TIE_EPSILON =
0.10` in `ranking.py`). Outside of a near-tie, the primary score decides
the order outright.

This design was arrived at the hard way. An earlier version averaged each
signal's own within-file rank unconditionally (a Borda average) for every
car, not just close calls. That broke on real data: in `acv_case_06`, the
primary signal was completely decisive (top score 1543.8 vs. a runner-up
of 8.4 — a ~180× margin), but the mode-transition counts across all 8 cars
were 290–293 — essentially flat noise with a total spread of 3 — and the
true faulty car happened to sit at the bottom of that noise purely by
chance. The unconditional rank-average let 3 transitions' worth of noise
overrule a 180× primary margin, dropping the true car from rank 1 to rank
6 (self-check score 1.000 → 0.375). Restricting the secondary signal to
genuine near-ties (checked against the smallest real margin actually
observed across the 6 training files, 19.0% in `acv_case_02.xlsx` — the
10% threshold sits comfortably below every real result, so it never fires
on a file where the primary signal is already clear) fixed this without
losing anything: all 6 training cases return to a perfect rank-decay score
of 1.0000.

**Full tiebreak order**, applied only within a near-tie cluster:
1. Higher primary (CUSUM) score.
2. Higher secondary (mode-transition count) score.
3. Lower car identifier (numeric) — a final, fully deterministic fallback
   so the output never depends on incidental sort order.

Cars with no usable data (Section 1) are appended after all data-bearing
cars, sorted by car identifier, regardless of any of the above.

## 3. What was tried and did not work well

This section exists so the reasoning behind the shipped design doesn't
have to be re-derived later. Every entry below was implemented and tested
against the real training data, not reasoned about abstractly.

**Fixed-window features that don't adapt to both fault regimes.** Whole-
file mean deviation and last-quartile ("trailing window") mean deviation
each get every training file's ranking right, but each has a worst case
the other doesn't: whole-file mean is weakest on the one file where the
fault only emerges in the final day (margin as low as 0.018°C in an
earlier, un-normalised version of the feature); last-quartile is weakest
on a file where the fault is present from the very start, because
discarding the first 75% of the file throws away real confirming evidence
(margin as low as 0.044°C). CUSUM was adopted specifically because it
doesn't require committing to one window length in advance.

**Rolling-window maximum.** Smoothing the gap with a rolling mean and
taking the single highest value reached over the file is exactly as
sensitive to one healthy car's transient noise spike as it is to the real
fault — confirmed directly: in `acv_case_06`, a healthy car's rolling-max
value (2.31) came within 0.21 of the true faulty car's (2.53).

**A too-short CUSUM half-life.** A 30-minute half-life effectively
degenerates into the same fragility as the rolling-max method above (short
memory ≈ smoothed peak-picking) — its worst-case margin across the 5
training files (21.0%) was the tightest of every half-life tested (30 min
through infinite memory).

**Infinite-memory CUSUM, read as the all-time peak.** With no decay, the
statistic tracked was "the highest this car's cumulative sum has ever
been" — which means a healthy car's one-off noise spike on day one stays
in permanent contention against another car's real signal on day four,
because a peak never fades. This gave the *worst* margin of any half-life
tested (9.3%, worse than 30 minutes).

**Trend slope of the deviation over time.** A reasonable-sounding
generalisation of "the fault gets worse over time," but it only holds in
2 of 5 training files; in the other 3 the fault presents as a stable
elevated plateau from early on, and in one of those the slope is actually
slightly *negative* for the true faulty car. Mean rank-decay score across
the 5 files: 0.675 (vs. 1.000 for CUSUM).

**Correlation between a car's deviation and outdoor temperature.** Both
weak (mean score 0.525) and severely limited by data coverage — the
outdoor-temperature sensor reads the literal string `"Invalid"` for 6 of 8
cars in 2 of the 5 training files, so the feature is undefined for most
cars in nearly half the training data.

**An unconditional multi-signal ensemble (Borda rank average).** Described
above in Section 2.5 — broke a previously-perfect result on real data by
letting a near-flat secondary signal outvote a decisive primary one.
Replaced with a tiebreak that only activates on genuine near-ties.

**An absolute "is there a fault at all" gate**, tried two ways:

- *Global σ, pooled from all known-healthy cars across the 5 training
  files*, used to set a CUSUM slack `k` and looked for a value of `k` that
  separates real faults from a synthetic "no fault anywhere" scenario
  (built by replacing the true faulty car's data with a genuinely healthy
  car's, and repeating across all 7 possible replacements × all 5 files —
  35 synthetic all-healthy draws). No value of `k` from 0 to 0.5σ produced
  a clean separation; the weakest real fault (`acv_case_03`, values from
  26 to 155 depending on `k`) always overlapped the largest synthetic
  no-fault draw (581–688).
- *Local, per-file cross-sectional standardisation* (the same normalisation
  used in the shipped ranking feature, Section 2.3), applied to the same
  gating question. Still overlapped — in fact the largest no-fault draw
  rose to 1495.5, worse than the global-σ attempt.

Both failures trace to the same root cause, confirmed by inspection: the
largest "no-fault" outlier in every version came from `acv_case_01`, and
specifically from car 02 — which has a real, sustained, non-random
elevation in that file (visible even in the very first exploratory pass,
0.198°C above setpoint, the second-highest car in the file) that simply
isn't caused by a refrigerant leak. CUSUM is built to detect sustained
drift, and it cannot distinguish "sustained because of a leak" from
"sustained for some unrelated operational reason" using only a temperature
signal — no amount of threshold-tuning on this one feature fixes that,
because the two cases aren't actually separated in the feature itself.
**Consequence for this pipeline: it ranks; it does not gate.** It always
names a most-suspicious car and never claims "no fault exists." This is
consistent with the competition's own stated data generation process
(`ACV_Subsystem_Info_Kit.md`: "exactly one car... has a refrigerant
leakage fault" is guaranteed for both Train and Test), so it is not a gap
against what's actually being graded — it would only matter for a
deployment where "no fault" is a real possible outcome, which this
dataset's stated design rules out.

**Double-fault and zero-fault robustness (informational, not gated on).**
Synthetic tests — injecting a second real fault pattern onto an already-
healthy car, and neutralising the only real fault to simulate an all-
healthy fleet — showed the ranking degrades gracefully rather than
breaking: a genuine double fault put both problem cars in the top 2,
clearly separated from every healthy car (1843 and 1749 vs. a best-healthy
value of 567); an all-healthy fleet still produces a "top suspect" (as it
must, since this is a pure ranking method — see the gating discussion
above), with a magnitude that would not have been distinguishable from a
mild real fault by any threshold tested.

## Verification note

Re-run while writing this document, against the real dataset:

```
python run_pipeline.py --data-dir <PS3>/02_Datasets/ACV
```

Self-check (Step 1) result: **all 6 training cases at rank 1, mean
rank-decay score 1.0000** (`acv_case_04.xlsx`'s 4 no-data cars correctly
excluded from the ranking computation and appended last). Test-case
(Step 2) ranking for `acv_test_case.xlsx`: `01|08|04|03|02|07|05|06`, with
a margin (225.8) inside the range of margins observed across training, so
no confidence flag was raised in Step 3. Output CSV columns and separator
verified against `04_Example_Submission/acv_predictions.csv`.
