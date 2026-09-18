# ACV Subsystem — Rule-Based Ranking Algorithm

ACV is a **ranking** task: given one case file's telemetry for 8 cars,
order every car from most- to least-likely to have the refrigerant leak.

## 1. Required data input and format

### Input

One `.xlsx` file, one row per timestamp (sampled every 30s), columns
named `Car <NN> - <parameter>`, where `<NN>` is each car's identifier
exactly as it appears in that file's own headers. Two schema variants
exist, and the algorithm tries the first, falling back to the second:

| | Standard schema | Rich schema |
|---|---|---|
| Indoor reading | `Indoor Average Temperature` | `Passenger Cabin Temperature Detected Value` |
| Target reading | `ACV Control Temperature (Cooling)` | `Target Temperature Value` |

A car with neither pair of columns present, or present but entirely
empty, is treated as having no usable data — it is never assigned a
fabricated value.

### Output

One row per file:

```
file_id,ranked_cars
acv_test_case.xlsx,01|08|04|03|02|07|05|06
```

`ranked_cars` lists every car in the file, most- to least-suspicious,
pipe-separated, using each car's own header identifier. Cars with no
usable data are always placed last.

## 2. How the algorithm works

### Step 1 — per-timestep gap from the fleet

For each car, at every timestamp:

```
gap = (car's indoor − target) − median across all cars' (indoor − target) at that same timestamp
```

Comparing against the cross-car median at the same moment, rather than a
fixed number, cancels out anything that affects every car at once (for
example, ambient temperature swings), since a shared effect shifts every
car's reading together and leaves the relative comparison unaffected.

### Step 2 — local standardisation

The gap is divided by a local spread estimate: at each timestamp, pool
`|gap|` across all 8 cars over a trailing 1-hour window and take the
median of that pool. This keeps the statistic in comparable units
regardless of how much a given file's cars naturally vary from each
other, rather than comparing every file to the same fixed scale.

### Step 3 — decayed CUSUM

The standardised gap series `z` is fed into a one-sided cumulative sum
with exponential forgetting:

```
c_t = max(0, decay · c_{t-1} + (z_t − k))
```

- `k = 0.5` (in local-σ units) — ordinary jitter smaller than this is
  absorbed every step and never accumulates; only sustained excess beyond
  typical noise contributes to the running total.
- `decay` corresponds to a 4-hour half-life. A short memory behaves like
  a smoothed peak-picker and is thrown off by a single noisy moment in an
  otherwise-healthy car; a very long memory lets a stale, no-longer-
  relevant excursion from earlier in the file keep competing indefinitely
  against another car's real, recent signal. A few hours of memory
  balances those two failure modes.
- The car's score is `c_t` at the **end of the file** — the current
  decayed value, not the highest value it ever reached during the
  recording.

The car with the highest score is the primary suspect.

### Step 4 — tiebreak

A second, independent signal — the number of transitions in `ACV Running
Mode` over the file (control-mode switching, a different data source than
temperature) — is consulted **only** when two cars' primary scores are
within 10% of each other (`TIE_EPSILON = 0.10`). Outside of a near-tie,
the primary score decides the order outright.

Full tiebreak order, applied only within a near-tie cluster:
1. Higher primary (CUSUM) score.
2. Higher secondary (mode-transition count) score.
3. Lower car identifier — a final, fully deterministic fallback.

Cars with no usable data are appended after every data-bearing car,
sorted by identifier.

## 3. What was tried and did not work well

**Fixed-window features (whole-file mean deviation, or a fixed trailing
fraction of the file) instead of CUSUM.** Each gets every training
ranking right, but each has a distinct weak spot the other doesn't have:
a whole-file average is diluted when a fault only shows up in the final
portion of a long recording; a fixed trailing window discards real
confirming evidence when the fault is present from the very start
instead. CUSUM was adopted because it doesn't require committing to one
window length in advance — it accumulates evidence from wherever it
actually appears in the file.

**A rolling-window maximum** (smooth the gap with a rolling mean, take
the single highest value reached) is exactly as sensitive to one healthy
car's transient noise spike as it is to a real fault, since a single
outlying moment can produce as large a value as a genuine sustained
excess.

**A CUSUM half-life much shorter than a few hours** degenerates toward
the same fragility as the rolling maximum above — short memory is, in
effect, a smoothed peak-picker. A half-life much longer than that (or no
decay at all, reading the statistic as the all-time peak rather than the
current value) has the opposite problem: an old, unrelated noise spike
from any car stays in permanent contention, because a peak-tracking
statistic never forgets.

**Trend slope of the deviation over time**, as an alternative primary
signal — a reasonable-sounding way to capture "the fault gets worse over
time," but it assumes every fault presents as a monotonically worsening
trend. Some genuinely present as a stable elevated plateau from early in
the recording rather than a continuing climb, and slope handles that case
poorly.

**Correlation between a car's deviation and outdoor temperature**, on the
theory that a car with less cooling headroom should be more exposed to
ambient swings. Weak as a signal, and severely limited by coverage: the
outdoor-temperature sensor is reported as invalid for most cars in a
meaningful fraction of files, so the feature is undefined for most cars
whenever that happens.

**An unconditional multi-signal ensemble** (averaging each signal's own
within-file rank for every car, not just close calls) was tried before
the near-tie-only tiebreak described in Step 4. It broke on real data: in
one file the primary CUSUM signal was completely decisive (a roughly
180× margin over the runner-up), but the mode-transition counts across
all 8 cars were nearly flat noise, and the true car happened to sit at
the bottom of that noise purely by chance. The unconditional rank-average
let that noise overrule a decisive primary result, dropping the correct
car from the top rank to the middle of the pack. Restricting the
secondary signal to genuine near-ties fixed this without losing anything
in the cases where the ensemble had been helping.

**An absolute "is there a fault at all" gate**, built two ways — a single
global noise-floor threshold, and the same local per-file standardisation
used in Step 2 applied to a gating decision instead of a ranking one.
Neither produced a threshold that reliably separated a genuine fault from
an entirely healthy fleet: the two distributions overlapped in both
versions. The underlying reason isn't a calibration problem — it traces
to specific cars that have a real, sustained, non-random temperature
offset from their siblings for reasons unrelated to refrigerant (physical
position on the train, occupancy, or similar), which a sustained-drift
detector like CUSUM cannot distinguish from an actual leak using
temperature alone. Consequence: **this algorithm ranks; it does not
gate.** It always names a most-suspicious car and never outputs "no fault
detected."
