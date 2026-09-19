# SHM Subsystem (v3, incremental) — Streaming Damage + Cycles-to-Failure

v3 is v2 (`../v2_rule-based/algorithm.md`) with two additions, not a
different model:

1. The rainflow + Miner's-rule proxy is computed by a **persistent
   streaming engine** (`physics.RainflowState`) instead of a single
   whole-array call, closing pain point 1's remaining gap ("this is a
   batch algorithm... nothing here maintains rainflow's cycle-counting
   state incrementally as new samples arrive").
2. Each prediction also reports **cycles-to-failure** and a **confidence
   interval** on it, addressing the "rapid, online... assessment" half of
   pain point 2 without the accuracy loss that every ML approach tried
   in v1/v2 showed at this sample size (64 files).

Both are validated to reproduce v2's damage numbers to float precision
(~1e-13 relative), not merely "close" — see `test_physics.py` and the
retrain output below. Nothing about the physics changed; only when the
arithmetic runs.

## 1. Streaming rainflow (pain point 1, the part v2 left open)

ASTM E1049's rainflow algorithm is not inherently batch: it processes
turning points sequentially against a small stack (`points` in the
reference implementation) and closes a cycle the moment three
consecutive stack points satisfy the amplitude test. That test only
needs the current stack, never the whole series. `physics.RainflowState`
reimplements that stack directly (rather than calling the `rainflow`
package's whole-array `extract_cycles`) and exposes it as state a caller
can `feed()` new chunks into as data arrives:

- `_points`: the residual stack of unclosed reversals. Bounded by the
  number of currently-open reversals, not by how much data has been
  seen — this is the only state that needs to persist between chunks.
- `_a`, `_b`, running index: the minimum context needed to detect
  turning points across a chunk boundary (the same sign-change test
  `rainflow.reversals()` uses internally).
- `proxy`, `n_cycles`: running totals over cycles closed **so far**.
  Querying these mid-stream slightly *underestimates* final damage,
  because reversals still on the stack haven't closed yet — the same
  caveat every online rainflow implementation carries. Calling
  `.close()` once, at the genuine end of a monitoring interval, flushes
  the residual exactly like the batch algorithm's own end-of-series step.

This pattern (process one block/sample at a time, keep only the residual
stack between calls) is published, established practice, not invented
here:
- [Rainflow counting algorithm for very long stress histories](https://www.sciencedirect.com/science/article/abs/pii/0142112387900259)
  (Intl. J. Fatigue, 1989) — block-at-a-time counting without holding
  the whole history in memory.
- [A novel online 4-point rainflow counting algorithm for power electronics](https://www.sciencedirect.com/science/article/abs/pii/S0026271421000780)
  (Microelectronics Reliability, 2021) — sample-per-sample online
  counting built for real-time embedded use.

`run_pipeline.py` demonstrates this concretely: `--chunk-size` (default
4096 rows) controls how much of a 581,120-row file is "seen" at once,
and the result is identical for any chunk size, including 1 (verified in
`test_physics.py` against synthetic series with duplicate-value edge
cases, and against three real Train files at chunk sizes 1, 7, 4096, and
whole-file).

## 2. Cycles-to-failure (pain point 2, the addressable half)

Given the current damage estimate `D` (from the same Miner's-rule proxy
as v2) and the number of rainflow cycles `n` it was accumulated over, a
**stationary-loading extrapolation** projects forward:

```
cycles_to_failure = n * D_fail / D
cycles_remaining  = cycles_to_failure - n
```

This is standard Palmgren-Miner practice, not a new construction: damage
accumulates linearly in cycles under an unchanging load spectrum, so a
file's own damage-per-cycle rate is exactly the extrapolation slope Miner's
rule already assumes. See
[Understanding Miner's Rule](https://www.regalrexnord.com/en/regal-rexnord-insights/what-is-miners-rule)
and [Miner Rule, Cumulative Fatigue Damage and Load Spectra](https://atlasofengineering.com/materials-engineering/miner-rule/).

Two assumptions, stated rather than hidden:

- **Stationarity.** The recorded loading spectrum is assumed to continue
  unchanged. A single file cannot validate this — it is the structural
  cost of extrapolating from one snapshot instead of a genuine
  run-to-failure history. As covered in the prior research pass, this
  dataset's files are randomly numbered with no cross-file operating
  order, so no algorithm (ML included) can validate stationarity across
  files here; within-file extrapolation is the most that's derivable.
- **`D_fail = 1.0`** is the textbook Miner's-rule convention (used as the
  default), but experimentally observed critical damage sums range
  roughly **0.7–2.2** depending on material and load spectrum
  ([Miner's Rule and Cumulative Damage Models](https://help.reliasoft.com/articles/content/hotwire/issue116/hottopics116.htm)).
  `--d-fail` is exposed as a CLI flag for this reason. That extra scatter
  is **not** included in the confidence interval below — see next section.
- **Units are cycles, not time.** Per `../v1_testing/features.py`'s own
  note, the sampling rate is unknown, so no cycles-per-second conversion
  is possible; reporting a time-to-failure would be fabricating precision
  this data doesn't support.

## 3. Confidence interval on cycles-to-failure

Fatigue life is standardly treated as **log-normally distributed with
roughly constant log-variance** — this is the statistical model behind
ASTM E739 ("Standard Practice for Statistical Analysis of Linear or
Linearized Stress-Life (S-N) and Strain-Life (ε-N) Fatigue Data"):
`log(N) ~ Normal(mu, sigma^2)`, sigma assumed constant, confidence bounds
computed as `mu ± z·sigma` in log space. This log-normal-life,
constant-variance assumption is stated explicitly in, e.g.,
[Probabilistic S-N fields based on statistical distributions applied to metallic and composite materials](https://journals.sagepub.com/doi/10.1177/1687814019870395)
(Barbosa et al., 2019).

**What's adapted here, and why:** ASTM E739's sigma normally comes from
*repeated* fatigue tests at a fixed stress level — several specimens,
same load, different measured lives. This dataset has no such
repetition; every one of the 64 training files is a different,
unrepeated loading history. What does exist is this pipeline's own
leave-one-out residuals against `Train_Labels.csv` — the actual,
honestly cross-validated (`C` refit on the other 63 files each fold)
spread between predicted and true damage across 64 independent files.
`diagnostics.fit_log_residual_sigma()` computes
`sigma_log = std(log(true / predicted))` from that table (currently
`0.0398` on the shipped calibration) and `model.py`'s
`damage_confidence_interval()` / `cycles_to_failure_interval()` apply it
exactly as ASTM E739 applies its own sigma, propagated through the
inverse relationship `cycles_to_failure = n·D_fail/D` (same sigma
either way, since `log(1/D) = -log(D)`).

This is a genuine substitute, not a rebranded arbitrary number — it is
literally the model's own measured error on the only ground truth
available — but it is **not** the ASTM-standard same-stress-level
replicate scatter, and it does **not** include `D_fail`'s own
material-to-material variability (0.7–2.2 above). A reported interval
should be read as "how much this model's damage estimate itself
varies against known files," not as a full materials-uncertainty bound.
Default confidence level is 90% (two-sided, `z ≈ 1.645`); configurable
via `--confidence`.

## 4. Validation

```
$ python run_pipeline.py --retrain --data-dir <SHM> --input <SHM>/Test
...
Mean LOO score (max(0, 1-MAPE)): 0.9744     # identical to v2
C = 2.351993e+10   (m = 5.0)                 # identical to v2
sigma_log = 0.0398
...
test01.csv    0.032277   5.539e+06   [5.19e+06, 5.91e+06]  (90% CI on cycles-to-failure)
```

Predictions match v2's `shm_predictions.csv` to ~1e-13 relative
difference (floating-point order-of-summation noise from chunked vs.
whole-array accumulation, not a modeling difference) — checked directly
against a fresh v2 retrain in this session, and asserted in
`test_physics.py`'s `validate_against_batch` checks against the
`rainflow` package's own batch output on real Train files.

## 5. What this does not solve

Restating from the prior research pass, honestly: true cross-file
service-life forecasting (tracking one physical vehicle's accumulating
damage across many recording sessions over its operating history) is
still not derivable from this dataset — file numbers are randomly
assigned with no sequential or damage-level correlation, so there is no
operating-history trend for any algorithm to learn, ML or otherwise.
What v3 adds is the two things that *were* addressable without more
data: real online computation (§1) and a within-file, assumption-stated
life extrapolation with an honestly-sourced confidence interval (§2–3).
