"""
SHM Subsystem (v3, incremental) — Streaming Rainflow + Miner's-Rule Proxy
==========================================================================
v2 (`v2_rule-based/physics.py`) calls `rainflow.extract_cycles(x)` on the
whole array every time a prediction is needed -- correct, but it means
"online" monitoring would mean re-reading and re-processing the entire
stress history from scratch on every new sample.

The ASTM E1049-85 rainflow algorithm (the same standard v2 already uses,
just via the `rainflow` package's batch entry point) is not inherently
batch: it processes turning points sequentially against a small stack and
closes a cycle as soon as three consecutive stack points satisfy the
range test. Nothing in that test needs to see the whole series -- only
the stack accumulated so far. This module reimplements that stack
directly and exposes it as persistent state (`RainflowState`) that a
caller can `feed()` new chunks into as they arrive, getting an
up-to-date Miner's-rule proxy after every chunk in O(chunk size) time,
not O(total samples so far).

Precedent for doing this: online/block-processed rainflow counting is
published, not novel here -- see "Rainflow counting algorithm for very
long stress histories" (Intl. J. Fatigue, 1989,
https://doi.org/10.1016/0142-1123(89)90054-3, block-at-a-time counting
without holding the whole history) and "A novel online 4-point rainflow
counting algorithm for power electronics" (Microelectronics Reliability,
2021, https://doi.org/10.1016/j.microrel.2021.114100, sample-per-sample
online counting for real-time use).

Numerical contract: feeding a series to `RainflowState` in one call, in
many small chunks, or one sample at a time, and then calling `.close()`,
produces the same closed cycles (same ranges, means, counts) as
`rainflow.extract_cycles()` called once on the whole array. This is
checked in `validate_against_batch()` below and in `test_physics.py`.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import rainflow  # only used for load_stress_series' batch cross-check, not the streaming path

M_EXPONENT = 5.0   # S-N curve exponent -- same fixed value as v2, see v2_rule-based/algorithm.md


def load_stress_series(path) -> np.ndarray:
    """Load a raw single-column stress CSV (no header) as a 1-D array."""
    return np.loadtxt(path)


class RainflowState:
    """
    Persistent ASTM E1049 rainflow stack. Feed it raw stress samples as
    they arrive, in any chunking; query `.proxy` / `.n_cycles` at any
    point for the damage accumulated so far from cycles that have
    actually closed. Call `.close()` once, at the true end of the
    series, to flush the residual (still-open) reversals the same way
    a batch call to `rainflow.extract_cycles` does.

    State carried between `feed()` calls (all O(1), independent of how
    much data has been fed):
      - `_a`, `_b`      : last two raw samples seen, for turning-point
                          detection across chunk boundaries.
      - `_points`       : deque of unclosed turning points (index, value).
                          Bounded by the number of open reversals, not by
                          series length -- this is the actual "residual"
                          in ASTM E1049 terms.
      - `_index`        : running raw-sample position counter.
      - `proxy`         : running sum of count * range**m over cycles
                          CLOSED so far (an underestimate until `close()`
                          flushes the residual -- see module docstring).
      - `n_cycles`      : running sum of cycle counts closed so far
                          (half-cycles count 0.5, matching `rainflow`).

    Known divergence from `rainflow.extract_cycles`, by design, not a
    bug: for a total input of exactly 2 samples, the reference library's
    `reversals()` silently drops the last sample (its final-point yield
    only fires once its main loop has run, which needs a 3rd sample), so
    it reports 0 cycles where this class reports one closed half-cycle
    (it always treats a fed series' last sample as a reversal, per the
    "first and last points are always reversals" rule the library's own
    docstring states). Irrelevant for this pipeline's real files
    (581,120 rows each); see `test_physics.py`'s
    `test_known_two_sample_edge_case_in_reference_library`.
    """

    def __init__(self, m: float = M_EXPONENT):
        self.m = m
        self._a = None
        self._b = None
        self._started = False
        self._index = 0
        self._points: deque = deque()
        self._closed = False
        self.proxy = 0.0
        self.n_cycles = 0.0

    def _emit_reversal(self, index: int, value: float) -> None:
        self._points.append((index, value))
        while len(self._points) >= 3:
            (_, x1), (_, x2), (_, x3) = self._points[-3], self._points[-2], self._points[-1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            if X < Y:
                break
            elif len(self._points) == 3:
                self._close_cycle(self._points[0][1], self._points[1][1], 0.5)
                self._points.popleft()
            else:
                self._close_cycle(x1, x2, 1.0)
                last = self._points.pop()
                self._points.pop()
                self._points.pop()
                self._points.append(last)

    def _close_cycle(self, x1: float, x2: float, count: float) -> None:
        rng = abs(x1 - x2)
        self.proxy += count * rng ** self.m
        self.n_cycles += count

    def feed(self, chunk) -> None:
        """Push a new chunk of raw samples through the reversal detector
        and the rainflow stack. Safe to call repeatedly as data arrives;
        do not call after `close()`."""
        if self._closed:
            raise RuntimeError("RainflowState already closed -- no more data can be fed.")
        for x in chunk:
            x = float(x)
            if self._a is None:
                self._a = x
                if not self._started:
                    self._emit_reversal(self._index, self._a)
                    self._started = True
                self._index += 1
                continue
            if self._b is None:
                self._b = x
                self._index += 1
                continue
            if x == self._b:
                # Matches rainflow.reversals(): an exact-duplicate sample is
                # skipped rather than treated as its own turning point.
                self._index += 1
                continue
            d_prev = self._b - self._a
            d_next = x - self._b
            if d_prev * d_next < 0:
                self._emit_reversal(self._index - 1, self._b)
            self._a, self._b = self._b, x
            self._index += 1

    def close(self) -> None:
        """Flush the residual stack as half-cycles, exactly like the
        final `while len(points) > 1` step in `rainflow.extract_cycles`.
        Call this once, only when the series is truly finished (e.g. end
        of a fixed-length file, or a monitoring shift/inspection interval
        boundary) -- calling it mid-series and then feeding more data
        would double count whatever the residual still holds."""
        if self._closed:
            return
        # The last raw sample is always a reversal (matches `reversals()`),
        # even if it wasn't yet confirmed as a direction change.
        if self._b is not None:
            self._emit_reversal(self._index - 1, self._b)
        elif self._a is not None and not self._started:
            self._emit_reversal(self._index - 1, self._a)
        while len(self._points) > 1:
            self._close_cycle(self._points[0][1], self._points[1][1], 0.5)
            self._points.popleft()
        self._closed = True


def miner_proxy_incremental(x: np.ndarray, m: float = M_EXPONENT, chunk_size: int = 4096) -> tuple[float, float]:
    """
    Reference online computation: feeds `x` through `RainflowState` in
    chunks (simulating data arriving over time) and closes it at the end.
    Returns (proxy, n_cycles). For any fixed `x`, the result is identical
    regardless of `chunk_size` -- that's the property that makes this
    genuinely incremental rather than a relabelled batch call.
    """
    state = RainflowState(m)
    for start in range(0, len(x), chunk_size):
        state.feed(x[start:start + chunk_size])
    state.close()
    return state.proxy, state.n_cycles


def miner_proxy(x: np.ndarray, m: float = M_EXPONENT) -> float:
    """Batch Miner's-rule proxy, kept for calibration/validation parity
    with v2 -- computed via the same incremental engine in one shot."""
    proxy, _ = miner_proxy_incremental(x, m, chunk_size=len(x) or 1)
    return proxy


def rainflow_cycles(x: np.ndarray):
    """ASTM-standard batch cycle extraction via the `rainflow` package,
    kept only as the ground truth `validate_against_batch()` checks the
    incremental engine against -- not used in the prediction path."""
    return list(rainflow.extract_cycles(x))


def validate_against_batch(x: np.ndarray, m: float = M_EXPONENT, chunk_size: int = 4096) -> dict:
    """
    Cross-check: does chunked incremental processing reproduce the batch
    `rainflow` package's result on this exact series? Returns a dict with
    both proxies and their relative difference. Used in tests and as a
    runtime sanity check in run_pipeline.py.
    """
    batch_cycles = rainflow_cycles(x)
    ranges = np.array([c[0] for c in batch_cycles]) if batch_cycles else np.array([])
    counts = np.array([c[2] for c in batch_cycles]) if batch_cycles else np.array([])
    batch_proxy = float(np.sum(counts * ranges ** m)) if batch_cycles else 0.0
    batch_n = float(np.sum(counts)) if batch_cycles else 0.0

    inc_proxy, inc_n = miner_proxy_incremental(x, m, chunk_size)

    rel_diff = abs(inc_proxy - batch_proxy) / batch_proxy if batch_proxy else 0.0
    return {
        "batch_proxy": batch_proxy, "incremental_proxy": inc_proxy, "proxy_rel_diff": rel_diff,
        "batch_n_cycles": batch_n, "incremental_n_cycles": inc_n,
    }


def compute_train_proxies(data_dir: Path, m: float = M_EXPONENT) -> pd.DataFrame:
    """
    Rainflow + Miner's-rule proxy for every labelled training file, plus
    the cycle count each file's proxy was accumulated over (needed for
    the cycles-to-failure rate in model.py). Computed once and shared,
    same rationale as v2.
    """
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    rows = []
    for _, row in labels.iterrows():
        x = load_stress_series(data_dir / "Train" / row["filename"])
        proxy, n_cycles = miner_proxy_incremental(x, m)
        rows.append({
            "filename": row["filename"], "proxy": proxy, "n_cycles": n_cycles,
            "damage": row["damage"],
        })
    return pd.DataFrame(rows)
