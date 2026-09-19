"""
SHM Subsystem (v3, incremental) — Streaming vs. Batch Equivalence Tests
============================================================================
The one thing that has to hold for `RainflowState` to be a valid drop-in
for v2's batch `rainflow.extract_cycles` call: feeding a series through
in chunks, at any chunk size, must produce the same closed cycles as
feeding it in one call. These tests check that directly, on synthetic
edge cases and on real Train files, rather than asserting it in prose.

Run with: python -m pytest test_physics.py -v
"""

from pathlib import Path

import numpy as np
import pytest

import physics

DATA_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "SHM"

SYNTHETIC_SERIES = {
    "hand_worked_example": np.array([1.0, 5.0, 2.0, 8.0, 1.0, 6.0, 3.0, 9.0, 0.0, 4.0]),
    "gaussian_500": np.random.default_rng(0).normal(size=500),
    "gaussian_5000": np.random.default_rng(1).normal(size=5000),
    "with_duplicates": np.repeat(np.random.default_rng(2).normal(size=200), 3),
    "constant": np.full(50, 3.0),
    "three_points": np.array([1.0, 7.0, 3.0]),
    "empty": np.array([]),
}

CHUNK_SIZES = [1, 3, 7, 50, 4096]


@pytest.mark.parametrize("name", SYNTHETIC_SERIES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_incremental_matches_batch_synthetic(name, chunk_size):
    x = SYNTHETIC_SERIES[name]
    if len(x) == 0:
        pytest.skip("empty series: nothing to close")
    result = physics.validate_against_batch(x, chunk_size=chunk_size)
    assert result["proxy_rel_diff"] < 1e-9, result
    assert result["incremental_n_cycles"] == pytest.approx(result["batch_n_cycles"])


@pytest.mark.skipif(not DATA_DIR.exists(), reason="SHM dataset not available in this checkout")
@pytest.mark.parametrize("filename", ["train01.csv", "train12.csv", "train64.csv"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096, None])
def test_incremental_matches_batch_real_files(filename, chunk_size):
    x = physics.load_stress_series(DATA_DIR / "Train" / filename)
    result = physics.validate_against_batch(x, chunk_size=chunk_size or len(x))
    assert result["proxy_rel_diff"] < 1e-9, result
    assert result["incremental_n_cycles"] == result["batch_n_cycles"]


def test_chunking_is_order_independent_of_chunk_size():
    """The whole point of the streaming engine: the result must not
    depend on how the same series is sliced into arrival chunks."""
    x = np.random.default_rng(3).normal(size=3000)
    proxies = {cs: physics.miner_proxy_incremental(x, chunk_size=cs) for cs in [1, 4, 17, 500, 3000]}
    reference = proxies[3000]
    for cs, (proxy, n) in proxies.items():
        assert proxy == pytest.approx(reference[0], rel=1e-9), f"chunk_size={cs}"
        assert n == reference[1], f"chunk_size={cs}"


def test_state_underestimates_until_close():
    """Mid-stream (before close()), the running proxy should be <= the
    fully-closed proxy, since residual reversals haven't been counted yet."""
    x = np.random.default_rng(4).normal(size=2000)
    state = physics.RainflowState()
    state.feed(x)
    mid_proxy = state.proxy
    state.close()
    assert mid_proxy <= state.proxy


def test_feed_after_close_raises():
    state = physics.RainflowState()
    state.feed([1.0, 2.0, 1.0])
    state.close()
    with pytest.raises(RuntimeError):
        state.feed([3.0])


def test_known_two_sample_edge_case_in_reference_library():
    """Documented divergence, not a bug to fix: `rainflow.reversals()`
    silently drops the LAST point as a reversal when the series has
    exactly 2 elements total (its final-point yield only fires if the
    main loop executed at least once, which needs a 3rd sample). For an
    input of exactly 2 samples, the batch library therefore reports 0
    cycles while `RainflowState` (which always treats a fed series' last
    sample as a reversal on close(), per the documented "first and last
    points are always reversals" rule) reports one closed half-cycle.
    This never arises on real stress recordings (581,120 rows per file
    in this dataset) and is asserted here only so the divergence is
    visible and intentional rather than silently unequal."""
    import rainflow
    x = [1.0, 7.0]
    assert list(rainflow.extract_cycles(x)) == []

    state = physics.RainflowState()
    state.feed(x)
    state.close()
    assert state.n_cycles == 0.5
