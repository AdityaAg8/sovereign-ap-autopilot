"""Unit tests for evals/scorers/perf.py's latency/VRAM aggregation ("p50/p95 seconds per invoice,
peak VRAM per stage"). Uses an injected fake clock and VRAM sampler so this needs no real GPU and
no real wall-clock waiting."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from evals.scorers.perf import PerfCollector


def _fake_clock(times: list[float]) -> Callable[[], float]:
    it = iter(times)

    def _clock() -> float:
        return next(it)

    return _clock


def test_latency_stats_hand_computed_percentiles() -> None:
    # 4 calls with elapsed times 1, 2, 3, 4 seconds (clock ticks: 0->1, 1->3, 3->6, 6->10).
    clock = _fake_clock([0.0, 1.0, 1.0, 3.0, 3.0, 6.0, 6.0, 10.0])
    collector = PerfCollector(sample_fn=lambda: None, clock=clock)

    for _ in range(4):
        with collector.timed(sample_vram=False):
            pass

    stats = collector.latency_stats()
    assert stats.n == 4
    # numpy's linear-interpolation percentile of [1,2,3,4]: p50=2.5, p95=3.85
    assert stats.p50_s == pytest.approx(2.5)
    assert stats.p95_s == pytest.approx(3.85)


def test_latency_stats_raises_when_no_calls_recorded() -> None:
    collector = PerfCollector(sample_fn=lambda: None)
    with pytest.raises(ValueError, match="no timed calls"):
        collector.latency_stats()


def test_peak_vram_gb_is_none_when_sampler_never_reports_a_value() -> None:
    collector = PerfCollector(sample_fn=lambda: None, clock=_fake_clock([0.0, 0.001]))
    with collector.timed(sample_vram=True):
        pass
    assert collector.peak_vram_gb is None


def test_peak_vram_gb_tracks_the_max_across_calls() -> None:
    samples = iter([2.0, 5.0, 1.0, 8.0, 3.0])
    collector = PerfCollector(sample_fn=lambda: next(samples), clock=_fake_clock([0.0, 0.001] * 3))

    for _ in range(3):
        with collector.timed(sample_vram=True):
            pass

    # VramSampler takes one sample at __enter__ (before the block runs) plus whatever its
    # background thread polls in that ~0 duration; with sample_interval_s far longer than our
    # near-instant block, exactly one sample per call is guaranteed: 2.0, then 5.0 (> 2.0), then
    # 1.0 (< 5.0, no change) -> running peak across calls is max(2.0, 5.0, 1.0) = 5.0.
    assert collector.peak_vram_gb == 5.0


def test_sample_vram_false_never_calls_sample_fn() -> None:
    def _boom() -> float:
        raise AssertionError("sample_fn should not be called when sample_vram=False")

    collector = PerfCollector(sample_fn=_boom, clock=_fake_clock([0.0, 0.001]))
    with collector.timed(sample_vram=False):
        pass
    assert collector.peak_vram_gb is None
