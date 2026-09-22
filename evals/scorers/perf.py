"""Collects per-call latency (p50/p95) and peak VRAM across an eval run (PRD §5 "p50/p95
seconds per invoice, peak VRAM per stage", §10 M1.3). Reuses `scripts/smoke.py`'s VRAM sampler
(nvidia-smi polled on a background thread) rather than reimplementing it; adds the timing and
percentile bookkeeping an eval run needs across many invoices that smoke.py, which times single
one-off requests, does not."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from scripts.smoke import VramSampler, sample_vram_gb


@dataclass(frozen=True)
class LatencyStats:
    p50_s: float
    p95_s: float
    n: int


@dataclass
class PerfCollector:
    """Accumulates one latency sample (and, optionally, one peak-VRAM sample) per call to
    `timed()`, e.g. once per invoice processed during an eval run. `sample_fn` and `clock` are
    injectable so tests can exercise the aggregation logic without a GPU or real wall-clock
    waits."""

    sample_fn: Callable[[], float | None] = sample_vram_gb
    clock: Callable[[], float] = time.monotonic
    _latencies_s: list[float] = field(default_factory=list, init=False, repr=False)
    _peak_vram_gb: float | None = field(default=None, init=False, repr=False)

    @contextmanager
    def timed(self, *, sample_vram: bool = True) -> Iterator[None]:
        """Times the wrapped block and, if `sample_vram`, records the peak VRAM sampled while
        it ran. Records exactly one latency sample per use, even if VRAM sampling yields no
        value (e.g. no GPU present)."""
        start = self.clock()
        if sample_vram:
            with VramSampler(sample_fn=self.sample_fn) as sampler:
                yield
            peak = sampler.peak_gb
        else:
            yield
            peak = None
        self._latencies_s.append(self.clock() - start)
        if peak is not None:
            self._peak_vram_gb = (
                peak if self._peak_vram_gb is None else max(self._peak_vram_gb, peak)
            )

    def latency_stats(self) -> LatencyStats:
        if not self._latencies_s:
            raise ValueError("no timed calls recorded")
        samples = np.array(self._latencies_s, dtype=float)
        return LatencyStats(
            p50_s=float(np.percentile(samples, 50)),
            p95_s=float(np.percentile(samples, 95)),
            n=len(samples),
        )

    @property
    def peak_vram_gb(self) -> float | None:
        return self._peak_vram_gb
