"""Trainer preflight wiring (ADR-0013): `ml.extractor_b.gpu_job.run` checks the trainer's
`MEMORY_NEEDS` row (6 GB VRAM, 12 GB RAM) before anything else, so a short card or host refuses
the run typed before any model, tracker or data load. CPU only; the probe is injected."""

from __future__ import annotations

import pytest

from apx.common.errors import InsufficientHostMemoryError
from apx.llm.preflight import Headroom
from ml.extractor_b import gpu_job, train
from ml.extractor_b.interfaces import TrainConfig


@pytest.mark.parametrize("argv", [["--new-run"], ["--probe"]])
def test_run_checks_trainer_headroom_first(argv: list[str], cfg: TrainConfig) -> None:
    calls: list[tuple[float, float, float]] = []

    def short_host(need_vram_gb: float, need_ram_gb: float, *, wait_s: float) -> Headroom:
        calls.append((need_vram_gb, need_ram_gb, wait_s))
        raise InsufficientHostMemoryError("available RAM 8.00 GB < need 12.00 GB + margin 0.30 GB")

    with pytest.raises(InsufficientHostMemoryError):
        gpu_job.run(train.parse_args(argv), cfg, {}, headroom_check=short_host)
    assert calls == [(6.0, 12.0, 0.0)]
