"""Preflight wiring in `GpuStageScheduler` (ADR-0013): every stage entry checks headroom with its
`MEMORY_NEEDS` row after the releases and before the stage is active, a refusal propagates
typed and leaves no stage active, and the default preflight never passes without `nvidia-smi`."""

from __future__ import annotations

import subprocess

import pytest

from apx.common.errors import GpuProbeError, InsufficientGpuMemoryError
from apx.llm import preflight
from apx.llm.preflight import GpuWorkload, Headroom
from apx.workflows.gpu_stage import (
    STAGE_PREFLIGHT_WAIT_S,
    GpuStage,
    GpuStageScheduler,
    HeadroomPreflight,
)


class _Recorder:
    """Stands in for `check_gpu_headroom` (the probe), recording what the stage asked for."""

    def __init__(self, refuse: bool = False) -> None:
        self.calls: list[tuple[float, float, float]] = []
        self.refuse = refuse

    def __call__(self, need_vram_gb: float, need_ram_gb: float, *, wait_s: float) -> Headroom:
        self.calls.append((need_vram_gb, need_ram_gb, wait_s))
        if self.refuse:
            raise InsufficientGpuMemoryError("free VRAM 1.00 GB < need")
        return Headroom(free_vram_gb=12.0, free_ram_gb=None)


def test_every_stage_has_a_workload_row() -> None:
    for stage in GpuStage:
        assert GpuWorkload(stage.value) in preflight.MEMORY_NEEDS


async def test_each_stage_entry_checks_its_table_numbers_after_the_release() -> None:
    order: list[str] = []
    recorder = _Recorder()

    def hook(stage: GpuStage):  # type: ignore[no-untyped-def]
        async def release() -> None:
            order.append(f"release:{stage.value}")

        return release

    inner = HeadroomPreflight(check=recorder, wait_s=5.0)

    def checked(stage: GpuStage) -> None:
        order.append(f"check:{stage.value}")
        inner(stage)

    scheduler = GpuStageScheduler(
        {s: hook(s) for s in GpuStage}, idle_grace_s=0.0, poll_s=0.01, preflight=checked
    )
    for stage in (GpuStage.EXTRACTOR_A, GpuStage.EXTRACTOR_B, GpuStage.PARSE):
        async with scheduler.hold(stage):
            pass
    # Joining an already-active stage loads nothing, so it is not checked again.
    async with scheduler.hold(GpuStage.PARSE):
        pass
    assert recorder.calls == [(7.2, 0.0, 5.0), (4.4, 0.0, 5.0), (4.3, 0.0, 5.0)]
    assert order[:3] == ["release:parse", "release:extractor-b", "check:extractor-a"]
    assert order[3:] == [
        "release:extractor-a",
        "check:extractor-b",
        "release:extractor-b",
        "check:parse",
    ]


async def test_refusal_propagates_and_leaves_no_stage_active() -> None:
    recorder = _Recorder(refuse=True)
    scheduler = GpuStageScheduler(
        {}, poll_s=0.01, preflight=HeadroomPreflight(check=recorder, wait_s=0.0)
    )
    with pytest.raises(InsufficientGpuMemoryError):
        async with scheduler.hold(GpuStage.EXTRACTOR_A):
            pass
    assert scheduler.active is None
    recorder.refuse = False  # the next caller retries the check and gets in
    async with scheduler.hold(GpuStage.EXTRACTOR_A):
        assert scheduler.active is GpuStage.EXTRACTOR_A
    assert len(recorder.calls) == 2


async def test_default_preflight_refuses_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent(*_a: object, **_k: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", absent)
    scheduler = GpuStageScheduler({}, poll_s=0.01)  # default: HeadroomPreflight()
    with pytest.raises(GpuProbeError, match="not found"):
        async with scheduler.hold(GpuStage.EXTRACTOR_B):
            pass
    assert scheduler.active is None


def test_default_stage_wait_is_bounded() -> None:
    assert 0.0 < STAGE_PREFLIGHT_WAIT_S <= 60.0
