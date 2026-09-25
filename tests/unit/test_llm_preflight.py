"""VRAM/RAM preflight guard (`apx.llm.preflight`, ADR-0013): parses real `nvidia-smi` and
`/proc/meminfo` text, passes with headroom, raises the typed shortfall error at the margin
boundary, waits deterministically on an injected clock, and never passes when probing fails."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypedDict

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.common.errors import (
    ConfigError,
    GpuProbeError,
    GpuProbeTransientError,
    InsufficientGpuMemoryError,
    InsufficientHostMemoryError,
    ModelServingError,
)
from apx.llm import preflight
from apx.llm.preflight import (
    MEMORY_NEEDS,
    GpuMemory,
    GpuWorkload,
    MemoryNeed,
    check_gpu_headroom,
    check_workload_headroom,
    parse_meminfo_available_gb,
    parse_nvidia_smi_memory,
    query_gpu_memory,
    query_host_available_gb,
    run_nvidia_smi,
)

SMI_ONE = "11264, 12282\n"
SMI_TWO = "9123, 12282\n 512, 24564\n"
MEMINFO = (
    "MemTotal:       16481980 kB\n"
    "MemFree:         9302860 kB\n"
    "MemAvailable:   13559456 kB\n"
    "Buffers:          251724 kB\n"
)


def _gpu(free_gb: float) -> preflight.GpuProbe:
    return lambda: GpuMemory(free_gb=free_gb, total_gb=12.0)


def _ram(free_gb: float) -> preflight.RamProbe:
    return lambda: free_gb


def _no_ram() -> float:
    raise AssertionError("RAM must not be probed when it is not budgeted")


# --- parsing ---------------------------------------------------------------------------------


def test_parses_single_and_multi_gpu_nvidia_smi() -> None:
    assert parse_nvidia_smi_memory(SMI_ONE) == [GpuMemory(11264 / 1024, 12282 / 1024)]
    two = parse_nvidia_smi_memory(SMI_TWO)
    assert [g.free_gb for g in two] == [9123 / 1024, 0.5]
    assert two[1].total_gb == 24564 / 1024


def test_query_picks_the_requested_gpu_and_rejects_a_missing_index() -> None:
    calls: list[tuple[str, ...]] = []

    def smi(args: tuple[str, ...]) -> str:
        calls.append(args)
        return SMI_TWO

    assert query_gpu_memory(smi, gpu_index=1).free_gb == 0.5
    assert calls == [
        ("--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"),
    ]
    with pytest.raises(GpuProbeError, match="index 2"):
        query_gpu_memory(smi, gpu_index=2)


@pytest.mark.parametrize(
    "text",
    ["", "\n", "[N/A], [N/A]\n", "1024\n", "1024, 2048, 3\n", "4096, 2048\n", "-1, 2048\n"],
)
def test_bad_nvidia_smi_output_raises(text: str) -> None:
    with pytest.raises(GpuProbeError):
        parse_nvidia_smi_memory(text)


@given(
    st.lists(
        st.integers(1, 200_000).flatmap(
            lambda total: st.tuples(st.integers(0, total), st.just(total))
        ),
        min_size=1,
        max_size=8,
    )
)
def test_nvidia_smi_parser_round_trips(rows: list[tuple[int, int]]) -> None:
    text = "".join(f"{free}, {total}\n" for free, total in rows)
    parsed = parse_nvidia_smi_memory(text)
    assert [(g.free_gb * 1024, g.total_gb * 1024) for g in parsed] == [
        (float(f), float(t)) for f, t in rows
    ]


def test_parses_meminfo_available() -> None:
    assert parse_meminfo_available_gb(MEMINFO) == 13559456 / (1024 * 1024)


@given(st.integers(0, 2**40))
def test_meminfo_parser_is_exact(kb: int) -> None:
    text = f"MemTotal: {kb + 1} kB\nMemAvailable: {kb} kB\n"
    assert parse_meminfo_available_gb(text) == kb / (1024 * 1024)


@pytest.mark.parametrize(
    "text", ["MemTotal: 1 kB\n", "MemAvailable: lots kB\n", "MemAvailable: 12 MB\n", ""]
)
def test_bad_meminfo_raises(text: str) -> None:
    with pytest.raises(GpuProbeError):
        parse_meminfo_available_gb(text)


class _CgroupPaths(TypedDict):
    cgroup_max: Path
    cgroup_current: Path


def _host_files(tmp_path: Path, limit: str | None, current: str | None) -> _CgroupPaths:
    (tmp_path / "meminfo").write_text(MEMINFO, encoding="ascii")
    if limit is not None:
        (tmp_path / "memory.max").write_text(limit, encoding="ascii")
    if current is not None:
        (tmp_path / "memory.current").write_text(current, encoding="ascii")
    return {
        "cgroup_max": tmp_path / "memory.max",
        "cgroup_current": tmp_path / "memory.current",
    }


HOST_AVAILABLE_GB = 13559456 / (1024 * 1024)  # ~12.93 GB


@pytest.mark.parametrize(
    ("limit", "current"),
    [(None, None), ("max\n", "1073741824\n"), ("8589934592\n", None), (None, "1\n")],
)
def test_host_probe_uses_meminfo_without_a_cgroup_limit(
    tmp_path: Path, limit: str | None, current: str | None
) -> None:
    paths = _host_files(tmp_path, limit, current)
    assert query_host_available_gb(tmp_path / "meminfo", **paths) == HOST_AVAILABLE_GB


def test_host_probe_caps_at_the_cgroup_headroom(tmp_path: Path) -> None:
    gib = 1024**3
    paths = _host_files(tmp_path, f"{8 * gib}\n", f"{3 * gib}\n")
    assert query_host_available_gb(tmp_path / "meminfo", **paths) == 5.0
    over = _host_files(tmp_path, f"{2 * gib}\n", f"{3 * gib}\n")  # over its limit: no room
    assert query_host_available_gb(tmp_path / "meminfo", **over) == 0.0
    roomy = _host_files(tmp_path, f"{64 * gib}\n", "0\n")  # limit above host: host wins
    assert query_host_available_gb(tmp_path / "meminfo", **roomy) == HOST_AVAILABLE_GB


def test_host_probe_uses_the_injected_reader() -> None:
    files = {
        Path("/m"): MEMINFO,
        Path("/max"): str(4 * 1024**3),
        Path("/cur"): str(1024**3),
    }
    got = query_host_available_gb(
        Path("/m"), cgroup_max=Path("/max"), cgroup_current=Path("/cur"), read=files.get
    )
    assert got == 3.0


def test_host_probe_types_missing_and_bad_files(tmp_path: Path) -> None:
    paths = _host_files(tmp_path, None, None)
    with pytest.raises(GpuProbeError, match="cannot read"):
        query_host_available_gb(tmp_path / "absent", **paths)
    bad = _host_files(tmp_path, "lots\n", "1\n")
    with pytest.raises(GpuProbeError, match="unparsable"):
        query_host_available_gb(tmp_path / "meminfo", **bad)
    (tmp_path / "dir").mkdir()
    with pytest.raises(GpuProbeError, match="cannot read"):
        query_host_available_gb(tmp_path / "dir", **paths)


# --- nvidia-smi process failures -------------------------------------------------------------


def test_missing_nvidia_smi_raises_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", absent)
    with pytest.raises(GpuProbeError, match="not found") as info:
        run_nvidia_smi(preflight.NVIDIA_SMI_MEMORY_ARGS)
    assert type(info.value) is GpuProbeError  # deployment fault: the non-retryable class
    with pytest.raises(GpuProbeError):  # and the default check never passes blind
        check_gpu_headroom(1.0, 0.0)


def test_nvidia_smi_timeout_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    def hung(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["nvidia-smi"], 30)

    monkeypatch.setattr(subprocess, "run", hung)
    with pytest.raises(GpuProbeTransientError, match="timed out after 30 s"):
        run_nvidia_smi(preflight.NVIDIA_SMI_MEMORY_ARGS)


def test_other_os_failure_stays_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise PermissionError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", denied)
    with pytest.raises(GpuProbeError, match="failed") as info:
        run_nvidia_smi(preflight.NVIDIA_SMI_MEMORY_ARGS)
    assert type(info.value) is GpuProbeError


def test_unparsable_nvidia_smi_output_stays_non_retryable() -> None:
    with pytest.raises(GpuProbeError) as info:
        query_gpu_memory(lambda _args: "garbage\n")
    assert type(info.value) is GpuProbeError


def test_nonzero_nvidia_smi_exit_raises_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(9, ["nvidia-smi"], stderr="NVML: driver mismatch")

    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(GpuProbeTransientError, match="exited 9: NVML: driver mismatch"):
        run_nvidia_smi(preflight.NVIDIA_SMI_MEMORY_ARGS)


def test_probe_errors_are_model_serving_errors() -> None:
    for cls in (
        GpuProbeError,
        GpuProbeTransientError,
        InsufficientGpuMemoryError,
        InsufficientHostMemoryError,
    ):
        assert issubclass(cls, ModelServingError)
    assert issubclass(GpuProbeTransientError, GpuProbeError)  # existing handlers still catch it


# --- thresholds ------------------------------------------------------------------------------


def test_headroom_passes_and_reports_what_it_measured() -> None:
    room = check_gpu_headroom(4.0, 8.0, gpu_probe=_gpu(6.0), ram_probe=_ram(10.0))
    assert (room.free_vram_gb, room.free_ram_gb) == (6.0, 10.0)


def test_vram_shortfall_raises_with_need_margin_and_free() -> None:
    with pytest.raises(InsufficientGpuMemoryError) as info:
        check_gpu_headroom(7.25, 0.0, 0.5, gpu_probe=_gpu(7.5), ram_probe=_no_ram)
    assert str(info.value) == "free VRAM 7.50 GB < need 7.25 GB + margin 0.50 GB"


def test_ram_shortfall_raises_with_need_margin_and_free() -> None:
    with pytest.raises(InsufficientHostMemoryError) as info:
        check_gpu_headroom(6.0, 12.0, gpu_probe=_gpu(11.0), ram_probe=_ram(12.25))
    assert str(info.value) == "available RAM 12.25 GB < need 12.00 GB + margin 0.30 GB"


def test_zero_ram_need_does_not_probe_ram() -> None:
    assert check_gpu_headroom(1.0, 0.0, gpu_probe=_gpu(2.0), ram_probe=_no_ram).free_ram_gb is None


def test_margin_boundary_is_inclusive() -> None:
    check_gpu_headroom(4.0, 2.0, 0.5, gpu_probe=_gpu(4.5), ram_probe=_ram(2.5))
    with pytest.raises(InsufficientGpuMemoryError):
        check_gpu_headroom(4.0, 2.0, 0.5, gpu_probe=_gpu(4.4990234375), ram_probe=_ram(2.5))
    with pytest.raises(InsufficientHostMemoryError):
        check_gpu_headroom(4.0, 2.0, 0.5, gpu_probe=_gpu(4.5), ram_probe=_ram(2.4990234375))


@given(
    need=st.floats(0.0, 32.0),
    margin=st.floats(0.0, 2.0),
    free=st.floats(0.0, 40.0),
)
def test_vram_threshold_property(need: float, margin: float, free: float) -> None:
    fits = free >= need + margin
    try:
        check_gpu_headroom(need, 0.0, margin, gpu_probe=_gpu(free), ram_probe=_no_ram)
    except InsufficientGpuMemoryError:
        assert not fits
    else:
        assert fits


@pytest.mark.parametrize(
    "kwargs",
    [
        {"need_vram_gb": -1.0, "need_ram_gb": 0.0},
        {"need_vram_gb": 1.0, "need_ram_gb": -1.0},
        {"need_vram_gb": 1.0, "need_ram_gb": 0.0, "margin_gb": -0.1},
        {"need_vram_gb": 1.0, "need_ram_gb": 0.0, "wait_s": -1.0},
        {"need_vram_gb": 1.0, "need_ram_gb": 0.0, "poll_s": 0.0},
    ],
)
def test_invalid_arguments_raise_config_error(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        check_gpu_headroom(**kwargs, gpu_probe=_gpu(99.0), ram_probe=_ram(99.0))  # type: ignore[arg-type]


# --- bounded wait ----------------------------------------------------------------------------


class _FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_wait_times_out_deterministically() -> None:
    t = _FakeTime()
    probes: list[float] = []

    def gpu() -> GpuMemory:
        probes.append(t.now)
        return GpuMemory(free_gb=3.0, total_gb=12.0)

    with pytest.raises(InsufficientGpuMemoryError):
        check_gpu_headroom(
            4.0, 0.0, wait_s=2.5, poll_s=1.0, gpu_probe=gpu, clock=t.clock, sleep=t.sleep
        )
    assert t.sleeps == [1.0, 1.0, 0.5]
    assert probes == [100.0, 101.0, 102.0, 102.5]


def test_wait_succeeds_when_headroom_appears_mid_wait() -> None:
    t = _FakeTime()
    frees = iter([3.0, 3.5, 5.0])
    room = check_gpu_headroom(
        4.0,
        0.0,
        wait_s=10.0,
        poll_s=1.0,
        gpu_probe=lambda: GpuMemory(free_gb=next(frees), total_gb=12.0),
        clock=t.clock,
        sleep=t.sleep,
    )
    assert room.free_vram_gb == 5.0
    assert t.sleeps == [1.0, 1.0]


def test_zero_wait_never_sleeps() -> None:
    t = _FakeTime()
    with pytest.raises(InsufficientGpuMemoryError):
        check_gpu_headroom(4.0, 0.0, gpu_probe=_gpu(1.0), clock=t.clock, sleep=t.sleep)
    assert t.sleeps == []


def test_probe_failure_is_not_waited_out() -> None:
    t = _FakeTime()

    def broken() -> GpuMemory:
        raise GpuProbeError("nvidia-smi not found")

    with pytest.raises(GpuProbeError):
        check_gpu_headroom(1.0, 0.0, wait_s=30.0, gpu_probe=broken, clock=t.clock, sleep=t.sleep)
    assert t.sleeps == []


# --- needs table -----------------------------------------------------------------------------


def test_needs_table_matches_adr_0013() -> None:
    assert dict(MEMORY_NEEDS) == {
        GpuWorkload.EXTRACTOR_A: MemoryNeed(7.2, 0.0),
        GpuWorkload.EXTRACTOR_B: MemoryNeed(4.4, 0.0),
        GpuWorkload.PARSE: MemoryNeed(4.3, 0.0),
        GpuWorkload.ARBITER: MemoryNeed(8.0, 0.0),
        GpuWorkload.PAGE_SIMILARITY: MemoryNeed(4.7, 0.0),
        GpuWorkload.GUARD: MemoryNeed(1.0, 0.0),
        GpuWorkload.TRAINER: MemoryNeed(6.0, 12.0),
    }
    with pytest.raises(TypeError):
        MEMORY_NEEDS[GpuWorkload.GUARD] = MemoryNeed(0.0, 0.0)  # type: ignore[index]


def test_workload_check_uses_its_table_row() -> None:
    seen: list[tuple[float, float, float]] = []

    def check(vram: float, ram: float, *, wait_s: float) -> preflight.Headroom:
        seen.append((vram, ram, wait_s))
        return preflight.Headroom(free_vram_gb=vram, free_ram_gb=ram)

    check_workload_headroom(GpuWorkload.TRAINER, check=check, wait_s=4.0)
    assert seen == [(6.0, 12.0, 4.0)]
