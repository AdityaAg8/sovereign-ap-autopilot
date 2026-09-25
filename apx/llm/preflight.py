"""VRAM/RAM preflight guard (ADR-0013): before a GPU stage loads its model, or the trainer starts,
read free VRAM from `nvidia-smi` and available RAM from `/proc/meminfo` and raise a typed error
when either is below the stage's need plus a margin. Probes are injectable; no outbound calls."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

from apx.common.errors import (
    ConfigError,
    GpuProbeError,
    GpuProbeTransientError,
    InsufficientGpuMemoryError,
    InsufficientHostMemoryError,
)

_MIB_PER_GB: Final[float] = 1024.0
_KB_PER_GB: Final[float] = 1024.0 * 1024.0
NVIDIA_SMI_MEMORY_ARGS: Final[tuple[str, ...]] = (
    "--query-gpu=memory.free,memory.total",
    "--format=csv,noheader,nounits",
)
DEFAULT_MARGIN_GB: Final[float] = 0.3
_BYTES_PER_GB: Final[float] = 1024.0**3
MEMINFO_PATH: Final[Path] = Path("/proc/meminfo")
CGROUP_MEMORY_MAX: Final[Path] = Path("/sys/fs/cgroup/memory.max")
CGROUP_MEMORY_CURRENT: Final[Path] = Path("/sys/fs/cgroup/memory.current")


class GpuWorkload(StrEnum):
    """Everything that takes a share of the card. `apx.workflows.gpu_stage.GpuStage`'s values
    (parse, extractor-a, extractor-b, page-similarity) are workloads here, so a stage maps to
    its workload by value."""

    PARSE = "parse"
    EXTRACTOR_A = "extractor-a"
    EXTRACTOR_B = "extractor-b"
    ARBITER = "arbiter"
    PAGE_SIMILARITY = "page-similarity"
    GUARD = "guard"
    TRAINER = "trainer"


@dataclass(frozen=True, slots=True)
class MemoryNeed:
    """Memory a workload needs free before it loads, in GB (GiB)."""

    vram_gb: float
    ram_gb: float


MEMORY_NEEDS: Final[Mapping[GpuWorkload, MemoryNeed]] = MappingProxyType(
    {
        # ADR-0013 "Measurements": models-at-peak per stage, desktop excluded. The ADR budgets
        # VRAM only, so RAM needs outside the trainer are 0 (not checked) until measured.
        GpuWorkload.EXTRACTOR_A: MemoryNeed(vram_gb=7.2, ram_gb=0.0),
        GpuWorkload.EXTRACTOR_B: MemoryNeed(vram_gb=4.4, ram_gb=0.0),
        GpuWorkload.PARSE: MemoryNeed(vram_gb=4.3, ram_gb=0.0),  # 0.35 x 12 GB fraction cap
        GpuWorkload.ARBITER: MemoryNeed(vram_gb=8.0, ram_gb=0.0),  # unmeasured: an estimate
        # Measured 2026-09-25 (ADR-0013 needs table): page-embed alone embedding a 5-page A4
        # invoice at the 1750-token cap, whole-card peak 6260 MiB minus 1465 MiB idle = 4.68.
        GpuWorkload.PAGE_SIMILARITY: MemoryNeed(vram_gb=4.7, ram_gb=0.0),
        GpuWorkload.GUARD: MemoryNeed(vram_gb=1.0, ram_gb=0.0),
        GpuWorkload.TRAINER: MemoryNeed(vram_gb=6.0, ram_gb=12.0),
    }
)


@dataclass(frozen=True, slots=True)
class GpuMemory:
    """One GPU's memory as `nvidia-smi` reports it, in GB."""

    free_gb: float
    total_gb: float


@dataclass(frozen=True, slots=True)
class Headroom:
    """What the passing check measured. `free_ram_gb` is None when RAM was not budgeted."""

    free_vram_gb: float
    free_ram_gb: float | None


SmiRunner = Callable[[tuple[str, ...]], str]
GpuProbe = Callable[[], GpuMemory]
RamProbe = Callable[[], float]


def parse_nvidia_smi_memory(text: str) -> list[GpuMemory]:
    """Parse `memory.free,memory.total` csv (noheader, nounits, MiB), one line per GPU."""
    gpus: list[GpuMemory] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 2:
            raise GpuProbeError(f"unparsable nvidia-smi memory line: {line!r}")
        try:
            free_mib, total_mib = float(cells[0]), float(cells[1])
        except ValueError as exc:
            raise GpuProbeError(f"unparsable nvidia-smi memory line: {line!r}") from exc
        if not (0.0 <= free_mib <= total_mib) or total_mib <= 0.0:
            raise GpuProbeError(f"implausible nvidia-smi memory line: {line!r}")
        gpus.append(GpuMemory(free_gb=free_mib / _MIB_PER_GB, total_gb=total_mib / _MIB_PER_GB))
    if not gpus:
        raise GpuProbeError("nvidia-smi printed no GPU memory lines")
    return gpus


def parse_meminfo_available_gb(text: str) -> float:
    """`MemAvailable` from /proc/meminfo text (kB), in GB."""
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() != "MemAvailable":
            continue
        parts = rest.split()
        if len(parts) != 2 or parts[1] != "kB" or not parts[0].isdigit():
            raise GpuProbeError(f"unparsable /proc/meminfo line: {line!r}")
        return int(parts[0]) / _KB_PER_GB
    raise GpuProbeError("/proc/meminfo has no MemAvailable line")


def run_nvidia_smi(args: tuple[str, ...]) -> str:
    """Run the local `nvidia-smi` binary. A missing binary or other OS failure is a
    `GpuProbeError` (deployment fault); a timeout or non-zero exit is a retryable
    `GpuProbeTransientError` (NVML busy or a GPU reset in progress)."""
    try:
        done = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30
        )
    except FileNotFoundError as exc:
        raise GpuProbeError("nvidia-smi not found: cannot measure free VRAM") from exc
    except subprocess.TimeoutExpired as exc:
        raise GpuProbeTransientError(f"nvidia-smi timed out after {exc.timeout} s") from exc
    except subprocess.CalledProcessError as exc:
        raise GpuProbeTransientError(
            f"nvidia-smi exited {exc.returncode}: {(exc.stderr or '').strip()}"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuProbeError(f"nvidia-smi failed: {exc}") from exc
    return done.stdout


def query_gpu_memory(run: SmiRunner = run_nvidia_smi, gpu_index: int = 0) -> GpuMemory:
    """Free and total memory of GPU `gpu_index` (nvidia-smi order)."""
    gpus = parse_nvidia_smi_memory(run(NVIDIA_SMI_MEMORY_ARGS))
    if not 0 <= gpu_index < len(gpus):
        raise GpuProbeError(f"GPU index {gpu_index} not present ({len(gpus)} GPU(s) reported)")
    return gpus[gpu_index]


TextReader = Callable[[Path], str | None]
"""Reads a small text file; None when it does not exist, `GpuProbeError` when unreadable."""


def read_text_if_present(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise GpuProbeError(f"cannot read {path}: {exc}") from exc


def _cgroup_bytes(text: str, path: Path) -> int:
    value = text.strip()
    if not value.isdigit():
        raise GpuProbeError(f"unparsable {path}: {value!r}")
    return int(value)


def query_host_available_gb(
    meminfo: Path = MEMINFO_PATH,
    *,
    cgroup_max: Path = CGROUP_MEMORY_MAX,
    cgroup_current: Path = CGROUP_MEMORY_CURRENT,
    read: TextReader = read_text_if_present,
) -> float:
    """Available RAM in GB: host MemAvailable, capped by the container's cgroup v2 headroom
    (`memory.max` - `memory.current`) when both files exist and `memory.max` is not "max"."""
    text = read(meminfo)
    if text is None:
        raise GpuProbeError(f"cannot read {meminfo}: not found")
    available = parse_meminfo_available_gb(text)
    limit_text, current_text = read(cgroup_max), read(cgroup_current)
    if limit_text is None or current_text is None or limit_text.strip() == "max":
        return available
    limit = _cgroup_bytes(limit_text, cgroup_max)
    current = _cgroup_bytes(current_text, cgroup_current)
    return min(available, max(limit - current, 0) / _BYTES_PER_GB)


def _shortfall(
    need_vram_gb: float,
    need_ram_gb: float,
    margin_gb: float,
    gpu_probe: GpuProbe,
    ram_probe: RamProbe,
) -> tuple[Headroom, InsufficientGpuMemoryError | InsufficientHostMemoryError | None]:
    free_vram = gpu_probe().free_gb
    free_ram = ram_probe() if need_ram_gb > 0.0 else None
    headroom = Headroom(free_vram_gb=free_vram, free_ram_gb=free_ram)
    if free_vram < need_vram_gb + margin_gb:
        return headroom, InsufficientGpuMemoryError(
            f"free VRAM {free_vram:.2f} GB < need {need_vram_gb:.2f} GB + margin {margin_gb:.2f} GB"
        )
    if free_ram is not None and free_ram < need_ram_gb + margin_gb:
        return headroom, InsufficientHostMemoryError(
            f"available RAM {free_ram:.2f} GB < need {need_ram_gb:.2f} GB + margin "
            f"{margin_gb:.2f} GB"
        )
    return headroom, None


def check_gpu_headroom(
    need_vram_gb: float,
    need_ram_gb: float,
    margin_gb: float = DEFAULT_MARGIN_GB,
    *,
    wait_s: float = 0.0,
    poll_s: float = 1.0,
    gpu_probe: GpuProbe = query_gpu_memory,
    ram_probe: RamProbe = query_host_available_gb,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Headroom:
    """Pass when free VRAM >= need_vram + margin and (if `need_ram_gb` > 0) available RAM >=
    need_ram + margin. With `wait_s` > 0, re-probe every `poll_s` until headroom appears or the
    deadline passes (a released model's VRAM can take a moment to come back), then raise the
    last shortfall. Probe failures raise `GpuProbeError` at once; they are not waited out."""
    if min(need_vram_gb, need_ram_gb, margin_gb, wait_s) < 0.0 or poll_s <= 0.0:
        raise ConfigError(
            f"invalid preflight arguments: need_vram_gb={need_vram_gb} need_ram_gb={need_ram_gb} "
            f"margin_gb={margin_gb} wait_s={wait_s} poll_s={poll_s}"
        )
    deadline = clock() + wait_s
    while True:
        headroom, error = _shortfall(need_vram_gb, need_ram_gb, margin_gb, gpu_probe, ram_probe)
        if error is None:
            return headroom
        remaining = deadline - clock()
        if remaining <= 0.0:
            raise error
        sleep(min(poll_s, remaining))


class HeadroomCheck(Protocol):
    """The call contract of `check_gpu_headroom` that injecting callers rely on (its other
    parameters keep their defaults)."""

    def __call__(
        self, need_vram_gb: float, need_ram_gb: float, /, *, wait_s: float
    ) -> Headroom: ...


def check_workload_headroom(
    workload: GpuWorkload, *, check: HeadroomCheck = check_gpu_headroom, wait_s: float = 0.0
) -> Headroom:
    """Run the preflight with `workload`'s row of `MEMORY_NEEDS`."""
    need = MEMORY_NEEDS[workload]
    return check(need.vram_gb, need.ram_gb, wait_s=wait_s)
