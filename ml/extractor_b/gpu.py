"""GPU exclusivity for the Extractor B run (ADR-0011 §6): `nvidia-smi` preflight (memory in use,
compute processes) and the O_EXCL `data/ml/gpu.lock` that one GPU job holds at a time. A stale
lock is reported, never taken over. Torch-free; `nvidia-smi` is injectable for CPU tests."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from ml.extractor_b.interfaces import GpuBusyError, TrainingError, VramConfig

SmiRunner = Callable[[list[str]], str]
_HINT: Final[str] = "run `make gpu-free` and retry"
_MIB_PER_GIB: Final[float] = 1024.0


class GpuLockInfo(BaseModel):
    """Contents of `gpu.lock`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    pid: int
    job: str
    started_at: datetime


def run_nvidia_smi(args: list[str]) -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrainingError(f"nvidia-smi failed, cannot verify the GPU is free: {exc}") from exc
    return done.stdout


def query_used_mib(smi: SmiRunner | None = None) -> list[float]:
    """Used memory per GPU in MiB."""
    out = (smi or run_nvidia_smi)(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return [float(line.strip()) for line in out.splitlines() if line.strip()]
    except ValueError as exc:
        raise TrainingError(f"unparseable nvidia-smi memory output: {out!r}") from exc


def query_compute_pids(smi: SmiRunner | None = None) -> list[str]:
    out = (smi or run_nvidia_smi)(
        ["--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(lock_path: Path) -> GpuLockInfo | None:
    try:
        return GpuLockInfo.model_validate_json(lock_path.read_bytes())
    except (OSError, ValidationError):
        return None


def _lock_busy_message(lock_path: Path, alive: Callable[[int], bool]) -> str:
    info = read_lock(lock_path)
    if info is None:
        return f"{lock_path} exists but is unreadable; inspect and remove it by hand"
    if not alive(info.pid):
        return (
            f"stale {lock_path}: job {info.job!r} pid {info.pid} "
            f"(started {info.started_at:%Y-%m-%d %H:%M}Z) "
            "is not running; confirm and delete the lock by hand, it is never taken over"
        )
    return f"{lock_path} held by job {info.job!r} pid {info.pid}; {_HINT}"


def preflight_gpu(
    cfg: VramConfig,
    *,
    lock_path: Path,
    job: str,
    smi: SmiRunner | None = None,
    alive: Callable[[int], bool] = pid_alive,
    pid: int | None = None,
) -> None:
    """Raise `GpuBusyError` unless nvidia-smi shows < `gpu_idle_max_used_gib` used, no compute
    process, and `lock_path` can be created with O_EXCL (it then holds pid, job, start time)."""
    if lock_path.exists():
        raise GpuBusyError(_lock_busy_message(lock_path, alive))
    used = query_used_mib(smi)
    if not used:
        raise TrainingError("nvidia-smi reported no GPU")
    limit_mib = cfg.gpu_idle_max_used_gib * _MIB_PER_GIB
    if max(used) >= limit_mib:
        raise GpuBusyError(f"GPU has {max(used):.0f} MiB in use (limit {limit_mib:.0f}); {_HINT}")
    if procs := query_compute_pids(smi):
        raise GpuBusyError(f"GPU has compute processes: {'; '.join(procs)}; {_HINT}")

    info = GpuLockInfo(
        pid=os.getpid() if pid is None else pid, job=job, started_at=datetime.now(UTC)
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise GpuBusyError(_lock_busy_message(lock_path, alive)) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(info.model_dump_json())
        fh.flush()
        os.fsync(fh.fileno())


def release_lock(lock_path: Path, *, pid: int | None = None) -> bool:
    """Remove the lock only if this process (or `pid`) owns it. Returns whether it was removed."""
    info = read_lock(lock_path)
    owner = os.getpid() if pid is None else pid
    if info is None or info.pid != owner:
        return False
    lock_path.unlink(missing_ok=True)
    return True


@contextlib.contextmanager
def gpu_lock(
    cfg: VramConfig,
    *,
    lock_path: Path,
    job: str,
    smi: SmiRunner | None = None,
    alive: Callable[[int], bool] = pid_alive,
) -> Iterator[GpuLockInfo]:
    """Preflight and hold `gpu.lock` for the block; released in `finally`."""
    preflight_gpu(cfg, lock_path=lock_path, job=job, smi=smi, alive=alive)
    try:
        info = read_lock(lock_path)
        if info is None:
            raise TrainingError(f"{lock_path} vanished right after it was created")
        yield info
    finally:
        release_lock(lock_path)
