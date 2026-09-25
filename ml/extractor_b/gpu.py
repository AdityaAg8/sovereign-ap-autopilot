"""GPU exclusivity for the Extractor B run (ADR-0011 §6): `nvidia-smi` preflight (memory in use,
compute processes) and the O_EXCL `data/ml/gpu.lock` that one GPU job holds, with a kernel flock
kept for the whole job. A lock whose flock is free is stale and reclaimed atomically; a held one is
refused. Torch-free; `nvidia-smi`, the identity readers and the file locker are injectable, so
the lock logic is tested on any host."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from ml.extractor_b.interfaces import GpuBusyError, TrainingError, VramConfig

log = logging.getLogger(__name__)
SmiRunner = Callable[[list[str]], str]
_HINT: Final[str] = "run `make gpu-free` and retry"
_MIB_PER_GIB: Final[float] = 1024.0
LOCK_FORMAT: Final[int] = 2
_CONTAINER_ID_RE: Final[re.Pattern[str]] = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
_TRAINER_CHECK: Final[str] = "docker ps --filter name=extractor-b-trainer"

# Open fds of the locks this process holds (path -> fd); the flock lives as long as the fd.
_held: dict[str, int] = {}
_held_mutex = threading.Lock()


class HolderIdentity(BaseModel):
    """Where the lock holder runs (boot, host name, container). For messages only: liveness is
    the kernel flock, which works across containers that bind-mount the same `data/ml`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    boot_id: str | None
    hostname: str
    container_id: str | None


class GpuLockInfo(BaseModel):
    """Contents of `gpu.lock`. `format` 1 (pid, job, created-at) predates the holder flock and
    identity; `format` 2 holders keep an exclusive flock on the file until they release it."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    pid: int
    job: str
    started_at: datetime
    format: int = 1
    holder: HolderIdentity | None = None


class FlockUnavailableError(TrainingError):
    """flock(2) was used on a host that has none (Windows). There `DEFAULT_LOCKER` is None
    instead, so a lock is never reclaimed (ADR-0011 amendment 4)."""


class FileLocker(Protocol):
    """Exclusive whole-file locks with flock(2) semantics: held per open file (not per thread or
    pid) and dropped when that fd closes or its holder dies. The seam lets tests drive the lock
    logic with a flock-equivalent on hosts that have no flock."""

    def try_lock(self, fd: int) -> bool:
        """Take the lock without blocking; False when another open file holds it."""
        ...

    def lock(self, fd: int) -> None:
        """Take the lock, blocking until it is free."""
        ...


class FcntlLocker:
    """The production locker: `fcntl.flock`, imported on use so this module loads on Windows,
    where every call raises `FlockUnavailableError`."""

    def try_lock(self, fd: int) -> bool:
        if sys.platform == "win32":
            raise FlockUnavailableError("flock(2) is unavailable on Windows")
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def lock(self, fd: int) -> None:
        if sys.platform == "win32":
            raise FlockUnavailableError("flock(2) is unavailable on Windows")
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


# flock(2) is the liveness proof; without it (None: Windows) a lock is never reclaimed.
DEFAULT_LOCKER: Final[FileLocker | None] = None if sys.platform == "win32" else FcntlLocker()


class LockState(StrEnum):
    HELD = "held"  # a live holder keeps the flock (or we cannot prove it does not)
    STALE = "stale"  # format 2 with a free flock: its holder is gone, reclaimed automatically
    MANUAL = "manual"  # legacy, or unreadable without flock: never reclaimed, message says how


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def read_boot_id() -> str | None:
    text = _read_text("/proc/sys/kernel/random/boot_id")
    return (text.strip() or None) if text is not None else None


def read_container_id() -> str | None:
    """The 64-hex container id from `/proc/self/cgroup` (cgroup v1, or v2 without a cgroup
    namespace), else from `/proc/self/mountinfo` (Docker's per-container hostname/resolv.conf
    bind mounts). None when neither names one, e.g. on the host."""
    for path in ("/proc/self/cgroup", "/proc/self/mountinfo"):
        text = _read_text(path)
        if text and (match := _CONTAINER_ID_RE.search(text)):
            return match.group(1)
    return None


@dataclass(frozen=True)
class ProcProbe:
    """The readers that describe the holder in lock messages; injected so tests are
    deterministic."""

    boot_id: Callable[[], str | None] = read_boot_id
    hostname: Callable[[], str] = socket.gethostname
    container_id: Callable[[], str | None] = read_container_id

    def identity(self) -> HolderIdentity:
        return HolderIdentity(
            boot_id=self.boot_id(), hostname=self.hostname(), container_id=self.container_id()
        )


DEFAULT_PROBE: Final[ProcProbe] = ProcProbe()


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


def flock_is_free(lock_path: Path, locker: FileLocker | None = DEFAULT_LOCKER) -> bool | None:
    """Whether nobody holds an flock on `lock_path`: a non-blocking LOCK_EX that succeeds (and is
    dropped at once). None when flock is unavailable or the file cannot be opened."""
    if locker is None:
        return None
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except OSError:
        return None
    try:
        return locker.try_lock(fd)
    except OSError:
        return None
    finally:
        os.close(fd)  # also drops our probe flock


def classify_lock(
    info: GpuLockInfo | None, *, flock_free: bool | None, alive: Callable[[int], bool]
) -> tuple[LockState, str]:
    """Decide whether a lock is still held, with the reason.

    Format 2: the holder keeps an exclusive flock for the whole job and the kernel drops it when
    the holder dies (reboot included), so a free flock is stale and a held one is HELD whatever
    the host, container or pid says: each `make train-b` runs in a fresh container where the
    holder is pid 1. Unknown (no flock support) is HELD.

    Format 1 (pid only, no flock) is never reclaimed: its pid comes from an unknown pid
    namespace (pid 1 always looks alive in a container) and no age limit is safe for runs that
    last days. It needs a one-time manual delete; every lock written now is format 2.

    Unreadable (empty or partial metadata): a holder that crashed between the O_EXCL create and
    the metadata write leaves such a file. Callers classify under the acquire guard, which a live
    writer holds from create to fsync, so a free flock here means that holder is gone: STALE. A
    held flock is HELD, and unknown flock support stays MANUAL."""
    if info is None:
        if flock_free is None:
            return LockState.MANUAL, "unreadable"
        if flock_free:
            return LockState.STALE, "unreadable and no process holds its flock"
        return LockState.HELD, "unreadable but a process holds its flock"
    if info.format < LOCK_FORMAT or info.holder is None:
        if alive(info.pid):
            return LockState.MANUAL, "legacy lock whose pid looks alive in this namespace"
        return LockState.MANUAL, "legacy lock whose pid is not running in this namespace"
    if flock_free is None:
        return LockState.HELD, "flock unavailable, liveness cannot be proven"
    if flock_free:
        return LockState.STALE, "no process holds its flock"
    return LockState.HELD, "its holder keeps the flock"


def _manual_step(lock_path: Path) -> str:
    return (
        f"confirm no trainer runs (`{_TRAINER_CHECK}` is empty and no train job on the host), "
        f"then `rm {lock_path}`; `make gpu-free` does not clear it"
    )


def _busy_message(lock_path: Path, info: GpuLockInfo | None, state: LockState, reason: str) -> str:
    if info is None:
        if state is LockState.HELD:
            return f"{lock_path} is unreadable ({reason}); wait for it to finish, or {_HINT}"
        return f"{lock_path} exists but is unreadable; {_manual_step(lock_path)}"
    if state is LockState.MANUAL:
        return (
            f"{lock_path} is a legacy lock (job {info.job!r} pid {info.pid}, started "
            f"{info.started_at:%Y-%m-%d %H:%M}Z, {reason}); it is never taken over: "
            f"{_manual_step(lock_path)}"
        )
    where = f" on {info.holder.hostname}" if info.holder else ""
    return (
        f"{lock_path} held by job {info.job!r} pid {info.pid}{where} "
        f"(since {info.started_at:%Y-%m-%d %H:%M}Z, {reason}); wait for it to finish, or "
        f"{_HINT}"
    )


def _inspect(
    lock_path: Path, alive: Callable[[int], bool], locker: FileLocker | None
) -> tuple[GpuLockInfo | None, LockState, str]:
    info = read_lock(lock_path)
    state, reason = classify_lock(info, flock_free=flock_is_free(lock_path, locker), alive=alive)
    return info, state, reason


@contextlib.contextmanager
def _acquire_guard(lock_path: Path, locker: FileLocker | None) -> Iterator[None]:
    """Serialize every inspect/create/reclaim/release of `lock_path` with a blocking flock on a
    sibling `.guard` file, held only for those few syscalls. The kernel drops it when its holder
    dies, so it never goes stale; the file stays in place on purpose (unlinking flock files
    reintroduces the race). A no-op without flock, where nothing is ever reclaimed."""
    if locker is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path.with_name(lock_path.name + ".guard"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        locker.lock(fd)
        yield
    finally:
        os.close(fd)


def _create_and_hold(lock_path: Path, info: GpuLockInfo, locker: FileLocker | None) -> None:
    """O_EXCL-create the lock, take its flock, then write the metadata. The fd stays open (and
    the flock held) until `release_lock` or process exit. O_CLOEXEC is the default, so exec'd
    children do not inherit it; forked workers do, which keeps the lock held while they run."""
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        if locker is not None and not locker.try_lock(fd):
            raise GpuBusyError(f"{lock_path} was just created but another process holds its flock")
        os.write(fd, info.model_dump_json().encode("utf-8"))
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
        raise
    with _held_mutex:
        _held[str(lock_path)] = fd


def _acquire(
    lock_path: Path, info: GpuLockInfo, alive: Callable[[int], bool], locker: FileLocker | None
) -> None:
    """Under the guard: create the lock, or re-check an existing one and reclaim it when stale,
    so two racing reclaimers never both hold it (the loser sees the winner's flock)."""
    with _acquire_guard(lock_path, locker):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _create_and_hold(lock_path, info, locker)
            return
        except FileExistsError:
            pass
        old, state, reason = _inspect(lock_path, alive, locker)
        if state is not LockState.STALE:
            raise GpuBusyError(_busy_message(lock_path, old, state, reason))
        if old is None:
            log.warning("reclaiming stale GPU lock %s (no metadata): %s", lock_path, reason)
        else:
            log.warning(
                "reclaiming stale GPU lock %s (job %r pid %d on %s, started %s): %s",
                lock_path,
                old.job,
                old.pid,
                old.holder.hostname if old.holder else "?",
                old.started_at.isoformat(),
                reason,
            )
        lock_path.unlink()
        try:
            _create_and_hold(lock_path, info, locker)
        except FileExistsError as exc:  # an unguarded (legacy) writer got in; never overwrite
            again, state, reason = _inspect(lock_path, alive, locker)
            raise GpuBusyError(_busy_message(lock_path, again, state, reason)) from exc


def preflight_gpu(
    cfg: VramConfig,
    *,
    lock_path: Path,
    job: str,
    smi: SmiRunner | None = None,
    alive: Callable[[int], bool] = pid_alive,
    pid: int | None = None,
    probe: ProcProbe = DEFAULT_PROBE,
    locker: FileLocker | None = DEFAULT_LOCKER,
) -> None:
    """Raise `GpuBusyError` unless nvidia-smi shows < `gpu_idle_max_used_gib` used, no compute
    process, and `lock_path` can be taken: created with O_EXCL, or reclaimed when its flock is
    free (`classify_lock`). On success this process holds the lock and its flock until
    `release_lock`. A held lock refuses before nvidia-smi runs. nvidia-smi is the general "GPU
    is free" check (llama-server and friends), not a liveness proof: the flock is that. With no
    `locker` (no flock) a free path is still taken but an existing lock is never reclaimed."""
    with _acquire_guard(lock_path, locker):
        if lock_path.exists():
            old, state, reason = _inspect(lock_path, alive, locker)
            if state is not LockState.STALE:
                raise GpuBusyError(_busy_message(lock_path, old, state, reason))
    used = query_used_mib(smi)
    if not used:
        raise TrainingError("nvidia-smi reported no GPU")
    limit_mib = cfg.gpu_idle_max_used_gib * _MIB_PER_GIB
    if max(used) >= limit_mib:
        raise GpuBusyError(f"GPU has {max(used):.0f} MiB in use (limit {limit_mib:.0f}); {_HINT}")
    if procs := query_compute_pids(smi):
        raise GpuBusyError(f"GPU has compute processes: {'; '.join(procs)}; {_HINT}")

    info = GpuLockInfo(
        pid=os.getpid() if pid is None else pid,
        job=job,
        started_at=datetime.now(UTC),
        format=LOCK_FORMAT,
        holder=probe.identity(),
    )
    _acquire(lock_path, info, alive, locker)


def _close_held(lock_path: Path) -> None:
    with _held_mutex:
        fd = _held.pop(str(lock_path), None)
    if fd is not None:
        os.close(fd)


def release_lock(
    lock_path: Path,
    *,
    pid: int | None = None,
    probe: ProcProbe = DEFAULT_PROBE,
    locker: FileLocker | None = DEFAULT_LOCKER,
) -> bool:
    """Under the guard, remove the lock only if this process (or `pid`) owns it: same pid and,
    for a format 2 lock, the same holder identity (pid 1 in another container is not us). The
    holder flock is dropped last, after the unlink. Returns whether it was removed.

    Windows refuses to unlink a file that has an open handle, so there the fd is closed first.
    Nobody can observe that gap: every inspect and reclaim runs under the guard held here, and a
    host without a locker never reclaims at all."""
    owner = os.getpid() if pid is None else pid
    with _acquire_guard(lock_path, locker):
        info = read_lock(lock_path)
        if info is None or info.pid != owner:
            return False
        if info.holder is not None and info.holder != probe.identity():
            return False
        if sys.platform == "win32":
            _close_held(lock_path)
            lock_path.unlink(missing_ok=True)
        else:
            lock_path.unlink(missing_ok=True)
            _close_held(lock_path)
    return True


@contextlib.contextmanager
def gpu_lock(
    cfg: VramConfig,
    *,
    lock_path: Path,
    job: str,
    smi: SmiRunner | None = None,
    alive: Callable[[int], bool] = pid_alive,
    probe: ProcProbe = DEFAULT_PROBE,
    locker: FileLocker | None = DEFAULT_LOCKER,
) -> Iterator[GpuLockInfo]:
    """Preflight and hold `gpu.lock` (and its flock) for the block; released in `finally`."""
    preflight_gpu(
        cfg, lock_path=lock_path, job=job, smi=smi, alive=alive, probe=probe, locker=locker
    )
    try:
        info = read_lock(lock_path)
        if info is None:
            raise TrainingError(f"{lock_path} vanished right after it was created")
        yield info
    finally:
        release_lock(lock_path, probe=probe, locker=locker)
