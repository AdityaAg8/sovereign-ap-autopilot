"""Tests for `gpu.lock` liveness by kernel flock (ADR-0011 §6, one GPU job at a time): a holder
keeping the flock is held whatever its host or container, a free flock is stale and reclaimed
exactly once (thread and process races, a real killed child), an empty lock left by a crash
before the metadata write is reclaimed when its flock is free and held otherwise, legacy pid-only
locks need the documented manual step, and without flock nothing is ever reclaimed."""

from __future__ import annotations

import contextlib
import fcntl
import logging
import multiprocessing as mp
import multiprocessing.queues as mp_queues
import multiprocessing.synchronize as mp_sync
import os
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ml.extractor_b.gpu as gpu
from ml.extractor_b.gpu import (
    LOCK_FORMAT,
    GpuLockInfo,
    HolderIdentity,
    LockState,
    ProcProbe,
    classify_lock,
    flock_is_free,
    preflight_gpu,
    read_lock,
    release_lock,
)
from ml.extractor_b.interfaces import GpuBusyError, VramConfig

CFG = VramConfig()
BOOT = "boot-a"
HOST = "trainer"
CONTAINER = "c" * 64
OTHER_CONTAINER = "d" * 64
HOLDER_PID = 1  # the trainer is pid 1 inside its container


@pytest.fixture(autouse=True)
def _close_held_locks() -> Iterator[None]:
    yield
    with gpu._held_mutex:
        fds = list(gpu._held.values())
        gpu._held.clear()
    for fd in fds:
        os.close(fd)


def idle_smi(args: list[str]) -> str:
    return "312\n" if args[0].startswith("--query-gpu") else ""


def probe(
    *, boot: str | None = BOOT, host: str = HOST, container: str | None = CONTAINER
) -> ProcProbe:
    return ProcProbe(boot_id=lambda: boot, hostname=lambda: host, container_id=lambda: container)


def write_v2(
    lock: Path,
    *,
    pid: int = HOLDER_PID,
    boot: str | None = BOOT,
    host: str = HOST,
    container: str | None = CONTAINER,
) -> str:
    """A format 2 lock file as its holder wrote it; nobody holds its flock unless `hold` does."""
    raw = GpuLockInfo(
        pid=pid,
        job="eval",
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
        format=LOCK_FORMAT,
        holder=HolderIdentity(boot_id=boot, hostname=host, container_id=container),
    ).model_dump_json()
    lock.write_text(raw, encoding="utf-8")
    return raw


@contextlib.contextmanager
def hold(lock: Path) -> Iterator[None]:
    """Keep an exclusive flock on `lock` through a separate open file, like a live holder."""
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def alive_only(*pids: int) -> Callable[[int], bool]:
    return lambda pid: pid in pids


def take(
    lock: Path, p: ProcProbe | None = None, alive: Callable[[int], bool] | None = None
) -> None:
    preflight_gpu(
        CFG,
        lock_path=lock,
        job="train",
        smi=idle_smi,
        alive=alive or alive_only(HOLDER_PID, os.getpid()),
        probe=p or probe(),
    )


def assert_reclaimed(lock: Path, caplog: pytest.LogCaptureFixture) -> None:
    info = read_lock(lock)
    assert info is not None and info.pid == os.getpid() and info.job == "train"
    assert info.format == LOCK_FORMAT
    assert info.holder == HolderIdentity(boot_id=BOOT, hostname=HOST, container_id=CONTAINER)
    assert flock_is_free(lock) is False  # the new holder keeps the flock
    assert (
        "reclaiming stale GPU lock" in caplog.text and "no process holds its flock" in caplog.text
    )


def test_live_holder_in_another_container_is_held(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    raw = write_v2(lock, host="run-2", container=OTHER_CONTAINER)

    def smi(args: list[str]) -> str:
        raise AssertionError("nvidia-smi must not run while the lock is held")

    with hold(lock), pytest.raises(GpuBusyError, match="held by job 'eval' pid 1 on run-2"):
        preflight_gpu(CFG, lock_path=lock, job="t", smi=smi, probe=probe())
    assert lock.read_text(encoding="utf-8") == raw


def test_live_holder_in_another_boot_record_is_still_held(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    write_v2(lock, boot="boot-other", host="run-2", container=None)
    with hold(lock), pytest.raises(GpuBusyError, match="keeps the flock"):
        take(lock)


def test_same_identity_live_holder_is_held(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    write_v2(lock)
    with hold(lock), pytest.raises(GpuBusyError, match="held by job 'eval'"):
        take(lock)


@pytest.mark.parametrize(
    "holder",
    [
        {},  # same identity, holder died
        {"boot": "boot-before-reboot"},
        {"host": "run-1", "container": OTHER_CONTAINER},
    ],
    ids=["same-identity", "other-boot", "other-container"],
)
def test_free_flock_is_stale_and_reclaimed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, holder: dict[str, str]
) -> None:
    lock = tmp_path / "gpu.lock"
    write_v2(lock, **holder)
    with caplog.at_level(logging.WARNING, logger="ml.extractor_b.gpu"):
        take(lock)
    assert_reclaimed(lock, caplog)
    assert release_lock(lock, probe=probe()) and not lock.exists()


def test_stale_lock_is_kept_when_gpu_is_busy(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    raw = write_v2(lock)

    def busy(args: list[str]) -> str:
        return "9000\n" if args[0].startswith("--query-gpu") else ""

    with pytest.raises(GpuBusyError, match="MiB in use"):
        preflight_gpu(CFG, lock_path=lock, job="t", smi=busy, probe=probe())
    assert lock.read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("alive", [True, False], ids=["pid-alive", "pid-gone"])
def test_legacy_lock_needs_the_manual_step(tmp_path: Path, alive: bool) -> None:
    lock = tmp_path / "gpu.lock"
    raw = GpuLockInfo(pid=1, job="eval", started_at=datetime.now(UTC)).model_dump_json()
    lock.write_text(raw, encoding="utf-8")
    with pytest.raises(GpuBusyError) as err:
        take(lock, alive=alive_only(1) if alive else alive_only())
    msg = str(err.value)
    assert "legacy lock" in msg and "never taken over" in msg
    assert "docker ps --filter name=extractor-b-trainer" in msg and f"rm {lock}" in msg
    assert "`make gpu-free` does not clear it" in msg
    assert lock.read_text(encoding="utf-8") == raw


def test_empty_lock_with_free_flock_is_reclaimed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A holder that crashed between the O_EXCL create and the metadata write left a real,
    empty file whose flock the kernel already dropped: stale, reclaimed under the guard."""
    lock = tmp_path / "gpu.lock"
    lock.touch()
    assert lock.stat().st_size == 0 and read_lock(lock) is None and flock_is_free(lock)
    with caplog.at_level(logging.WARNING, logger=gpu.log.name):
        take(lock)
    info = read_lock(lock)
    assert info is not None and info.pid == os.getpid() and info.format == LOCK_FORMAT
    assert flock_is_free(lock) is False  # we hold it now
    assert "no metadata" in caplog.text


def test_empty_lock_with_held_flock_is_held(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.touch()
    with hold(lock), pytest.raises(GpuBusyError, match=r"unreadable .*holds its flock"):
        take(lock)
    assert lock.read_bytes() == b""


def test_unreadable_lock_without_flock_needs_the_manual_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu, "_HAS_FLOCK", False)
    lock = tmp_path / "gpu.lock"
    lock.touch()
    with pytest.raises(GpuBusyError, match=f"unreadable; confirm no trainer runs .*rm {lock}"):
        take(lock)
    assert lock.read_bytes() == b""


def test_classify_unreadable_by_flock_state() -> None:
    assert classify_lock(None, flock_free=True, alive=alive_only())[0] is LockState.STALE
    assert classify_lock(None, flock_free=False, alive=alive_only())[0] is LockState.HELD
    assert classify_lock(None, flock_free=None, alive=alive_only())[0] is LockState.MANUAL


def test_without_flock_a_stale_lock_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu, "_HAS_FLOCK", False)
    lock = tmp_path / "gpu.lock"
    raw = write_v2(lock)
    with pytest.raises(GpuBusyError, match="flock unavailable"):
        take(lock)
    assert lock.read_text(encoding="utf-8") == raw
    fresh = tmp_path / "fresh.lock"
    take(fresh)  # a free path is still taken, just never reclaimed later
    assert release_lock(fresh, probe=probe()) and not fresh.exists()


def test_unknown_flock_state_is_held() -> None:
    info = GpuLockInfo.model_validate_json(
        GpuLockInfo(
            pid=1,
            job="eval",
            started_at=datetime.now(UTC),
            format=LOCK_FORMAT,
            holder=HolderIdentity(boot_id=BOOT, hostname=HOST, container_id=None),
        ).model_dump_json()
    )
    assert classify_lock(info, flock_free=None, alive=alive_only())[0] is LockState.HELD


def test_release_refuses_same_pid_from_another_container(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    write_v2(lock, pid=os.getpid(), container=OTHER_CONTAINER)
    assert not release_lock(lock, probe=probe())
    assert release_lock(lock, probe=probe(container=OTHER_CONTAINER)) and not lock.exists()


def test_release_drops_the_flock(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    take(lock)
    fd = gpu._held[str(lock)]
    assert release_lock(lock, probe=probe()) and not lock.exists()
    assert str(lock) not in gpu._held
    with pytest.raises(OSError):
        os.fstat(fd)  # closed, so its flock is gone


def test_thread_race_on_stale_lock_has_one_winner(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    write_v2(lock, boot="boot-old")
    n = 8
    barrier = threading.Barrier(n)
    results: list[str] = []
    mutex = threading.Lock()
    p = probe()

    def worker() -> None:
        barrier.wait()
        try:
            take(lock, p)
            outcome = "won"
        except GpuBusyError:
            outcome = "busy"
        with mutex:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["busy"] * (n - 1) + ["won"]
    info = read_lock(lock)
    assert info is not None and info.pid == os.getpid()


def _race_child(
    lock: str,
    start: mp_sync.Barrier,
    done: mp_sync.Barrier,
    out: mp_queues.Queue[tuple[str, int]],
) -> None:
    """Real probe. The winner keeps its flock until every loser has checked the lock."""
    start.wait()
    try:
        preflight_gpu(CFG, lock_path=Path(lock), job="train", smi=idle_smi)
        out.put(("won", os.getpid()))
    except GpuBusyError:
        out.put(("busy", os.getpid()))
    done.wait()


def test_process_race_on_stale_lock_has_one_winner(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    write_v2(lock, boot="boot-before-reboot", host="gone", container=None)
    ctx = mp.get_context("spawn")
    n = 4
    start, done = ctx.Barrier(n), ctx.Barrier(n)
    out: mp_queues.Queue[tuple[str, int]] = ctx.Queue()
    procs = [ctx.Process(target=_race_child, args=(str(lock), start, done, out)) for _ in range(n)]
    for proc in procs:
        proc.start()
    results = [out.get(timeout=60) for _ in range(n)]
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    winners = [pid for outcome, pid in results if outcome == "won"]
    assert len(winners) == 1 and sum(o == "busy" for o, _ in results) == n - 1
    info = read_lock(lock)
    assert info is not None and info.pid == winners[0] and info.format == LOCK_FORMAT
    assert flock_is_free(lock) is True  # the winner exited, so the kernel dropped its flock


def _holding_child(lock: str, ready: mp_sync.Event) -> None:
    preflight_gpu(CFG, lock_path=Path(lock), job="child", smi=idle_smi)
    ready.set()
    threading.Event().wait()  # hold the lock until killed


def test_real_child_holds_then_dies_and_the_lock_is_reclaimed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock = tmp_path / "gpu.lock"
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(target=_holding_child, args=(str(lock), ready))
    child.start()
    try:
        assert ready.wait(timeout=60)
        # A checker in "another container" still sees the live child as the holder.
        with pytest.raises(GpuBusyError, match=r"held by job 'child'.*keeps the flock"):
            take(lock, probe(host="run-2", container=OTHER_CONTAINER))
    finally:
        child.kill()
        child.join(timeout=60)
    with caplog.at_level(logging.WARNING, logger="ml.extractor_b.gpu"):
        take(lock)
    assert_reclaimed(lock, caplog)
