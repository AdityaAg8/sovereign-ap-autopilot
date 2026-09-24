"""Tests for the Extractor B GPU preflight and `gpu.lock` (ADR-0011 §6) with a fake nvidia-smi:
busy memory or a compute process refuse, a legacy pid-only lock is reported with the manual step
and never taken over, and the lock is released after the block."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.extractor_b.gpu import (
    GpuLockInfo,
    gpu_lock,
    pid_alive,
    preflight_gpu,
    query_used_mib,
    read_lock,
    release_lock,
)
from ml.extractor_b.interfaces import GpuBusyError, TrainingError, VramConfig

CFG = VramConfig()


def fake_smi(used: str = "312\n", apps: str = "") -> Callable[[list[str]], str]:
    def run(args: list[str]) -> str:
        return used if args[0].startswith("--query-gpu") else apps

    return run


def test_idle_gpu_takes_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "ml" / "gpu.lock"
    preflight_gpu(CFG, lock_path=lock, job="train", smi=fake_smi())
    info = read_lock(lock)
    assert info is not None and info.pid == os.getpid() and info.job == "train"


def test_memory_in_use_is_busy(tmp_path: Path) -> None:
    with pytest.raises(GpuBusyError, match="MiB in use"):
        preflight_gpu(CFG, lock_path=tmp_path / "l", job="t", smi=fake_smi("9000\n"))
    assert not (tmp_path / "l").exists()


def test_compute_process_is_busy(tmp_path: Path) -> None:
    with pytest.raises(GpuBusyError, match="llama-server"):
        preflight_gpu(
            CFG, lock_path=tmp_path / "l", job="t", smi=fake_smi(apps="4242, llama-server\n")
        )


def test_legacy_lock_with_live_pid_is_busy(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.write_text(
        GpuLockInfo(pid=1234, job="eval", started_at=datetime.now(UTC)).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(GpuBusyError, match=r"legacy lock .*job 'eval' pid 1234.*looks alive"):
        preflight_gpu(CFG, lock_path=lock, job="t", smi=fake_smi(), alive=lambda pid: True)


def test_legacy_lock_with_dead_pid_is_reported_not_taken(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    original = GpuLockInfo(pid=999_999, job="eval", started_at=datetime.now(UTC)).model_dump_json()
    lock.write_text(original, encoding="utf-8")
    with pytest.raises(GpuBusyError, match=r"not running.*never taken over"):
        preflight_gpu(CFG, lock_path=lock, job="t", smi=fake_smi(), alive=lambda pid: False)
    assert lock.read_text(encoding="utf-8") == original


def test_unreadable_lock_with_a_held_flock_is_busy(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.write_text("garbage", encoding="utf-8")
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a live holder still writing metadata
        with pytest.raises(GpuBusyError, match="unreadable"):
            preflight_gpu(CFG, lock_path=lock, job="t", smi=fake_smi())
    finally:
        os.close(fd)
    assert lock.read_text(encoding="utf-8") == "garbage"


def test_bad_or_empty_smi_output_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(TrainingError):
        preflight_gpu(CFG, lock_path=tmp_path / "l", job="t", smi=fake_smi("N/A\n"))
    with pytest.raises(TrainingError):
        preflight_gpu(CFG, lock_path=tmp_path / "l", job="t", smi=fake_smi(""))


def test_context_manager_releases_even_on_error(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    with (
        pytest.raises(RuntimeError),
        gpu_lock(CFG, lock_path=lock, job="t", smi=fake_smi()) as info,
    ):
        assert info.job == "t" and lock.exists()
        raise RuntimeError("boom")
    assert not lock.exists()


def test_release_only_by_owner(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    preflight_gpu(CFG, lock_path=lock, job="t", smi=fake_smi(), pid=4321)
    assert not release_lock(lock)
    assert release_lock(lock, pid=4321) and not lock.exists()


def test_query_used_mib_parses_multi_gpu() -> None:
    assert query_used_mib(fake_smi("10\n 2048 \n")) == [10.0, 2048.0]


def test_pid_alive_for_self_and_nonsense() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
