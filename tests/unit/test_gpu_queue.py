"""Unit tests for scripts/gpu_queue.sh, run against a stand-in `make` that records what it ran:
items run in order with one log each while data/ml/gpu.lock holds a format-1 lock that
ml.extractor_b.gpu never reclaims, the first failure stops the queue, a rerun skips finished items,
a held lock or a self-locking target refuses before anything runs, and the lock is released."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ml.extractor_b.gpu import LockState, classify_lock, read_lock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_queue.sh"

# Stand-in for make: appends its argv to ran.txt, copies the lock the queue holds, fails on FAIL.
FAKE_MAKE = """#!/usr/bin/env bash
echo "$*" >> ran.txt
cp data/ml/gpu.lock "seen-$1.lock"
echo "output of $*"
[ "$1" != FAIL ] || { echo boom; exit 3; }
"""


def _bash() -> str:
    found = shutil.which("bash")
    assert found is not None, "bash is required to run scripts/gpu_queue.sh"
    return found


def _queue(tmp_path: Path, *items: str) -> subprocess.CompletedProcess[str]:
    fake = tmp_path / "fake-make.sh"
    fake.write_text(FAKE_MAKE, encoding="utf-8", newline="\n")
    env = {**os.environ, "MAKE": f"{tmp_path.as_posix()}/fake-make.sh"}
    env.pop("GPU_QUEUE_LOCK", None)
    env.pop("GPU_QUEUE_LOGS", None)
    fake.chmod(0o755)
    return subprocess.run(
        [_bash(), SCRIPT.as_posix(), *items],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _ran(tmp_path: Path) -> list[str]:
    ran = tmp_path / "ran.txt"
    return ran.read_text(encoding="utf-8").splitlines() if ran.exists() else []


def test_queue_runs_in_order_logs_each_item_and_releases_the_lock(tmp_path: Path) -> None:
    result = _queue(tmp_path, "eval-b", "cal-rows-b CAL_B_MODEL=extractor-b-challenger")
    assert result.returncode == 0, result.stderr
    assert _ran(tmp_path) == ["eval-b", "cal-rows-b CAL_B_MODEL=extractor-b-challenger"]
    assert "output of eval-b" in (tmp_path / "logs" / "eval-b.log").read_text(encoding="utf-8")
    item_log = tmp_path / "logs" / "cal-rows-b_CAL_B_MODEL_extractor-b-challenger.log"
    assert item_log.is_file()
    assert (tmp_path / "logs" / "gpu_queue" / "eval-b.done").is_file()
    assert not (tmp_path / "data" / "ml" / "gpu.lock").exists()
    # While an item ran, the lock was a valid format-1 lock the trainer never takes over.
    info = read_lock(tmp_path / "seen-eval-b.lock")
    assert info is not None and info.format == 1 and "eval-b" in info.job
    state, _ = classify_lock(info, flock_free=True, alive=lambda _pid: False)
    assert state is LockState.MANUAL


def test_first_failure_stops_the_queue_and_a_rerun_resumes(tmp_path: Path) -> None:
    result = _queue(tmp_path, "eval-b", "FAIL", "cal-rows-a")
    assert result.returncode == 3
    assert "FAIL failed (exit 3)" in result.stderr and "boom" in result.stderr
    assert _ran(tmp_path) == ["eval-b", "FAIL"]
    assert not (tmp_path / "logs" / "gpu_queue" / "FAIL.done").exists()
    assert not (tmp_path / "data" / "ml" / "gpu.lock").exists()

    (tmp_path / "ran.txt").unlink()
    result = _queue(tmp_path, "eval-b", "cal-rows-a")
    assert result.returncode == 0, result.stderr
    assert _ran(tmp_path) == ["cal-rows-a"] and "eval-b: done" in result.stdout


def test_a_held_lock_refuses_before_anything_runs(tmp_path: Path) -> None:
    lock = tmp_path / "data" / "ml" / "gpu.lock"
    lock.parent.mkdir(parents=True)
    held = {"pid": 1, "job": "extractor_b_train", "started_at": "2026-09-27T00:00:00Z"}
    lock.write_text(json.dumps(held), encoding="utf-8")
    result = _queue(tmp_path, "eval-b")
    assert result.returncode == 2 and "is held" in result.stderr
    assert _ran(tmp_path) == []
    assert json.loads(lock.read_text(encoding="utf-8")) == held  # never removed someone else's


@pytest.mark.parametrize("target", ["train-b", "export-b", "retrain"])
def test_self_locking_targets_are_refused(tmp_path: Path, target: str) -> None:
    result = _queue(tmp_path, "eval-b", target)
    assert result.returncode == 2 and "takes data/ml/gpu.lock itself" in result.stderr
    assert _ran(tmp_path) == [] and not (tmp_path / "data" / "ml" / "gpu.lock").exists()
