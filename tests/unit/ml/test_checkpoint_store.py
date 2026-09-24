"""Tests for the atomic Extractor B checkpoint store (ADR-0011 §3): torn, tampered or unsealed
checkpoints are never chosen, the best is never pruned, and retention keeps newest + best."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.extractor_b.checkpoints import (
    COMPLETE_NAME,
    BestPointer,
    FsCheckpointStore,
    atomic_write_json,
    retention_plan,
)
from ml.extractor_b.interfaces import CheckpointCorruptError, CheckpointMeta, TrainingError
from tests.unit.ml.conftest import make_fingerprint

NOW = datetime(2026, 9, 23, tzinfo=UTC)


def _store(tmp_path: Path, keep_last: int = 2) -> FsCheckpointStore:
    return FsCheckpointStore(tmp_path / "run", fingerprint=make_fingerprint(), keep_last=keep_last)


def _meta(step: int) -> CheckpointMeta:
    return CheckpointMeta(
        step=step, epoch=step / 10, created_at=NOW, fingerprint=make_fingerprint(), files={}
    )


def _save(store: FsCheckpointStore, step: int, payload: bytes = b"weights") -> Path:
    out = store.begin(step)
    (out / "adapter_model.safetensors").write_bytes(payload + str(step).encode())
    (out / "nested").mkdir()
    (out / "nested" / "rng_state.pth").write_bytes(b"rng")
    return store.commit(step, _meta(step))


def test_commit_then_verify(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = _save(store, 25)
    meta = store.verify(path)
    assert meta is not None and meta.step == 25
    assert set(meta.files) == {"adapter_model.safetensors", "nested/rng_state.pth"}
    assert store.latest_verified() == meta
    assert not list(store.root.glob("*.partial"))


def test_flipped_byte_is_rejected_and_older_one_chosen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, 25)
    newest = _save(store, 50)
    blob = newest / "adapter_model.safetensors"
    data = bytearray(blob.read_bytes())
    data[0] ^= 0x01
    blob.write_bytes(bytes(data))
    assert store.verify(newest) is None
    latest = store.latest_verified()
    assert latest is not None and latest.step == 25


@pytest.mark.parametrize("damage", ["missing_file", "missing_complete", "extra_file", "bad_json"])
def test_damaged_checkpoint_is_never_latest(tmp_path: Path, damage: str) -> None:
    store = _store(tmp_path)
    _save(store, 25)
    newest = _save(store, 50)
    if damage == "missing_file":
        (newest / "nested" / "rng_state.pth").unlink()
    elif damage == "missing_complete":
        (newest / COMPLETE_NAME).unlink()
    elif damage == "extra_file":
        (newest / "sneaky.bin").write_bytes(b"x")
    else:
        (newest / COMPLETE_NAME).write_text("{not json", encoding="utf-8")
    latest = store.latest_verified()
    assert latest is not None and latest.step == 25


def test_leftover_partial_is_never_latest_and_is_cleaned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, 25)
    torn = store.begin(50)
    (torn / "adapter_model.safetensors").write_bytes(b"half")
    latest = store.latest_verified()
    assert latest is not None and latest.step == 25
    assert store.clean_stale_partials() == [torn]
    assert not torn.exists()


def test_renamed_directory_with_wrong_step_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = _save(store, 25)
    os.rename(path, store.path_for(75))
    assert store.verify(store.path_for(75)) is None


def test_other_run_fingerprint_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = _save(store, 25)
    other = FsCheckpointStore(
        store.run_dir, fingerprint=make_fingerprint(base_revision="rev-9"), keep_last=2
    )
    assert other.verify(path) is None
    with pytest.raises(CheckpointCorruptError):
        other.verify_step(25)


def test_commit_guards(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(TrainingError):
        store.commit(25, _meta(25))
    store.begin(25)
    with pytest.raises(TrainingError):
        store.commit(25, _meta(26))
    bad = _meta(25).model_copy(update={"fingerprint": make_fingerprint(image_digest="x")})
    with pytest.raises(TrainingError):
        store.commit(25, bad)


def test_resave_of_same_step_replaces_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, 25, b"old")
    _save(store, 25, b"new")
    assert (store.path_for(25) / "adapter_model.safetensors").read_bytes() == b"new25"
    assert store.verified_steps() == [25]


def test_retention_keeps_newest_two_and_best(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, 25)
    store.set_best(25)
    for step in (50, 75, 100, 125):
        _save(store, step)
    assert store.verified_steps() == [25, 100, 125]
    best = store.best()
    assert best is not None and best.step == 25


def test_set_best_requires_a_verified_checkpoint(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(CheckpointCorruptError):
        store.set_best(25)
    assert store.best() is None and store.best_step() is None


def test_invalid_best_pointer_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.run_dir.mkdir(parents=True)
    (store.run_dir / "best.json").write_text('{"step": "x"}', encoding="utf-8")
    with pytest.raises(CheckpointCorruptError):
        store.best_step()


def test_atomic_write_json_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "a" / "best.json"
    atomic_write_json(target, BestPointer(step=3))
    atomic_write_json(target, BestPointer(step=4))
    assert BestPointer.model_validate_json(target.read_bytes()).step == 4
    assert [p.name for p in target.parent.iterdir()] == ["best.json"]


@given(
    steps=st.sets(st.integers(0, 10_000), max_size=30),
    best=st.one_of(st.none(), st.integers(0, 10_000)),
    keep=st.integers(0, 5),
)
def test_retention_plan_property(steps: set[int], best: int | None, keep: int) -> None:
    doomed = retention_plan(sorted(steps), best_step=best, keep_last=keep)
    kept = steps - set(doomed)
    newest = set(sorted(steps)[len(steps) - keep :]) if keep else set()
    expected = newest | ({best} & steps)
    assert kept == expected
    assert set(doomed) <= steps
    assert best not in doomed


def test_retention_plan_rejects_negative_keep() -> None:
    with pytest.raises(ValueError):
        retention_plan([1], best_step=None, keep_last=-1)


@given(payloads=st.lists(st.binary(min_size=0, max_size=64), min_size=1, max_size=5))
def test_commit_verify_round_trip(
    tmp_path_factory: pytest.TempPathFactory, payloads: list[bytes]
) -> None:
    store = FsCheckpointStore(
        tmp_path_factory.mktemp("rt") / "run", fingerprint=make_fingerprint(), keep_last=2
    )
    out = store.begin(10)
    for n, blob in enumerate(payloads):
        (out / f"f{n}.bin").write_bytes(blob)
    path = store.commit(10, _meta(10))
    meta = store.verify(path)
    assert meta is not None and len(meta.files) == len(payloads)
    assert all((path / f"f{n}.bin").read_bytes() == b for n, b in enumerate(payloads))


class _Crash(Exception):
    """Stands in for a process kill at one point of a re-save."""


def _crash_on(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rename_src: str | None = None,
    rmtree_name: str | None = None,
) -> None:
    import shutil

    real_rename, real_rmtree = Path.rename, shutil.rmtree

    def rename(self: Path, target: Path) -> Path:
        if rename_src is not None and self.name == rename_src:
            raise _Crash(self.name)
        return real_rename(self, target)

    def rmtree(path: Path, *a: object, **kw: object) -> None:
        if rmtree_name is not None and Path(path).name == rmtree_name:
            raise _Crash(str(path))
        real_rmtree(path)

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr("ml.extractor_b.checkpoints.shutil.rmtree", rmtree)


def _resave_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **crash: str
) -> FsCheckpointStore:
    store = _store(tmp_path)
    _save(store, 25, b"old")
    store.set_best(25)
    with monkeypatch.context() as m:
        _crash_on(m, **crash)
        with pytest.raises(_Crash):
            _save(store, 25, b"new")
    return _store(tmp_path)  # a fresh process


def _payload(store: FsCheckpointStore) -> bytes:
    return (store.path_for(25) / "adapter_model.safetensors").read_bytes()


def test_crash_after_old_moved_aside_restores_the_old_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _resave_crashing(tmp_path, monkeypatch, rename_src="checkpoint-000025.partial")
    assert not store.path_for(25).exists() and store.replaced_path_for(25).is_dir()
    store.clean_stale_partials()
    assert store.verified_steps() == [25] and _payload(store) == b"old25"
    assert store.best() is not None
    assert sorted(p.name for p in store.root.iterdir()) == ["checkpoint-000025"]


def test_crash_before_old_is_deleted_keeps_the_new_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _resave_crashing(tmp_path, monkeypatch, rmtree_name="checkpoint-000025.replaced")
    assert store.replaced_path_for(25).is_dir()
    store.clean_stale_partials()
    assert store.verified_steps() == [25] and _payload(store) == b"new25"
    assert sorted(p.name for p in store.root.iterdir()) == ["checkpoint-000025"]


def test_crash_before_old_is_moved_aside_keeps_the_old_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _resave_crashing(tmp_path, monkeypatch, rename_src="checkpoint-000025")
    store.clean_stale_partials()
    assert store.verified_steps() == [25] and _payload(store) == b"old25"
    assert sorted(p.name for p in store.root.iterdir()) == ["checkpoint-000025"]


def test_next_commit_settles_a_leftover_replaced_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _resave_crashing(tmp_path, monkeypatch, rename_src="checkpoint-000025.partial")
    _save(store, 25, b"newer")  # no cleanup ran first
    assert _payload(store) == b"newer25" and not store.replaced_path_for(25).exists()


def test_failed_verification_after_resave_puts_the_old_step_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _save(store, 25, b"old")
    real_verify = FsCheckpointStore.verify
    calls = {"n": 0}

    def flaky(self: FsCheckpointStore, path: Path) -> CheckpointMeta | None:
        calls["n"] += 1
        return None if calls["n"] == 1 else real_verify(self, path)

    monkeypatch.setattr(FsCheckpointStore, "verify", flaky)
    with pytest.raises(CheckpointCorruptError):
        _save(store, 25, b"new")
    assert _payload(store) == b"old25" and not store.replaced_path_for(25).exists()


def test_replaced_dir_next_to_a_corrupt_final_is_left_for_a_human(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _save(store, 25, b"old")
    store.path_for(25).rename(store.replaced_path_for(25))
    _save(store, 30)
    bad = store.path_for(25)
    bad.mkdir()
    (bad / "junk").write_bytes(b"x")
    with pytest.raises(CheckpointCorruptError):
        store.clean_stale_partials()
    assert store.replaced_path_for(25).is_dir()
