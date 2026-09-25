"""Atomic, verifiable checkpoints for the Extractor B run (ADR-0011 §3-4): `.partial` dirs sealed
by a `COMPLETE.json` of per-file sha256s, the `best.json` pointer, retention and the run
fingerprint. Torch-free; the trainer only writes payload files into the dir `begin` returns."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from ml.extractor_b.config import config_sha256
from ml.extractor_b.interfaces import (
    CheckpointCorruptError,
    CheckpointMeta,
    RunFingerprint,
    TrainConfig,
    TrainingError,
)

log = logging.getLogger(__name__)

COMPLETE_NAME: Final[str] = "COMPLETE.json"
BEST_NAME: Final[str] = "best.json"
_FINAL_RE: Final[re.Pattern[str]] = re.compile(r"^checkpoint-(\d{6,})$")
_REPLACED_SUFFIX: Final = ".replaced"
_REPLACED_RE: Final[re.Pattern[str]] = re.compile(r"^checkpoint-(\d{6,})\.replaced$")
_CHUNK: Final[int] = 1 << 20


class BestPointer(BaseModel):
    """Contents of `best.json`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    step: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    """Persist a rename. Windows cannot open a directory for fsync; NTFS journals the rename."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to `<path>.tmp`, fsync, `os.replace` onto `path`, fsync the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, model: BaseModel) -> None:
    atomic_write_bytes(path, model.model_dump_json(indent=2).encode("utf-8"))


def retention_plan(
    verified_steps: Sequence[int], *, best_step: int | None, keep_last: int
) -> list[int]:
    """Steps to delete: everything except the `keep_last` newest and the best."""
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    steps = sorted(set(verified_steps))
    # Newest first, so keep_last above the count keeps them all (a `len - keep_last` start would
    # go negative and wrap around, pruning newer checkpoints).
    keep = set(steps[::-1][:keep_last])
    if best_step is not None:
        keep.add(best_step)
    return [s for s in steps if s not in keep]


def compute_fingerprint(
    cfg: TrainConfig,
    *,
    manifest_path: Path,
    quarantine_path: Path,
    base_revision: str,
    image_digest: str,
    git_sha: str,
) -> RunFingerprint:
    """Hash the canonical config JSON and both files; the fingerprint is stable across runs."""
    return RunFingerprint(
        config_sha256=config_sha256(cfg),
        manifest_sha256=sha256_file(manifest_path),
        quarantine_sha256=sha256_file(quarantine_path),
        base_revision=base_revision,
        image_digest=image_digest,
        rung=cfg.rung,
        git_sha=git_sha,
    )


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() != COMPLETE_NAME
    }


class FsCheckpointStore:
    """`CheckpointStore` on a local (or bind-mounted) filesystem under `<run_dir>/checkpoints/`.
    A checkpoint counts only if `COMPLETE.json` parses, names this directory's step, carries the
    run's fingerprint and every listed file (and no other) matches its sha256."""

    def __init__(self, run_dir: Path, *, fingerprint: RunFingerprint, keep_last: int) -> None:
        self.run_dir = run_dir
        self.root = run_dir / "checkpoints"
        self.fingerprint = fingerprint
        self.keep_last = keep_last

    def path_for(self, step: int) -> Path:
        return self.root / f"checkpoint-{step:06d}"

    def begin(self, step: int) -> Path:
        partial = self.root / f"checkpoint-{step:06d}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        return partial

    def commit(self, step: int, meta: CheckpointMeta) -> Path:
        """Hash and fsync the payload, write `COMPLETE.json` last, rename into place, verify,
        then prune. `meta.files` is recomputed here, so callers pass it empty."""
        partial = self.root / f"checkpoint-{step:06d}.partial"
        if not partial.is_dir():
            raise TrainingError(f"commit of step {step} without begin(): {partial} is missing")
        if meta.step != step:
            raise TrainingError(f"commit step {step} != meta.step {meta.step}")
        if meta.fingerprint.comparable() != self.fingerprint.comparable():
            raise TrainingError("checkpoint meta carries a different run fingerprint")
        for p in partial.rglob("*"):
            if p.is_file():
                _fsync_file(p)
        sealed = meta.model_copy(update={"files": _hash_tree(partial)})
        atomic_write_json(partial / COMPLETE_NAME, sealed)
        _fsync_dir(partial)

        final = self.path_for(step)
        aside = self.replaced_path_for(step)
        if aside.exists():  # an older interrupted re-save: settle it before touching `final`
            self._settle_replaced(step)
        if final.exists():  # re-save of a step after resume: keep the old one until verified
            final.rename(aside)
            _fsync_dir(self.root)
        partial.rename(final)
        _fsync_dir(self.root)
        if self.verify(final) is None:
            if aside.exists():
                shutil.rmtree(final)
                aside.rename(final)
                _fsync_dir(self.root)
            raise CheckpointCorruptError(
                f"checkpoint {final} failed verification right after commit"
            )
        if aside.exists():
            shutil.rmtree(aside)
        self.prune()
        return final

    def replaced_path_for(self, step: int) -> Path:
        return self.root / f"checkpoint-{step:06d}{_REPLACED_SUFFIX}"

    def _settle_replaced(self, step: int) -> None:
        """Finish an interrupted re-save of `step`: restore the old copy when no final dir exists,
        drop it when the final one verifies, else leave both for a human (never guess)."""
        aside, final = self.replaced_path_for(step), self.path_for(step)
        if not final.exists():
            log.warning("restoring %s interrupted during a re-save", aside)
            aside.rename(final)
            _fsync_dir(self.root)
        elif self.verify(final) is not None:
            shutil.rmtree(aside)
        else:
            raise CheckpointCorruptError(
                f"{final} fails verification and {aside} still exists; inspect by hand"
            )

    def verify(self, path: Path) -> CheckpointMeta | None:
        match = _FINAL_RE.match(path.name)
        complete = path / COMPLETE_NAME
        if match is None or not complete.is_file():
            return None
        try:
            meta = CheckpointMeta.model_validate_json(complete.read_bytes())
        except (ValidationError, OSError):
            return None
        if meta.step != int(match.group(1)):
            return None
        if meta.fingerprint.comparable() != self.fingerprint.comparable():
            return None
        try:
            actual = _hash_tree(path)
        except OSError:
            return None
        return meta if actual == meta.files else None

    def _final_dirs(self) -> list[tuple[int, Path]]:
        if not self.root.is_dir():
            return []
        found = [
            (int(m.group(1)), p)
            for p in self.root.iterdir()
            if p.is_dir() and (m := _FINAL_RE.match(p.name))
        ]
        return sorted(found, reverse=True)

    def verified_steps(self) -> list[int]:
        return sorted(step for step, path in self._final_dirs() if self.verify(path) is not None)

    def latest_verified(self) -> CheckpointMeta | None:
        for step, path in self._final_dirs():
            meta = self.verify(path)
            if meta is not None:
                return meta
            log.warning("rejected checkpoint step %d at %s: failed verification", step, path)
        return None

    def verify_step(self, step: int) -> CheckpointMeta:
        """For `--resume <step>`: the pinned checkpoint or `CheckpointCorruptError`."""
        meta = self.verify(self.path_for(step))
        if meta is None:
            raise CheckpointCorruptError(f"pinned checkpoint step {step} is missing or corrupt")
        return meta

    def set_best(self, step: int) -> None:
        if self.verify(self.path_for(step)) is None:
            raise CheckpointCorruptError(f"best.json cannot name unverified step {step}")
        atomic_write_json(self.run_dir / BEST_NAME, BestPointer(step=step))

    def best_step(self) -> int | None:
        path = self.run_dir / BEST_NAME
        if not path.is_file():
            return None
        try:
            return BestPointer.model_validate_json(path.read_bytes()).step
        except ValidationError as exc:
            raise CheckpointCorruptError(f"{path} is not a valid best pointer") from exc

    def best(self) -> CheckpointMeta | None:
        step = self.best_step()
        return None if step is None else self.verify(self.path_for(step))

    def prune(self) -> list[int]:
        """Delete verified checkpoints outside `keep_last` newest + best. Unverified final dirs
        are left for a human to inspect."""
        verified = self.verified_steps()
        doomed = retention_plan(verified, best_step=self.best_step(), keep_last=self.keep_last)
        for step in doomed:
            shutil.rmtree(self.path_for(step))
        return doomed

    def clean_stale_partials(self) -> list[Path]:
        """Startup cleanup: settle interrupted re-saves first (a `.replaced` dir may be the only
        copy of its step), then delete leftover `.partial` dirs."""
        if not self.root.is_dir():
            return []
        for p in sorted(self.root.iterdir()):
            match = _REPLACED_RE.match(p.name)
            if match is not None and p.is_dir():
                self._settle_replaced(int(match.group(1)))
        stale = sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.endswith(".partial"))
        for p in stale:
            log.warning("removing stale partial checkpoint %s", p)
            shutil.rmtree(p)
        return stale
