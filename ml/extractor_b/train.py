"""CLI for the Extractor B fine-tune: `--new-run`, `--resume auto|<step>` and `--probe`, with
guards, GPU lock, fingerprinted resume and exit codes 0 (done or paused), 2 (typed
TrainingError) and 3 (no ladder rung fits). Heavy deps load only on the GPU path."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from apx.common.errors import ApxError
from ml.extractor_b.checkpoints import atomic_write_bytes, compute_fingerprint
from ml.extractor_b.config import (
    DEFAULT_CONFIG_PATH,
    check_serving_rung,
    load_config,
    rung_spec,
)
from ml.extractor_b.guards import (
    compute_page1_phashes,
    guard_outputs_current,
    parse_raw_rows,
    raw_stamp,
    run_guards,
    write_guard_outputs,
)
from ml.extractor_b.interfaces import (
    DataLeakError,
    GuardReport,
    LadderRung,
    RunFingerprint,
    SampleSource,
    TrainConfig,
    TrainingError,
    TrainingSample,
)
from ml.extractor_b.loop import ResumeMode, assert_offline_env

EXIT_OK: Final[int] = 0
EXIT_TRAINING_ERROR: Final[int] = 2
EXIT_NO_RUNG: Final[int] = 3
EXIT_RUNG_CHANGE: Final[int] = 4
"""The probe's first fitting rung differs from `SERVING_RUNG` (ADR-0012 §5)."""
GPU_LOCK_PATH: Final[Path] = Path("data/ml/gpu.lock")
SPLITS_PATH: Final[Path] = Path("evals/splits.json")
MODELS_LOCK_PATH: Final[Path] = Path("infra/models.lock.json")
BASE_INTENT: Final[str] = "extractor_b_train_base"


@dataclass(frozen=True, slots=True)
class Args:
    mode: ResumeMode | None
    probe: bool
    rung: LadderRung | None
    config: Path


def _resume_arg(value: str) -> ResumeMode:
    if value == "auto":
        return "auto"
    if value.isdigit():
        return int(value)
    raise argparse.ArgumentTypeError("--resume takes 'auto' or a step number")


def parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.train", description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--probe", action="store_true", help="walk the VRAM ladder, write probe.json"
    )
    group.add_argument("--new-run", action="store_true", help="start a new run directory")
    group.add_argument(
        "--resume", type=_resume_arg, metavar="auto|STEP", help="resume the newest run"
    )
    parser.add_argument(
        "--rung", type=LadderRung, choices=list(LadderRung), help="override the config rung"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ns = parser.parse_args(argv)
    mode: ResumeMode | None = "new" if ns.new_run else ns.resume
    return Args(mode=mode, probe=bool(ns.probe), rung=ns.rung, config=ns.config)


def effective_config(args: Args) -> TrainConfig:
    cfg = load_config(args.config)
    return cfg if args.rung is None else cfg.model_copy(update={"rung": args.rung})


def train_split_sha256(splits_path: Path | None = None) -> frozenset[str]:
    """sha256s of the train pool of `evals/splits.json` (the only documents training may see).
    A splits file with any cross-pool overlap is itself a leak and stops the run."""
    from pydantic import ValidationError

    from evals.splits import POOL_MEMBERS, load_splits, validate_splits

    splits_path = SPLITS_PATH if splits_path is None else splits_path
    try:
        splits = load_splits(splits_path)
    except (OSError, ValueError, ValidationError) as exc:
        raise TrainingError(f"cannot read {splits_path}: {exc}") from exc
    if problems := validate_splits(splits):
        raise DataLeakError(f"{splits_path} is not leak-free: {problems[0]}")
    return frozenset(
        sha
        for set_name, split_name in POOL_MEMBERS["train"]
        for sha in splits.sets.get(set_name, {}).get(split_name, [])
    )


def prepare_data(
    cfg: TrainConfig, *, train_shas: frozenset[str], data_root: Path = Path("data")
) -> tuple[list[TrainingSample], GuardReport]:
    """Run the pre-run guards on `raw_train.jsonl` + `raw_dev.jsonl` unless their outputs are
    current; returns the kept rows and the report. CORD rows are checked against their raw
    `gt_parse` from `data_root`'s CORD train index (ADR-0019 amendment 1)."""
    data_dir = cfg.data_dir
    raw = [data_dir / "raw_train.jsonl", data_dir / "raw_dev.jsonl"]
    for p in raw:
        if not p.is_file():
            raise TrainingError(f"missing {p}; build it with the data-prep step first")
    if not guard_outputs_current(data_dir, raw):
        lines = [line for p in raw for line in p.read_text(encoding="utf-8").splitlines()]
        rows, invalid = parse_raw_rows(lines)
        cord_trees = None
        if any(r.source is SampleSource.CORD for r in rows):
            from ml.extractor_b.data import cord_trees as load_cord_trees

            cord_trees = load_cord_trees(data_root)
        kept, quarantined, report = run_guards(
            rows,
            train_split_sha256=train_shas,
            page1_phash=compute_page1_phashes(rows, data_dir),
            cfg=cfg.guards,
            max_seq_len=rung_spec(cfg).max_seq_len,
            image_root=data_dir,
            quarantine_path=data_dir / "quarantine.jsonl",
            pre_quarantined=invalid,
            cord_trees=cord_trees,
        )
        write_guard_outputs(data_dir, kept, quarantined, report)
        atomic_write_bytes(data_dir / "raw.sha256", raw_stamp(raw).encode("ascii"))
    kept = [
        TrainingSample.model_validate_json(line)
        for line in (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = GuardReport.model_validate_json((data_dir / "guard_report.json").read_bytes())
    return kept, report


@dataclass(frozen=True, slots=True)
class BaseCheckpoint:
    """The HF safetensors base the LoRA trains on: local directory and pinned Hub revision."""

    path: Path
    revision: str


def resolve_base(env: Mapping[str, str], lock_path: Path | None = None) -> BaseCheckpoint:
    """`EXTRACTOR_B_BASE_PATH` / `EXTRACTOR_B_BASE_REVISION` when set; otherwise derived from the
    `extractor_b_train_base` intent in `infra/models.lock.json` (path `models/<repo_id>`,
    revision `revision_sha`). Never guessed: no env and no lock entry is a `TrainingError`."""
    path = env.get("EXTRACTOR_B_BASE_PATH", "").strip()
    revision = env.get("EXTRACTOR_B_BASE_REVISION", "").strip()
    if not (path and revision):
        entry = _lock_entry(MODELS_LOCK_PATH if lock_path is None else lock_path)
        if entry is None:
            raise TrainingError(
                f"set EXTRACTOR_B_BASE_PATH and EXTRACTOR_B_BASE_REVISION, or add the "
                f"{BASE_INTENT!r} intent to infra/models.yaml and resolve it into the lock"
            )
        path = path or f"models/{entry['repo_id']}"
        revision = revision or str(entry["revision_sha"])
    return BaseCheckpoint(Path(path), revision)


def _lock_entry(lock_path: Path) -> dict[str, Any] | None:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read {lock_path}: {exc}") from exc
    for entry in lock.get("models", []):
        if (
            entry.get("intent") == BASE_INTENT
            and entry.get("repo_id")
            and entry.get("revision_sha")
        ):
            return dict(entry)
    return None


def fingerprint_for(cfg: TrainConfig, env: dict[str, str]) -> RunFingerprint:
    return compute_fingerprint(
        cfg,
        manifest_path=cfg.data_dir / "manifest.jsonl",
        quarantine_path=cfg.data_dir / "quarantine.jsonl",
        base_revision=resolve_base(env).revision,
        image_digest=_required_env(env, "TRAINER_IMAGE_DIGEST"),
        git_sha=env.get("GIT_SHA", "unknown"),
    )


def _required_env(env: dict[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise TrainingError(f"{name} must be set so the run fingerprint is complete")
    return value


def run(
    args: Args, *, env: dict[str, str], gpu_job: Callable[[Args, TrainConfig, dict[str, str]], int]
) -> int:
    """Steps 1-3 of the run flow (config, offline env, guards) then the GPU job under the lock."""
    from ml.extractor_b.gpu import gpu_lock

    cfg = effective_config(args)
    if not args.probe:
        check_serving_rung(cfg)
    assert_offline_env(env)
    job = "extractor_b_probe" if args.probe else "extractor_b_train"
    with gpu_lock(cfg.vram, lock_path=GPU_LOCK_PATH, job=job):
        prepare_data(cfg, train_shas=train_split_sha256())
        return gpu_job(args, cfg, env)


def _gpu_job(args: Args, cfg: TrainConfig, env: dict[str, str]) -> int:
    """Model load, trainer and probe live in `ml.extractor_b.gpu_job` (trainer image only)."""
    module = importlib.import_module("ml.extractor_b.gpu_job")
    result: int = module.run(args, cfg, env)
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    gpu_job: Callable[[Args, TrainConfig, dict[str, str]], int] = _gpu_job,
) -> int:
    args = parse_args(argv)
    try:
        return run(args, env=dict(os.environ), gpu_job=gpu_job)
    except TrainingError as exc:
        print(f"extractor_b.train: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_TRAINING_ERROR
    except ApxError as exc:  # ConfigError and friends: also a typed stop, never exit 0
        print(f"extractor_b.train: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_TRAINING_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
