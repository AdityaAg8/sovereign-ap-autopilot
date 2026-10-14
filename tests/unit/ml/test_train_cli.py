"""Tests for the Extractor B training CLI up to the GPU boundary: argument parsing,
exit code 2 for typed errors, the offline-env check, the guards run under the GPU lock, and the
fingerprint inputs. The GPU job itself is replaced by a recording stub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.extractor_b import train
from ml.extractor_b.interfaces import LadderRung, TrainConfig, TrainingError
from tests.unit.ml.conftest import RowFactory

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def test_parse_args_modes() -> None:
    assert train.parse_args(["--new-run"]).mode == "new"
    assert train.parse_args(["--resume", "auto"]).mode == "auto"
    assert train.parse_args(["--resume", "125"]).mode == 125
    probe = train.parse_args(["--probe", "--rung", "L2"])
    assert probe.probe and probe.mode is None and probe.rung is LadderRung.L2


@pytest.mark.parametrize("argv", [[], ["--new-run", "--resume", "auto"], ["--resume", "x"]])
def test_parse_args_requires_exactly_one_mode(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        train.parse_args(argv)


def _setup(tmp_path: Path, make_row: RowFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [make_row("a"), make_row("b"), make_row("c", role="dev")]
    data = tmp_path / "data"
    data.mkdir()
    (data / "pages").mkdir()
    for p in (tmp_path / "pages").iterdir():
        (data / "pages" / p.name).write_bytes(p.read_bytes())
    (data / "raw_train.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in rows if r.role == "train"), encoding="utf-8"
    )
    (data / "raw_dev.jsonl").write_text(rows[2].model_dump_json() + "\n", encoding="utf-8")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"data_dir: {data.as_posix()}\nruns_dir: {(tmp_path / 'runs').as_posix()}\n",
        encoding="utf-8",
    )
    splits = {
        "generated_at": "2026-09-23T00:00:00Z",
        "sets": {"uae": {"train": sorted(r.document_sha256 for r in rows), "cal": [], "test": []}},
    }
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    monkeypatch.setattr(train, "SPLITS_PATH", tmp_path / "splits.json")
    monkeypatch.setattr(train, "GPU_LOCK_PATH", tmp_path / "gpu.lock")
    monkeypatch.setattr("ml.extractor_b.gpu.run_nvidia_smi", _idle_smi)
    return cfg_path


def _idle_smi(args: list[str]) -> str:
    return "100\n" if args[0].startswith("--query-gpu") else ""


def test_main_runs_guards_under_lock_then_gpu_job(
    tmp_path: Path, make_row: RowFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _setup(tmp_path, make_row, monkeypatch)
    for k, v in OFFLINE.items():
        monkeypatch.setenv(k, v)
    seen: dict[str, Any] = {}

    def job(args: train.Args, cfg: TrainConfig, env: dict[str, str]) -> int:
        seen["lock"] = (tmp_path / "gpu.lock").exists()
        seen["manifest"] = (cfg.data_dir / "manifest.jsonl").read_text(encoding="utf-8")
        return 0

    assert train.main(["--new-run", "--config", str(cfg_path)], gpu_job=job) == 0
    assert seen["lock"] and not (tmp_path / "gpu.lock").exists()
    assert [json.loads(x)["sample_id"] for x in seen["manifest"].splitlines()] == ["a", "b", "c"]
    manifest = tmp_path / "data" / "manifest.jsonl"
    written = manifest.stat().st_mtime_ns
    # A second run with unchanged raw files reuses the guard outputs instead of rewriting them.
    assert train.main(["--resume", "auto", "--config", str(cfg_path)], gpu_job=job) == 0
    assert manifest.stat().st_mtime_ns == written


def test_main_returns_2_on_typed_errors(
    tmp_path: Path,
    make_row: RowFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _setup(tmp_path, make_row, monkeypatch)
    for k in OFFLINE:
        monkeypatch.delenv(k, raising=False)
    assert train.main(["--new-run", "--config", str(cfg_path)], gpu_job=lambda *a: 0) == 2
    assert "offline env" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("nope: 1\n", encoding="utf-8")
    assert train.main(["--new-run", "--config", str(bad)], gpu_job=lambda *a: 0) == 2


def test_leaky_splits_file_stops(tmp_path: Path) -> None:
    splits = {
        "generated_at": "2026-09-23T00:00:00Z",
        "sets": {"uae": {"train": ["a" * 64], "cal": ["a" * 64], "test": []}},
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(TrainingError, match="leak"):
        train.train_split_sha256(path)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(TrainingError):
        train.train_split_sha256(path)


def test_prepare_data_requires_raw_files(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="raw_train"):
        train.prepare_data(TrainConfig(data_dir=tmp_path), train_shas=frozenset())


def test_fingerprint_needs_base_and_image_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "quarantine.jsonl").write_text("", encoding="utf-8")
    empty_lock = tmp_path / "models.lock.json"
    empty_lock.write_text('{"models": []}', encoding="utf-8")
    monkeypatch.setattr(train, "MODELS_LOCK_PATH", empty_lock)
    cfg = TrainConfig(data_dir=tmp_path)
    with pytest.raises(TrainingError, match="EXTRACTOR_B_BASE_REVISION"):
        train.fingerprint_for(cfg, {"TRAINER_IMAGE_DIGEST": "sha256:x"})  # lock has no base
    env = {"EXTRACTOR_B_BASE_PATH": "models/x", "EXTRACTOR_B_BASE_REVISION": "r1"}
    with pytest.raises(TrainingError, match="TRAINER_IMAGE_DIGEST"):
        train.fingerprint_for(cfg, env)
    fp = train.fingerprint_for(cfg, env | {"TRAINER_IMAGE_DIGEST": "sha256:x"})
    assert fp.base_revision == "r1" and fp.image_digest == "sha256:x" and fp.git_sha == "unknown"


def test_repo_lock_pins_the_extractor_b_train_base() -> None:
    """ADR-0012 §4: the base resolves from the committed lock, not from env overrides."""
    base = train.resolve_base({})
    assert base.path.as_posix() == "models/Qwen/Qwen3.5-4B"
    assert base.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def test_resolve_base_from_models_lock(tmp_path: Path) -> None:
    lock = tmp_path / "models.lock.json"
    entry = {"intent": "extractor_b_train_base", "repo_id": "org/base", "revision_sha": "abc"}
    lock.write_text(json.dumps({"models": [{"intent": "other"}, entry]}), encoding="utf-8")
    base = train.resolve_base({}, lock)
    assert base.path == Path("models/org/base") and base.revision == "abc"
    pinned = train.resolve_base({"EXTRACTOR_B_BASE_PATH": "/m/b"}, lock)
    assert pinned.path == Path("/m/b") and pinned.revision == "abc"
    lock.write_text("{bad", encoding="utf-8")
    with pytest.raises(TrainingError):
        train.resolve_base({}, lock)
