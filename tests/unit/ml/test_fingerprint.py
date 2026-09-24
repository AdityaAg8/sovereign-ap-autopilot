"""Tests for the Extractor B run fingerprint (ADR-0011 §4): every comparable field changes it,
and the git SHA is recorded but never blocks a resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.extractor_b.checkpoints import compute_fingerprint
from ml.extractor_b.interfaces import LadderRung, TrainConfig
from tests.unit.ml.conftest import make_fingerprint, sha


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_sha256", sha("other")),
        ("manifest_sha256", sha("other")),
        ("quarantine_sha256", sha("other")),
        ("base_revision", "rev-2"),
        ("image_digest", "sha256:other"),
        ("rung", LadderRung.L2),
    ],
)
def test_each_comparable_field_matters(field: str, value: object) -> None:
    assert make_fingerprint().comparable() != make_fingerprint(**{field: value}).comparable()


def test_git_sha_is_not_compared() -> None:
    assert make_fingerprint(git_sha="a").comparable() == make_fingerprint(git_sha="b").comparable()


def _fp(tmp_path: Path, cfg: TrainConfig, manifest: bytes) -> tuple[str, ...]:
    (tmp_path / "m.jsonl").write_bytes(manifest)
    (tmp_path / "q.jsonl").write_bytes(b"")
    return compute_fingerprint(
        cfg,
        manifest_path=tmp_path / "m.jsonl",
        quarantine_path=tmp_path / "q.jsonl",
        base_revision="r",
        image_digest="d",
        git_sha="g",
    ).comparable()


def test_compute_fingerprint_tracks_config_manifest_and_rung(tmp_path: Path) -> None:
    base = _fp(tmp_path, TrainConfig(), b"row\n")
    assert _fp(tmp_path, TrainConfig(), b"row\n") == base
    assert _fp(tmp_path, TrainConfig(), b"row2\n") != base
    assert _fp(tmp_path, TrainConfig(rung=LadderRung.L1), b"row\n") != base
    assert _fp(tmp_path, TrainConfig.model_validate({"optim": {"epochs": 3}}), b"row\n") != base
