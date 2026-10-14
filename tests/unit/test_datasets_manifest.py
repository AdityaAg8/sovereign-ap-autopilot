"""Unit tests for evals/datasets/manifest.py: the data/manifest.json pydantic contract and its
load/save round trip. No network, no GPU."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.datasets.manifest import (
    DatasetEntry,
    DatasetStatus,
    Manifest,
    ManifestFile,
    load_manifest,
    save_manifest,
)

VALID_SHA256 = "a" * 64


def test_manifest_file_accepts_contract_shape() -> None:
    entry = ManifestFile(
        path="data/fatura/FATURA.zip",
        sha256=VALID_SHA256,
        bytes=1234,
        split="test",
        doc_id=None,
    )
    assert entry.path == "data/fatura/FATURA.zip"
    assert entry.split == "test"


def test_manifest_file_rejects_bad_sha256() -> None:
    with pytest.raises(ValidationError):
        ManifestFile(path="x", sha256="not-a-hash", bytes=1, split=None, doc_id=None)


def test_manifest_file_rejects_negative_bytes() -> None:
    with pytest.raises(ValidationError):
        ManifestFile(path="x", sha256=VALID_SHA256, bytes=-1, split=None, doc_id=None)


def test_manifest_file_rejects_unknown_split() -> None:
    with pytest.raises(ValidationError):
        ManifestFile.model_validate(
            {"path": "x", "sha256": VALID_SHA256, "bytes": 1, "split": "bogus", "doc_id": None}
        )


def test_dataset_entry_defaults_files_to_empty_list() -> None:
    entry = DatasetEntry(
        status=DatasetStatus.DEFERRED,
        source="https://example.org",
        version="1",
        license="MIT",
        blocked="token missing",
    )
    assert entry.files == []
    assert entry.status is DatasetStatus.DEFERRED


def test_load_manifest_returns_empty_manifest_when_file_missing(tmp_path: Path) -> None:
    manifest = load_manifest(tmp_path / "does-not-exist.json")
    assert manifest.datasets == {}


def test_save_then_load_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        datasets={
            "fatura": DatasetEntry(
                status=DatasetStatus.PRESENT,
                source="https://zenodo.org/records/8261508",
                version="8261508",
                license="CC-BY-4.0",
                blocked=None,
                files=[
                    ManifestFile(
                        path="data/fatura/FATURA.zip",
                        sha256=VALID_SHA256,
                        bytes=1000,
                        split=None,
                        doc_id=None,
                    )
                ],
            )
        },
    )

    written = save_manifest(manifest, path)

    assert written == path
    reloaded = load_manifest(path)
    assert reloaded.datasets["fatura"].status is DatasetStatus.PRESENT
    assert reloaded.datasets["fatura"].files[0].sha256 == VALID_SHA256


def test_save_manifest_writes_lf_line_endings_and_pretty_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"), datasets={}
    )

    save_manifest(manifest, path)

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["datasets"] == {}
    assert "generated_at" in parsed
