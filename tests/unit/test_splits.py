"""Unit tests for evals/splits.py: the evals/splits.json pydantic contract, building fatura/
extractbench/docile groups from a manifest, `register_documents`'s no-reshuffle guarantee, and
the duplicate/leak checkers (PRD §5, §6.2 I5, ADR-0005). No network, no GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evals.datasets.manifest import DatasetEntry, DatasetStatus, Manifest, ManifestFile, Split
from evals.splits import (
    SplitsFile,
    build_from_manifest,
    empty_splits,
    find_duplicate_documents,
    find_leaks,
    load_splits,
    register_documents,
    save_splits,
    validate_splits,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _manifest_file(sha256: str, split: Split | None, doc_id: str | None) -> ManifestFile:
    return ManifestFile(path=f"data/x/{sha256}", sha256=sha256, bytes=1, split=split, doc_id=doc_id)


def test_empty_splits_has_every_known_set_and_split_empty() -> None:
    splits = empty_splits()
    assert splits.sets["uae"] == {"train": [], "cal": [], "test": []}
    assert splits.sets["docile"] == {"train": [], "val": []}
    assert splits.sets["fatura"] == {"train": [], "test": []}
    assert splits.sets["extractbench"] == {"eval": []}
    assert splits.sets["attacks"] == {"eval": []}


def test_load_splits_returns_empty_shape_when_file_missing(tmp_path: Path) -> None:
    splits = load_splits(tmp_path / "does-not-exist.json")
    assert splits.sets["fatura"] == {"train": [], "test": []}


def test_save_then_load_round_trips_and_sorts_and_dedupes(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    splits = SplitsFile(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        sets={
            "uae": {"train": [], "cal": [], "test": []},
            "docile": {"train": [], "val": []},
            "fatura": {"train": [SHA_B, SHA_A, SHA_A], "test": [SHA_C]},
            "extractbench": {"eval": []},
            "attacks": {"eval": []},
        },
    )

    written = save_splits(splits, path)

    assert written == path
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    reloaded = load_splits(path)
    assert reloaded.sets["fatura"]["train"] == [SHA_A, SHA_B]  # sorted + de-duplicated


def test_build_from_manifest_extracts_fatura_and_extractbench_per_doc_splits() -> None:
    manifest = Manifest(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        datasets={
            "fatura": DatasetEntry(
                status=DatasetStatus.PRESENT,
                source="https://zenodo.org/records/8261508",
                version="8261508",
                license="CC-BY-4.0",
                files=[
                    _manifest_file(SHA_A, None, None),  # the zip itself: not a document
                    _manifest_file(SHA_B, "train", "doc-b"),
                    _manifest_file(SHA_C, "test", "doc-c"),
                ],
            ),
            "extractbench": DatasetEntry(
                status=DatasetStatus.PRESENT,
                source="https://huggingface.co/datasets/llamaindex/ExtractBench",
                version="short",
                license="Apache-2.0",
                files=[_manifest_file(SHA_D, "eval", "doc-d")],
            ),
        },
    )

    built = build_from_manifest(manifest)

    assert built.sets["fatura"] == {"train": [SHA_B], "test": [SHA_C]}
    assert built.sets["extractbench"] == {"eval": [SHA_D]}
    assert built.sets["docile"] == {"train": [], "val": []}  # deferred: no docile entry
    assert built.sets["uae"] == {"train": [], "cal": [], "test": []}


def test_build_from_manifest_preserves_uae_and_attacks_from_existing() -> None:
    existing = SplitsFile(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        sets={
            "uae": {"train": [SHA_A], "cal": [SHA_B], "test": [SHA_C]},
            "docile": {"train": [], "val": []},
            "fatura": {"train": [], "test": []},
            "extractbench": {"eval": []},
            "attacks": {"eval": [SHA_D]},
        },
    )
    manifest = Manifest(generated_at=existing.generated_at, datasets={})

    built = build_from_manifest(manifest, existing)

    assert built.sets["uae"] == {"train": [SHA_A], "cal": [SHA_B], "test": [SHA_C]}
    assert built.sets["attacks"] == {"eval": [SHA_D]}


def test_register_documents_adds_new_sha_without_reshuffling_existing() -> None:
    splits = empty_splits()
    splits = register_documents(splits, "uae", {SHA_A: "train", SHA_B: "cal"})

    # Registering SHA_A again under a *different* split must not move it: no reshuffling.
    reregistered = register_documents(splits, "uae", {SHA_A: "test", SHA_C: "test"})

    assert reregistered.sets["uae"]["train"] == [SHA_A]
    assert reregistered.sets["uae"]["cal"] == [SHA_B]
    assert reregistered.sets["uae"]["test"] == [SHA_C]


def test_register_documents_rejects_unknown_split_name() -> None:
    splits = empty_splits()
    try:
        register_documents(splits, "fatura", {SHA_A: "cal"})
    except ValueError as exc:
        assert "cal" in str(exc)
    else:
        raise AssertionError("expected ValueError for a split name fatura does not have")


def test_find_duplicate_documents_flags_repeated_sha_within_one_split() -> None:
    splits = SplitsFile(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        sets={
            "uae": {"train": [], "cal": [], "test": []},
            "docile": {"train": [], "val": []},
            "fatura": {"train": [SHA_A, SHA_A], "test": []},
            "extractbench": {"eval": []},
            "attacks": {"eval": []},
        },
    )
    problems = find_duplicate_documents(splits)
    assert any(SHA_A in p and "fatura.train" in p for p in problems)


def test_find_leaks_flags_sha_shared_between_train_and_test_pools() -> None:
    splits = SplitsFile(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        sets={
            "uae": {"train": [], "cal": [], "test": []},
            "docile": {"train": [], "val": []},
            "fatura": {"train": [SHA_A], "test": [SHA_A]},
            "extractbench": {"eval": []},
            "attacks": {"eval": []},
        },
    )
    problems = find_leaks(splits)
    assert any(SHA_A in p and "'train'" in p and "'test'" in p for p in problems)


def test_find_leaks_flags_sha_shared_between_cal_and_extractbench_eval() -> None:
    splits = SplitsFile(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        sets={
            "uae": {"train": [], "cal": [SHA_A], "test": []},
            "docile": {"train": [], "val": []},
            "fatura": {"train": [], "test": []},
            "extractbench": {"eval": [SHA_A]},
            "attacks": {"eval": []},
        },
    )
    assert find_leaks(splits) != []


def test_validate_splits_is_clean_for_disjoint_pools() -> None:
    splits = SplitsFile(
        generated_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        sets={
            "uae": {"train": [SHA_A], "cal": [SHA_B], "test": [SHA_C]},
            "docile": {"train": [], "val": [SHA_D]},
            "fatura": {"train": [], "test": []},
            "extractbench": {"eval": []},
            "attacks": {"eval": []},
        },
    )
    assert validate_splits(splits) == []


def test_splits_json_round_trip_matches_pydantic_contract(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    save_splits(empty_splits(now=datetime.fromisoformat("2026-01-01T00:00:00+00:00")), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw["sets"]) == {"uae", "docile", "fatura", "extractbench", "attacks"}
