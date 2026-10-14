"""Unit tests for `apx.learning.splits` (I5): documents an evaluation set
holds out stay unassigned so their corrections are never exported for training, everything else
trains, and an unreadable splits file fails closed to "unassigned"."""

from __future__ import annotations

import json
from pathlib import Path

from apx.db.enums import LabelSplit
from apx.learning.splits import split_for_document


def _splits(tmp_path: Path) -> Path:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-09-23T00:00:00+00:00",
                "sets": {
                    "uae": {"train": ["t" * 64], "cal": ["c" * 64], "test": ["x" * 64]},
                    "fatura": {"train": [], "cal": [], "test": ["f" * 64]},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_held_out_documents_are_never_assigned_to_training(tmp_path: Path) -> None:
    path = _splits(tmp_path)
    for held_out in ("x" * 64, "f" * 64, "c" * 64):
        assert split_for_document(held_out, path) is None


def test_training_and_unseen_production_documents_train(tmp_path: Path) -> None:
    path = _splits(tmp_path)
    assert split_for_document("t" * 64, path) is LabelSplit.TRAIN
    assert split_for_document("0" * 64, path) is LabelSplit.TRAIN


def test_unreadable_splits_file_fails_closed_to_unassigned(tmp_path: Path) -> None:
    assert split_for_document("0" * 64, tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert split_for_document("0" * 64, bad) is None


def test_the_committed_splits_file_is_readable() -> None:
    """The default path must resolve in the repo, or every production correction would be left
    unassigned and M10.1's "corrections appear in the next manifest" would silently fail."""
    assert split_for_document("0" * 64) is LabelSplit.TRAIN
