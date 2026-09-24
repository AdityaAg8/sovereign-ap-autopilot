"""Unit tests for `apx.learning.export`'s split guard (PRD §10 M10.1, §6.2 I5): the export
must refuse a non-training-split row on its own, independent of the database query's `WHERE`
clause, and must build a correct manifest row for a genuine training-split correction. No
database is used here (constructs `apx.db.models.Label` rows directly, never flushed); the
end-to-end query + file write is covered by `tests/integration/test_corrections.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apx.db.enums import LabelSplit
from apx.db.models import Label
from apx.learning.errors import SplitViolationError
from apx.learning.export import _require_train_split, _to_manifest_row

_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _label(*, split: LabelSplit | None) -> Label:
    return Label(
        id=1,
        invoice_id="a" * 64,
        field_name="total",
        corrected_value="123.45",
        corrected_normalized="123.45",
        source="review",
        split=split,
        labeled_by="reviewer1",
        notes=None,
        created_at=_CREATED_AT,
    )


def test_require_train_split_accepts_a_training_row() -> None:
    _require_train_split(_label(split=LabelSplit.TRAIN))  # must not raise


def test_require_train_split_rejects_the_calibration_split() -> None:
    with pytest.raises(SplitViolationError, match="not the training split"):
        _require_train_split(_label(split=LabelSplit.CAL))


def test_require_train_split_rejects_the_test_split() -> None:
    with pytest.raises(SplitViolationError, match="not the training split"):
        _require_train_split(_label(split=LabelSplit.TEST))


def test_require_train_split_rejects_an_unassigned_split() -> None:
    with pytest.raises(SplitViolationError, match="not the training split"):
        _require_train_split(_label(split=None))


def test_to_manifest_row_builds_a_row_from_a_training_label() -> None:
    row = _to_manifest_row(_label(split=LabelSplit.TRAIN))

    assert row.invoice_id == "a" * 64
    assert row.field_name == "total"
    assert row.corrected_value == "123.45"
    assert row.split is LabelSplit.TRAIN
    assert row.labeled_by == "reviewer1"
    assert row.created_at == _CREATED_AT


def test_to_manifest_row_refuses_a_non_training_label() -> None:
    with pytest.raises(SplitViolationError):
        _to_manifest_row(_label(split=LabelSplit.CAL))
