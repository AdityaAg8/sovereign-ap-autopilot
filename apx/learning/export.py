"""Retraining export (PRD §10 M10.1, §6.2 I5): turns reviewer corrections in the `labels`
table into a JSON-lines training manifest for `ml/`, reading only corrections already
assigned to the training split. The query below is the only thing that decides which rows are
eligible; `_require_train_split` re-checks every fetched row by identity against
`LabelSplit.TRAIN` before it is written out, so a future change to the query (or to this
function) that widens what gets exported fails loudly instead of silently leaking eval data
into training.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apx.common.types import InvoiceId, UserId
from apx.db.enums import LabelSplit
from apx.db.models import Label as LabelRow
from apx.learning.errors import SplitViolationError
from apx.learning.interfaces import TrainingManifestRow


def _require_train_split(row: LabelRow) -> None:
    """Raises `SplitViolationError` unless `row.split` is exactly the training split (I5).
    Kept as its own function so it can be unit tested directly against a row that never
    touched the database, independent of whether the export query's `WHERE` clause is doing
    its job."""
    if row.split is not LabelSplit.TRAIN:
        raise SplitViolationError(
            f"label {row.id} is split={row.split!r}, not the training split; refusing to export it"
        )


def _to_manifest_row(row: LabelRow) -> TrainingManifestRow:
    _require_train_split(row)
    assert row.split is not None  # narrowed by _require_train_split above (mypy cannot see it)
    return TrainingManifestRow(
        invoice_id=InvoiceId(row.invoice_id),
        field_name=row.field_name,
        corrected_value=row.corrected_value,
        corrected_normalized=row.corrected_normalized,
        split=row.split,
        labeled_by=UserId(row.labeled_by) if row.labeled_by else None,
        created_at=row.created_at,
    )


async def export_training_manifest(session: AsyncSession, *, out_path: Path) -> int:
    """Writes one JSON object per line to `out_path`, one per training-split correction that
    has a non-null corrected value, ordered by id for a reproducible file. Returns the number
    of rows written."""
    rows = (
        await session.execute(
            select(LabelRow)
            .where(LabelRow.split == LabelSplit.TRAIN)
            .where(LabelRow.corrected_value.is_not(None))
            .order_by(LabelRow.id)
        )
    ).scalars()

    manifest_rows = [_to_manifest_row(row) for row in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for manifest_row in manifest_rows:
            handle.write(manifest_row.model_dump_json())
            handle.write("\n")
    return len(manifest_rows)
