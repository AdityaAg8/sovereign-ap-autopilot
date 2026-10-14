"""Typed value objects for the corrections store, active-learning candidate selection
and retraining export: what a reviewer's fix looked like, what makes an (invoice, field) pair
worth reviewing next, and one row of the manifest handed to `ml/` training code
(I5)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from apx.common.types import Frozen, InvoiceId, UserId
from apx.db.enums import FieldSource, LabelSplit


class CorrectionRecord(Frozen):
    """One reviewer fix, exactly as recorded by `apx.api.service.DbInvoiceService.edit_field`
    into the `labels` table: invoice, field, before/after value, the grounding box and
    model/revision that produced the value being corrected, the reviewer and the timestamp."""

    invoice_id: InvoiceId
    field_name: str | None
    previous_value: str | None
    corrected_value: str | None
    previous_boxes: tuple[dict[str, Any], ...] | None
    previous_source: FieldSource | None
    previous_model_revision: str | None
    reviewer: UserId | None
    created_at: datetime
    notes: str | None


class ActiveLearningCandidate(Frozen):
    """One (invoice, field) pair ranked for the next round of human review, with the
    deterministic reasons that produced its score (`apx.learning.selection`)."""

    invoice_id: InvoiceId
    field_name: str
    score: float
    low_confidence: bool
    disagreement: bool
    corrected_often: bool


class TrainingManifestRow(Frozen):
    """One row of the retraining export (`apx.learning.export`): a training-split correction
    with a non-null corrected value, ready for `ml/` to consume."""

    invoice_id: InvoiceId
    field_name: str | None
    corrected_value: str | None
    corrected_normalized: str | None
    split: LabelSplit
    labeled_by: UserId | None
    created_at: datetime
