"""Active-learning candidate selection (PRD §10 M10.1): scores every (invoice, field) pair
that has not yet been reviewer-corrected on three deterministic signals -- low model
confidence, extractor A/B disagreement and how often that field name has needed correction
historically -- and ranks them for the next round of human review. The rule is a fixed
weighted sum with no learned or tuned parameters, so a run is exactly reproducible from the
same database state (I6: no threshold here is adjusted to make anything pass).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apx.common.types import InvoiceId
from apx.db.enums import FieldSource
from apx.db.models import Field as FieldRow
from apx.db.models import Label as LabelRow
from apx.learning.interfaces import ActiveLearningCandidate

LOW_CONFIDENCE_WEIGHT = 0.5
DISAGREEMENT_WEIGHT = 0.3
CORRECTED_OFTEN_WEIGHT = 0.2
"""Fixed, documented weights for the active-learning score (sum to 1.0): PRD §10 M10.1 asks
for a "documented, deterministic rule", not a learned one."""

LOW_CONFIDENCE_THRESHOLD = 0.7
"""A field is "low confidence" when its latest non-human value's confidence is below this, or
the value has no recorded confidence at all (treated as maximally uncertain)."""

CORRECTED_OFTEN_THRESHOLD = 3
"""A field name that has been reviewer-corrected at least this many times across all invoices
counts as "corrected-often" (PRD §10 M10.1 row)."""


def score_candidate(*, low_confidence: bool, disagreement: bool, corrected_often: bool) -> float:
    """Deterministic weighted-sum score in `[0, 1]`; higher means "review this next". A pure
    function so property tests can check its shape without a database."""
    return (
        (LOW_CONFIDENCE_WEIGHT if low_confidence else 0.0)
        + (DISAGREEMENT_WEIGHT if disagreement else 0.0)
        + (CORRECTED_OFTEN_WEIGHT if corrected_often else 0.0)
    )


@dataclass(frozen=True, slots=True)
class _FieldSnapshot:
    invoice_id: str
    field_name: str
    confidence: float | None


async def _latest_field_snapshots(session: AsyncSession) -> list[_FieldSnapshot]:
    """The single latest (highest-id) `fields` row per (invoice_id, name), excluding pairs
    whose latest row is already `FieldSource.HUMAN` -- those have already been corrected and
    are not review candidates any more."""
    latest_ids = select(func.max(FieldRow.id)).group_by(FieldRow.invoice_id, FieldRow.name)
    rows = (
        await session.execute(
            select(FieldRow)
            .where(FieldRow.id.in_(latest_ids.scalar_subquery()))
            .where(FieldRow.source != FieldSource.HUMAN)
        )
    ).scalars()
    return [
        _FieldSnapshot(invoice_id=row.invoice_id, field_name=row.name, confidence=row.confidence)
        for row in rows
    ]


async def _disagreement_pairs(session: AsyncSession) -> set[tuple[str, str]]:
    """(invoice_id, field_name) pairs where extractor A and extractor B produced different
    normalized values on the same invoice -- read straight from stored rows, not a fuzzy
    heuristic. Ordered by id so, if a field was re-extracted, the later row wins."""
    a_values: dict[tuple[str, str], str | None] = {}
    for invoice_id, name, normalized_value in (
        await session.execute(
            select(FieldRow.invoice_id, FieldRow.name, FieldRow.normalized_value)
            .where(FieldRow.source == FieldSource.EXTRACTOR_A)
            .order_by(FieldRow.id)
        )
    ).all():
        a_values[(invoice_id, name)] = normalized_value

    disagreements: set[tuple[str, str]] = set()
    for invoice_id, name, normalized_value in (
        await session.execute(
            select(FieldRow.invoice_id, FieldRow.name, FieldRow.normalized_value)
            .where(FieldRow.source == FieldSource.EXTRACTOR_B)
            .order_by(FieldRow.id)
        )
    ).all():
        key = (invoice_id, name)
        if key in a_values and a_values[key] != normalized_value:
            disagreements.add(key)
    return disagreements


async def _corrected_often_field_names(session: AsyncSession) -> set[str]:
    """Field names corrected at least `CORRECTED_OFTEN_THRESHOLD` times across all invoices,
    from the `labels` table."""
    rows = (
        await session.execute(
            select(LabelRow.field_name, func.count(LabelRow.id))
            .where(LabelRow.field_name.is_not(None))
            .group_by(LabelRow.field_name)
            .having(func.count(LabelRow.id) >= CORRECTED_OFTEN_THRESHOLD)
        )
    ).all()
    return {name for name, _count in rows if name is not None}


async def select_active_learning_candidates(
    session: AsyncSession, *, limit: int = 50
) -> list[ActiveLearningCandidate]:
    """Ranks every not-yet-corrected (invoice, field) pair by `score_candidate` and returns
    the top `limit`, ties broken by `(invoice_id, field_name)` ascending so the result is
    fully deterministic given the same database state (PRD §10 M10.1)."""
    snapshots = await _latest_field_snapshots(session)
    disagreements = await _disagreement_pairs(session)
    corrected_often_names = await _corrected_often_field_names(session)

    candidates = []
    for snap in snapshots:
        low_confidence = snap.confidence is None or snap.confidence < LOW_CONFIDENCE_THRESHOLD
        disagreement = (snap.invoice_id, snap.field_name) in disagreements
        corrected_often = snap.field_name in corrected_often_names
        candidates.append(
            ActiveLearningCandidate(
                invoice_id=InvoiceId(snap.invoice_id),
                field_name=snap.field_name,
                score=score_candidate(
                    low_confidence=low_confidence,
                    disagreement=disagreement,
                    corrected_often=corrected_often,
                ),
                low_confidence=low_confidence,
                disagreement=disagreement,
                corrected_often=corrected_often,
            )
        )

    candidates.sort(key=lambda c: (-c.score, c.invoice_id, c.field_name))
    return candidates[:limit]
