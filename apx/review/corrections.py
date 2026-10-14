"""The review corrections an approval covers, derived from an invoice's `fields` rows, and their
canonical sha256 digest. The API computes both when a reviewer approves and records the digest on
the `approvals` row; the workflow's `confirm_review` activity re-derives them to check that a
`review` signal carries exactly the approved corrections and that no edit landed since
(KNOWN_ISSUES KI-02)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from apx.common.types import Sha256Hex
from apx.db.enums import FieldSource
from apx.db.models import Field as FieldRow
from apx.workflows.corrections import (
    SUPPLIER_ADDRESS_FIELD,
    ReviewCorrections,
    corrections_from_fields,
)


def corrections_for_rows(rows: Sequence[FieldRow]) -> ReviewCorrections:
    """`corrections_from_fields` over an invoice's field rows in id order: the latest value of
    each field with a reviewer's row winning, plus the latest model-sourced values, which are
    what the workflow's `Invoice` was built from (a human row never reaches it). Those are the
    normalized values, as `GroundedInvoiceBuilder` reads supplier TRN and currency, except the
    supplier address, which it reads raw (so an unedited address is never a "correction").
    Raises `MissingCorrectionsError` (or a subclass) while a classified value is missing."""
    latest: dict[str, str | None] = {}
    extracted: dict[str, str | None] = {}
    human: set[str] = set()
    for row in rows:
        is_human = row.source is FieldSource.HUMAN
        raw_read = row.name == SUPPLIER_ADDRESS_FIELD
        if not is_human:
            extracted[row.name] = row.value if raw_read else row.normalized_value
        if row.name in human and not is_human:
            continue
        latest[row.name] = row.value if raw_read else (row.normalized_value or row.value)
        if is_human:
            human.add(row.name)
    return corrections_from_fields(latest, extracted=extracted)


def corrections_digest(corrections: ReviewCorrections) -> Sha256Hex:
    """sha256 of `corrections` as canonical JSON (sorted keys, no whitespace, decimals as their
    exact string form). Equal corrections always give the same digest, and a `ReviewCorrections`
    that crossed Temporal's Pydantic converter digests the same as the one the API built, so the
    digest recorded with an approval can be compared with a signal's corrections."""
    canonical = json.dumps(
        corrections.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
