"""Shared primitive types for every apx module: ids, enums, money, bounding boxes, grounded
field values and idempotency keys. Depends on nothing else in apx (PRD §3, §6.2 I1, §6.4)."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, NewType

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""Lowercase hex SHA-256 digest; the identity of a document (§5 splits, §3 dedupe)."""

InvoiceId = NewType("InvoiceId", str)
"""Stable invoice id; equals the source document sha256 for ingested invoices."""

UserId = NewType("UserId", str)
VendorId = NewType("VendorId", str)

IdempotencyKey = Annotated[str, StringConstraints(pattern=r"^[^:\s]+:[a-z][a-z0-9_]*$")]
"""Deterministic key `invoice_id:step` carried by every side-effecting activity (§6.4)."""


class Frozen(BaseModel):
    """Base for immutable value objects passed between pipeline stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Language(StrEnum):
    """Document languages in scope (§1)."""

    EN = "en"
    AR = "ar"


class Role(StrEnum):
    """Human roles (§1). Capabilities per role live in apx.policy."""

    REVIEWER = "reviewer"
    APPROVER = "approver"
    ADMIN = "admin"


class Stage(StrEnum):
    """GPU stages; each maps to one llama-swap group with peak VRAM <= 11.5 GB (§2 C2, §4)."""

    SCAN = "scan"
    PARSE = "parse"
    EXTRACT = "extract"
    ARBITRATE = "arbitrate"
    FRAUD = "fraud"
    FRAUD_PAGE = "fraud_page"
    AGENT = "agent"


class Route(StrEnum):
    """Policy outcome for an invoice (§6.3)."""

    AUTO = "auto"
    REVIEW = "review"
    BLOCK = "block"


class Money(Frozen):
    """Exact monetary amount; never a float on the money path (§8)."""

    amount: Decimal
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "AED"


class BBox(Frozen):
    """Axis-aligned box on one page in page-normalised coordinates (0..1, origin top-left)."""

    page: Annotated[int, Field(ge=0)]
    x0: Annotated[float, Field(ge=0.0, le=1.0)]
    y0: Annotated[float, Field(ge=0.0, le=1.0)]
    x1: Annotated[float, Field(ge=0.0, le=1.0)]
    y1: Annotated[float, Field(ge=0.0, le=1.0)]


class FieldValue(Frozen):
    """One extracted field after grounding. `word_ids` and `boxes` come only from parse words,
    never from model output (I1); an ungrounded value has empty `word_ids` and `boxes`."""

    name: str
    raw: str | None
    normalized: str | None
    word_ids: tuple[str, ...] = ()
    boxes: tuple[BBox, ...] = ()


def make_idempotency_key(invoice_id: InvoiceId, step: str) -> IdempotencyKey:
    """Build the deterministic `invoice_id:step` key for a side-effecting activity (§6.4).

    Raises apx.common.errors.IdempotencyKeyError if `step` is not a lowercase identifier.
    """
    raise NotImplementedError
