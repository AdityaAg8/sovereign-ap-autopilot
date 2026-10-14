"""SQLAlchemy 2.0 typed ORM models for the Postgres schema: users/roles, vendor and
admin-verified bank master data, PO/GRN 3-way-match data, invoices and their grounded fields,
review decisions/approvals/proposals, the hash-chained audit log, LLM call telemetry and
training labels. Money is always `Numeric`, never `float`."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType

from apx.common.types import Language, Role, Route, Stage
from apx.db.base import Base, CreatedAtMixin
from apx.db.enums import (
    BankDetailSource,
    DbInvoiceStatus,
    DecisionAction,
    FieldSource,
    LabelSplit,
    ProposalKind,
    ProposalStatus,
)

MONEY = Numeric(18, 2)
"""Exact monetary amount column type: 18 total digits, 2 decimal places, never `float`."""

QUANTITY = Numeric(18, 4)
"""Line-item quantity column type; more decimal places than money for unit fractions."""

CURRENCY = String(3)
SHA256_PATTERN = r"^[0-9a-f]{64}$"


def pg_enum(enum_cls: type[StrEnum], name: str) -> SAEnum:
    """Builds a native Postgres enum type from a `StrEnum`, persisting each member's string
    *value* (e.g. `"received"`) rather than SQLAlchemy's default of its Python attribute
    *name* (e.g. `"RECEIVED"`) -- required so DB rows compare equal to the enums used
    throughout `apx` (`StrEnum` members already equal their value)."""
    return SAEnum(
        enum_cls, name=name, native_enum=True, values_callable=lambda x: [e.value for e in x]
    )


# ---------------------------------------------------------------------------------------
# Users and roles
# ---------------------------------------------------------------------------------------


class User(Base, CreatedAtMixin):
    """Authenticated human user (mirrors `apx.common.master.User`)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, server_default=text("true"))

    role_links: Mapped[list[UserRole]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RoleDef(Base):
    """Lookup row for one `apx.common.types.Role` value."""

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[Role] = mapped_column(pg_enum(Role, "role_name"), nullable=False, unique=True)

    user_links: Mapped[list[UserRole]] = relationship(back_populates="role")


class UserRole(Base):
    """Many-to-many join: a user's granted roles."""

    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="RESTRICT"), primary_key=True
    )

    user: Mapped[User] = relationship(back_populates="role_links")
    role: Mapped[RoleDef] = relationship(back_populates="user_links")


# ---------------------------------------------------------------------------------------
# Vendors and admin-verified bank details (P1, G-3)
# ---------------------------------------------------------------------------------------


class Vendor(Base, CreatedAtMixin):
    """Vendor master record (mirrors `apx.common.master.Vendor`)."""

    __tablename__ = "vendors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    trn: Mapped[str | None] = mapped_column(String(32))

    bank_details: Mapped[list[BankDetail]] = relationship(back_populates="vendor")


class BankVerification(Base):
    """An admin's out-of-band verification of a vendor IBAN (P1). This is the only record
    type a `bank_details.verification_id` FK may point to (G-3): a bank account can only be
    marked `verified` by referencing a real row here, never by a bare boolean flip."""

    __tablename__ = "bank_verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[str] = mapped_column(
        ForeignKey("vendors.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    iban: Mapped[str] = mapped_column(String(34), nullable=False)
    verified_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    verified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    method: Mapped[str] = mapped_column(String(32), nullable=False, server_default="admin_callback")
    notes: Mapped[str | None] = mapped_column(Text)

    bank_details: Mapped[list[BankDetail]] = relationship(back_populates="verification")


class BankDetail(Base):
    """A vendor bank account (mirrors `apx.common.master.BankAccount`). `verified` is a
    Postgres `GENERATED` column derived from `verification_id`, so it is physically
    impossible to set it true without a matching `bank_verifications` row (G-3); the CHECK
    constraint further ties `source` to that FK so `source='admin_callback'` iff verified."""

    __tablename__ = "bank_details"
    __table_args__ = (
        CheckConstraint(
            "(source = 'admin_callback') = (verification_id IS NOT NULL)",
            name="source_matches_verification",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[str] = mapped_column(
        ForeignKey("vendors.id", ondelete="CASCADE"), nullable=False, index=True
    )
    iban: Mapped[str] = mapped_column(String(34), nullable=False)
    source: Mapped[BankDetailSource] = mapped_column(
        pg_enum(BankDetailSource, "bank_detail_source"),
        nullable=False,
        server_default=BankDetailSource.UNVERIFIED.value,
    )
    verification_id: Mapped[int | None] = mapped_column(
        ForeignKey("bank_verifications.id", ondelete="RESTRICT")
    )
    verified: Mapped[bool] = mapped_column(
        Computed("verification_id IS NOT NULL", persisted=True), nullable=False
    )
    is_current: Mapped[bool] = mapped_column(nullable=False, server_default=text("true"))
    changed_by: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    vendor: Mapped[Vendor] = relationship(back_populates="bank_details")
    verification: Mapped[BankVerification | None] = relationship(back_populates="bank_details")


class BankDetailHistory(Base):
    """Append-only history of every `bank_details` row change (insert/update/delete),
    populated by the `bank_details_history_sync` trigger (db/migrations). Satisfies the
    "history table" requirement for bank detail changes (P2)."""

    __tablename__ = "bank_details_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bank_detail_id: Mapped[int] = mapped_column(nullable=False, index=True)
    vendor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    iban: Mapped[str] = mapped_column(String(34), nullable=False)
    source: Mapped[BankDetailSource] = mapped_column(
        pg_enum(BankDetailSource, "bank_detail_source"),
        nullable=False,
    )
    verification_id: Mapped[int | None] = mapped_column(Integer)
    verified: Mapped[bool] = mapped_column(nullable=False)
    is_current: Mapped[bool] = mapped_column(nullable=False)
    changed_by: Mapped[str] = mapped_column(String(64), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    operation: Mapped[str] = mapped_column(String(1), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    __table_args__ = (CheckConstraint("operation IN ('I', 'U', 'D')", name="operation_code"),)


# ---------------------------------------------------------------------------------------
# Purchase orders and goods-receipt notes (3-way match)
# ---------------------------------------------------------------------------------------


class PurchaseOrder(Base, CreatedAtMixin):
    __tablename__ = "pos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    po_number: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    vendor_id: Mapped[str] = mapped_column(
        ForeignKey("vendors.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    currency: Mapped[str] = mapped_column(CURRENCY, nullable=False, server_default="AED")
    total: Mapped[Decimal | None] = mapped_column(MONEY)

    lines: Mapped[list[PoLine]] = relationship(back_populates="po", cascade="all, delete-orphan")


class PoLine(Base):
    __tablename__ = "po_lines"
    __table_args__ = (UniqueConstraint("po_id", "line_no", name="uq_po_lines_po_id_line_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    po_id: Mapped[int] = mapped_column(
        ForeignKey("pos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    line_no: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(QUANTITY, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    net_amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(CURRENCY, nullable=False, server_default="AED")

    po: Mapped[PurchaseOrder] = relationship(back_populates="lines")


class Grn(Base, CreatedAtMixin):
    __tablename__ = "grns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    po_id: Mapped[int] = mapped_column(
        ForeignKey("pos.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    grn_number: Mapped[str | None] = mapped_column(String(64), unique=True)
    received_at: Mapped[date] = mapped_column(Date, nullable=False)

    lines: Mapped[list[GrnLine]] = relationship(back_populates="grn", cascade="all, delete-orphan")


class GrnLine(Base):
    __tablename__ = "grn_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    grn_id: Mapped[int] = mapped_column(
        ForeignKey("grns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    po_line_id: Mapped[int | None] = mapped_column(ForeignKey("po_lines.id", ondelete="SET NULL"))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(QUANTITY, nullable=False)

    grn: Mapped[Grn] = relationship(back_populates="lines")


# ---------------------------------------------------------------------------------------
# Ingest dedupe
# ---------------------------------------------------------------------------------------


class IngestedDocument(Base):
    """Persisted sha256 dedupe record for `apx.ingest` (mirrors
    `apx.ingest.interfaces.IngestedDocument`). The primary key on `sha256` is the atomic gate
    that `apx.ingest.ingestor.Ingestor` uses to guarantee exactly one `InvoiceWorkflow` starts
    per document, even when the watcher and the API upload path race on the same bytes:
    whichever concurrent `INSERT ... ON CONFLICT DO NOTHING` wins the row is the only caller
    that proceeds to start the workflow. `workflow_id` is set once that call succeeds; a row
    with `workflow_id IS NULL` was rolled back by a failed start (see `Ingestor.ingest`) and its
    sha256 becomes ingestible again. `failed_at`/`failure` record a workflow that gave up before
    its `invoices` row existed (`mark_failed`, migration 0011), so that failure has a durable,
    queryable home; both are set together or not at all."""

    __tablename__ = "ingested_documents"
    __table_args__ = (
        CheckConstraint(f"sha256 ~ '{SHA256_PATTERN}'", name="sha256_is_sha256"),
        CheckConstraint("(failed_at IS NULL) = (failure IS NULL)", name="failure_recorded_whole"),
        CheckConstraint(
            "media_type IN ('application/pdf', 'image/png', 'image/jpeg', 'image/tiff')",
            name="media_type_known",
        ),
        CheckConstraint("source IN ('watch', 'upload')", name="source_known"),
        CheckConstraint("size_bytes > 0", name="size_bytes_positive"),
    )

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    media_type: Mapped[str] = mapped_column(String(20), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(10), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    workflow_id: Mapped[str | None] = mapped_column(String(150))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure: Mapped[str | None] = mapped_column(Text)
    """The typed failure chain `InvoiceWorkflow` gave up with, cut to `mark_failed`'s limit."""


# ---------------------------------------------------------------------------------------
# Invoices and grounded fields
# ---------------------------------------------------------------------------------------


class Invoice(Base):
    """Mirrors `apx.common.invoice.Invoice`; `id` equals `document_sha256` for ingested
    invoices (see `apx.common.types.InvoiceId`)."""

    __tablename__ = "invoices"
    __table_args__ = (
        CheckConstraint(f"id ~ '{SHA256_PATTERN}'", name="id_is_sha256"),
        CheckConstraint(f"document_sha256 ~ '{SHA256_PATTERN}'", name="document_sha256_is_sha256"),
        CheckConstraint("kind IN ('invoice', 'credit_note')", name="kind_known"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    vendor_id: Mapped[str | None] = mapped_column(ForeignKey("vendors.id", ondelete="SET NULL"))
    language: Mapped[Language] = mapped_column(
        pg_enum(Language, "invoice_language"), nullable=False
    )
    number: Mapped[str] = mapped_column(String(100), nullable=False)
    issue_date: Mapped[date] = mapped_column(Date, nullable=False)
    due_date: Mapped[date | None] = mapped_column(Date)
    po_number: Mapped[str | None] = mapped_column(String(64))
    currency: Mapped[str] = mapped_column(CURRENCY, nullable=False, server_default="AED")
    subtotal: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    vat_total: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    total: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    iban: Mapped[str | None] = mapped_column(String(34))
    kind: Mapped[str | None] = mapped_column(String(16))
    """`apx.common.invoice.InvoiceKind` value (ADR-0012); NULL when unresolved or a legacy row."""
    status: Mapped[DbInvoiceStatus] = mapped_column(
        pg_enum(DbInvoiceStatus, "invoice_status"),
        nullable=False,
        server_default=DbInvoiceStatus.RECEIVED.value,
    )
    route: Mapped[Route | None] = mapped_column(pg_enum(Route, "route"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
        onupdate=text("now()"),
    )


class Field(Base, CreatedAtMixin):
    """One extracted/grounded field value on an invoice (I1)."""

    __tablename__ = "fields"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        Index("ix_fields_invoice_id_name", "invoice_id", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column()
    boxes: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    source: Mapped[FieldSource] = mapped_column(
        pg_enum(FieldSource, "field_source"), nullable=False
    )
    model_revision: Mapped[str | None] = mapped_column(String(100))
    """Pinned model revision (`apx.llm.interfaces.ModelSpec.revision`) that produced this
    value, when known; `None` for rows written before this was tracked or for `source ==
    FieldSource.HUMAN` rows (a reviewer, not a model). Lets a later correction record which
    model/revision it is correcting."""


# ---------------------------------------------------------------------------------------
# Fraud flags
# ---------------------------------------------------------------------------------------


class FraudFlag(Base):
    """One `apx.fraud.interfaces.FraudFlag` raised on an invoice, kept so review can show its
    plain-language explanation. Created by `0006_fraud_flags`."""

    __tablename__ = "fraud_flags"
    __table_args__ = (
        CheckConstraint("severity IN ('low', 'medium', 'high')", name="severity_known"),
        CheckConstraint("score >= 0 AND score <= 1", name="score_in_unit_interval"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rule_id: Mapped[str] = mapped_column(String(64), nullable=False)
    typology: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


PAGE_VECTOR_DIMENSIONS = 2048
"""Qwen3-VL-Embedding-2B's full embedding size (its `1_Pooling/config.json`); the
`page_embeddings.embedding` column is fixed to it, so a vector of any other size is refused."""


class PgVector(UserDefinedType[list[float]]):
    """pgvector's `vector(n)` column type, for DDL and drift checks only: rows are written and
    queried with explicit SQL casts (`apx.fraud.pg_page_similarity`), not through the ORM."""

    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({self.dimensions})"


class PageEmbedding(Base):
    """One page's Qwen3-VL-Embedding-2B vector, scoped to the invoice's resolved vendor, for
    learned near-duplicate detection. Created by `0010_page_embeddings`, which also enables
    the `vector` extension."""

    __tablename__ = "page_embeddings"
    __table_args__ = (CheckConstraint("page >= 0", name="page_non_negative"),)

    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), primary_key=True
    )
    page: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[str] = mapped_column(
        ForeignKey("vendors.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(PgVector(PAGE_VECTOR_DIMENSIONS), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------------------
# Decisions, approvals and agent/chat proposals (P3-P5, I3)
# ---------------------------------------------------------------------------------------


class Decision(Base):
    """A routing or human review decision on an invoice (`PolicyEngine.route` output or an
    `apx.api.interfaces.DecisionRequest`)."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[DecisionAction] = mapped_column(
        pg_enum(DecisionAction, "decision_action"), nullable=False
    )
    route: Mapped[Route | None] = mapped_column(pg_enum(Route, "route"))
    reasons: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    proposal_id: Mapped[str | None] = mapped_column(ForeignKey("proposals.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Approval(Base):
    """A human approval of a payment-affecting action (I3: policy allow + human approval by
    an allowed role + segregation of duties, enforced by `apx.policy`)."""

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id", ondelete="SET NULL"))
    approver_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    role_at_approval: Mapped[Role] = mapped_column(pg_enum(Role, "role_name"), nullable=False)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(CURRENCY, nullable=False, server_default="AED")
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    notes: Mapped[str | None] = mapped_column(Text)
    invoice_digest: Mapped[str | None] = mapped_column(String(64))
    """`apx.policy.writer.approval_digest` of the invoice and vendor payee at approval time. The
    workflow re-derives it before paying and holds when it does not match; NULL only on rows
    written before migration 0012, which the workflow treats as unbacked."""
    corrections_digest: Mapped[str | None] = mapped_column(String(64))
    """`apx.review.corrections.corrections_digest` of the review corrections the approver
    approved. The workflow acts on a `review` signal only when its corrections digest to this and
    the invoice's current `fields` rows still do; NULL on rows written before migration 0014 or
    by a caller that passed none, and the check then compares the signal with the current
    rows."""


class Proposal(Base):
    """Mirrors `apx.policy.interfaces.Proposal`: an agent/chat-drafted action that never
    executes by itself (P5)."""

    __tablename__ = "proposals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    invoice_id: Mapped[str | None] = mapped_column(ForeignKey("invoices.id", ondelete="CASCADE"))
    kind: Mapped[ProposalKind] = mapped_column(
        pg_enum(ProposalKind, "proposal_kind"), nullable=False
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_by: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[ProposalStatus] = mapped_column(
        pg_enum(ProposalStatus, "proposal_status"),
        nullable=False,
        server_default=ProposalStatus.PENDING.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------------------
# Audit chain (I4) and idempotency
# ---------------------------------------------------------------------------------------


class AuditLog(Base):
    """Hash-chained, append-only audit log (mirrors `apx.audit.interfaces.AuditEntry`).
    `db/migrations` attaches a trigger that rejects UPDATE/DELETE at the database level."""

    __tablename__ = "audit_log"
    __table_args__ = (CheckConstraint(f"hash ~ '{SHA256_PATTERN}'", name="hash_is_sha256"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"), index=True
    )
    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity: Mapped[str] = mapped_column(String(200), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)


class AuditAnchor(Base):
    """Append-only checkpoint of the `audit_log` chain's head at a point in time (mirrors
    `apx.audit.anchor.AuditAnchorRecord`). Its own independent hash chain (`prev_anchor_hash`
    / `anchor_hash`), combined with a copy kept outside this database, lets
    `apx.audit.anchor.verify_anchors` detect tail truncation and whole-chain rewrites that
    `audit_log`'s internal hash chain alone cannot see (I4). `db/migrations` attaches the
    same append-only trigger as `audit_log`."""

    __tablename__ = "audit_anchor"
    __table_args__ = (
        CheckConstraint(f"last_hash ~ '{SHA256_PATTERN}'", name="last_hash_is_sha256"),
        CheckConstraint(
            f"prev_anchor_hash ~ '{SHA256_PATTERN}'", name="prev_anchor_hash_is_sha256"
        ),
        CheckConstraint(f"anchor_hash ~ '{SHA256_PATTERN}'", name="anchor_hash_is_sha256"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"), index=True
    )
    last_id: Mapped[int] = mapped_column(Integer, nullable=False)
    last_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prev_anchor_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    anchor_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)


class LlmCall(Base):
    """Telemetry for one LLM call, keyed by `trace_id` (OTel span) for the operations dashboard."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str | None] = mapped_column(
        ForeignKey("invoices.id", ondelete="SET NULL"), index=True
    )
    stage: Mapped[Stage] = mapped_column(pg_enum(Stage, "llm_stage"))
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str | None] = mapped_column(String(100))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class IdempotencyRecord(Base):
    """Deterministic `invoice_id:step` key store for side-effecting activities; ASP
    send and archive look themselves up here before acting, so a retried activity is a no-op."""

    __tablename__ = "idempotency_keys"
    __table_args__ = (
        UniqueConstraint("invoice_id", "step", name="uq_idempotency_keys_invoice_id_step"),
        CheckConstraint(
            r"key ~ '^[^:\s]+:[a-z][a-z0-9_]*$'",
            name="key_format",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False
    )
    step: Mapped[str] = mapped_column(String(50), nullable=False)
    key: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


# ---------------------------------------------------------------------------------------
# Training labels (I5: eval/cal/test data never reach training)
# ---------------------------------------------------------------------------------------


class Label(Base, CreatedAtMixin):
    """A reviewer's field correction, one row per fix. `created_at` (`CreatedAtMixin`) is
    the correction timestamp and `labeled_by` the reviewer; the `previous_*` columns
    snapshot the field value being corrected (before/after value, grounding box and the
    model/revision that produced it) so the row alone is a complete, auditable record of
    the fix, independent of whether the original `fields` row is ever superseded again
    later."""

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[str] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    field_name: Mapped[str | None] = mapped_column(String(100))
    corrected_value: Mapped[str | None] = mapped_column(Text)
    corrected_normalized: Mapped[str | None] = mapped_column(Text)
    previous_value: Mapped[str | None] = mapped_column(Text)
    """The field's value immediately before this correction (`None` when there was no prior
    value, e.g. a field the automated pipeline never produced)."""
    previous_boxes: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    """Grounding box(es) of the value being corrected, copied from `fields.boxes` (I1: boxes
    always come from parse words, never invented here)."""
    previous_source: Mapped[FieldSource | None] = mapped_column(
        pg_enum(FieldSource, "field_source")
    )
    """Which extractor/arbiter/human produced the value being corrected."""
    previous_model_revision: Mapped[str | None] = mapped_column(String(100))
    """The model/revision that produced the value being corrected, copied from
    `fields.model_revision`."""
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    split: Mapped[LabelSplit | None] = mapped_column(pg_enum(LabelSplit, "label_split"))
    labeled_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    notes: Mapped[str | None] = mapped_column(Text)
