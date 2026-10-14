"""DB-only string enums persisted as Postgres native enum types. Kept local to `apx.db` (which
imports only `apx.common`, matching the leaf-module import direction in `docs/architecture.md`)
rather than importing the heavier `apx.workflows`/`apx.extract`/`apx.policy` packages; the
`test_db_models.py` unit tests assert these values stay in sync with those interfaces."""

from __future__ import annotations

from enum import StrEnum


class DbInvoiceStatus(StrEnum):
    """Mirrors `apx.workflows.interfaces.InvoiceStatus`."""

    RECEIVED = "received"
    SCANNED = "scanned"
    PARSED = "parsed"
    EXTRACTED = "extracted"
    GATED = "gated"
    VALIDATED = "validated"
    FRAUD_CHECKED = "fraud_checked"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    SENT = "sent"
    ARCHIVED = "archived"
    BLOCKED = "blocked"
    FAILED = "failed"


class FieldSource(StrEnum):
    """Provenance of a `fields` row value."""

    EXTRACTOR_A = "extractor_a"
    EXTRACTOR_B = "extractor_b"
    ARBITER = "arbiter"
    HUMAN = "human"


class BankDetailSource(StrEnum):
    """Mirrors the `source` literal on `apx.common.master.BankAccount`."""

    ADMIN_CALLBACK = "admin_callback"
    UNVERIFIED = "unverified"


class ProposalKind(StrEnum):
    """Mirrors `apx.policy.interfaces.Proposal.kind`."""

    EDIT_FIELD = "edit_field"
    ROUTE = "route"
    REQUEST_INFO = "request_info"
    APPROVE_PAYMENT = "approve_payment"
    CHANGE_BANK = "change_bank"


class ProposalStatus(StrEnum):
    """Mirrors `apx.policy.interfaces.Proposal.status`."""

    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"


class DecisionAction(StrEnum):
    """Mirrors `apx.api.interfaces.DecisionRequest.action`, plus `policy_route` for the
    automated `PolicyEngine.route` outcome that also gets logged to `decisions`."""

    APPROVE = "approve"
    REJECT = "reject"
    SEND_TO_REVIEW = "send_to_review"
    EXECUTE_PROPOSAL = "execute_proposal"
    POLICY_ROUTE = "policy_route"


class LabelSplit(StrEnum):
    """Dataset split a label belongs to
    (I5: eval/cal/test data never reach training); `None` means the label is not yet assigned to a
    split."""

    TRAIN = "train"
    CAL = "cal"
    TEST = "test"
