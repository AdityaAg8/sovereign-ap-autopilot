"""`PgAgentToolbox`: the typed implementation of the six agent tools over a `SqlReader` and a
`ProposalSink` only (ADR-0010 D1-D4). Every method scopes the `agent` origin to its scope
invoice (and that invoice's vendor), applies the contract's row cap, and puts document-derived
text only in `UntrustedText`; the two propose tools never do more than call `ProposalSink.submit`
(I2, P5)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final
from uuid import uuid4

from apx.agent.interfaces import (
    AUDIT_FACT_KEYS,
    EXPLAIN_MAX_EVENTS,
    VENDOR_MAX_ACCOUNTS,
    VENDOR_MAX_BANK_CHANGES,
    AccountView,
    AuditEventView,
    BankChangeView,
    CitedReason,
    DecisionExplanation,
    DraftVendorEmailParams,
    EmailDraftReceipt,
    EmailReason,
    ExplainDecisionParams,
    FieldStatus,
    GetInvoiceParams,
    InvoiceRow,
    InvoiceSearchResult,
    InvoiceView,
    ProposalReceipt,
    ProposalView,
    ReasonView,
    RequestApprovalParams,
    SearchInvoicesParams,
    SqlParam,
    SqlReader,
    SqlStatement,
    ToolArgumentError,
    ToolCaller,
    ToolNotFoundError,
    ToolPreconditionError,
    ToolScopeError,
    VendorHistory,
    VendorHistoryParams,
)
from apx.agent.sql import (
    SQL_EXPLAIN_DECISION_EVENTS,
    SQL_GET_INVOICE,
    SQL_SEARCH_INVOICES_RECEIVED_ASC,
    SQL_SEARCH_INVOICES_RECEIVED_DESC,
    SQL_SEARCH_INVOICES_TOTAL_DESC,
    SQL_VENDOR_ACCOUNTS,
    SQL_VENDOR_BANK_CHANGES,
    SQL_VENDOR_HEADER,
    SQL_VENDOR_INVOICES,
    escape_like,
)
from apx.common.types import InvoiceId, Money, Route, VendorId
from apx.common.untrusted import UntrustedText, to_untrusted
from apx.policy.interfaces import Proposal, ProposalSink

_TRN_RE: Final[re.Pattern[str]] = re.compile(r"^\d{15}$")
_INVOICE_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/._\-]{0,39}$")

_SEARCH_SORT_SQL: Final[dict[str, SqlStatement]] = {
    "received_desc": SQL_SEARCH_INVOICES_RECEIVED_DESC,
    "received_asc": SQL_SEARCH_INVOICES_RECEIVED_ASC,
    "total_desc": SQL_SEARCH_INVOICES_TOTAL_DESC,
}

_TOPIC_ACTION_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "route": ("activity:route",),
    "trust": ("activity:trust_gate",),
    "validation": ("activity:validate",),
    "fraud": ("activity:fraud",),
    "scan": ("activity:scan",),
    "approval": (
        "proposal.",
        "decision.",
        "approval.",
        "authorize:",
        "payment_state:",
        "activity:apply_review",
    ),
    "all": (),
}

_EMAIL_TEMPLATES: Final[dict[EmailReason, dict[str, tuple[str, str]]]] = {
    "missing_po": {
        "en": ("Missing purchase order number", "Your invoice did not include a PO number."),
        "ar": ("رقم أمر الشراء مفقود", "لم تتضمن فاتورتكم رقم أمر شراء."),
    },
    "po_mismatch": {
        "en": (
            "Purchase order does not match",
            "The PO number on your invoice does not match our records.",
        ),
        "ar": ("رقم أمر الشراء غير مطابق", "رقم أمر الشراء في فاتورتكم لا يطابق سجلاتنا."),
    },
    "amount_mismatch": {
        "en": (
            "Invoice amount does not match",
            "The amount on your invoice does not match the purchase order or receipt.",
        ),
        "ar": ("قيمة الفاتورة غير مطابقة", "قيمة فاتورتكم لا تطابق أمر الشراء أو إيصال الاستلام."),
    },
    "invalid_trn": {
        "en": (
            "Tax registration number issue",
            "The TRN on your invoice could not be validated.",
        ),
        "ar": ("مشكلة في الرقم الضريبي", "تعذر التحقق من الرقم الضريبي في فاتورتكم."),
    },
    "missing_field": {
        "en": ("Missing information on invoice", "Your invoice is missing required information."),
        "ar": ("معلومات ناقصة في الفاتورة", "تفتقر فاتورتكم إلى معلومات مطلوبة."),
    },
    "duplicate_suspected": {
        "en": (
            "Possible duplicate invoice",
            "This invoice looks similar to one we already have on file.",
        ),
        "ar": ("احتمال تكرار الفاتورة", "تبدو هذه الفاتورة مشابهة لفاتورة أخرى لدينا."),
    },
    "other": {
        "en": ("Question about your invoice", "We have a question about your invoice."),
        "ar": ("استفسار بخصوص فاتورتكم", "لدينا استفسار بخصوص فاتورتكم."),
    },
}
"""Fixed, per-reason, per-language subject/opening pairs (ADR-0010 D4); `details` is appended
as a separate paragraph, never interpolated into the template itself."""

_EMAIL_FORBIDDEN_RE: Final[re.Pattern[str]] = re.compile(
    r"https?://|www\.|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|\d{12,}|"
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
    re.IGNORECASE,
)
"""URLs, email addresses, 12+ digit runs and IBAN-like tokens (ADR-0010 D4: `details` may not
carry a channel or bank detail the vendor did not already have)."""


def _money(amount: Decimal | None, currency: str) -> Money | None:
    return None if amount is None else Money(amount=amount, currency=currency)


def _rows(seq: Sequence[Mapping[str, SqlParam]]) -> list[dict[str, Any]]:
    """Widens a reader result to `dict[str, Any]`: rows are an untyped I/O boundary, and every
    value is re-validated by a pydantic result model before it reaches the model or the caller."""
    return [dict(r) for r in seq]


def _json(value: object) -> Any:
    """`text()` queries carry no column type info, so asyncpg returns `json`/`jsonb` columns as
    raw text; the ORM path would decode them automatically, this one decodes explicitly. Already
    a `list`/`dict` (or `None`) passes through unchanged, so this is safe under either driver
    behaviour."""
    return json.loads(value) if isinstance(value, str) else value


def _typed_or_untrusted(
    raw: str | None, pattern: re.Pattern[str], source: str
) -> tuple[str | None, UntrustedText | None]:
    """T2 split: a value matching `pattern` is returned typed; anything else (including None)
    that has raw text becomes `UntrustedText` instead (ADR-0008 T2, ADR-0010)."""
    if raw is None:
        return None, None
    if pattern.fullmatch(raw):
        return raw, None
    return None, to_untrusted(raw, source)


def _invoice_row(row: dict[str, Any]) -> InvoiceRow:
    invoice_id = InvoiceId(str(row["invoice_id"]))
    number, _ = _typed_or_untrusted(
        row.get("invoice_number_raw") if row.get("invoice_number_raw") else None,
        _INVOICE_NUMBER_RE,
        f"invoice:{invoice_id}:field:invoice_number",
    )
    return InvoiceRow(
        invoice_id=invoice_id,
        status=str(row["status"]),
        route=row["route"],
        vendor_id=VendorId(str(row["vendor_id"])) if row.get("vendor_id") else None,
        vendor_name=row.get("vendor_name"),
        invoice_number=number,
        issue_date=row.get("issue_date"),
        total=_money(row.get("total"), str(row.get("currency") or "AED")),
        received_at=row["received_at"],
    )


def _field_rows(
    invoice_id: InvoiceId, fields_json: list[dict[str, Any]] | None
) -> tuple[tuple[FieldStatus, ...], dict[str, str], list[UntrustedText]]:
    """Splits the view's `fields_json` into the `FieldStatus` list, a `{name: normalized value}`
    map for typed header fields, and the `UntrustedText` blocks for free-text fields (supplier
    and buyer name/address, ADR-0010: "never from the document" applies to the vendor master,
    not these -- these two parties are exactly what the document itself claims)."""
    statuses: list[FieldStatus] = []
    typed: dict[str, str] = {}
    untrusted: list[UntrustedText] = []
    for entry in fields_json or []:
        name = str(entry["name"])
        value = entry.get("normalized_value") or entry.get("value")
        confidence = entry.get("confidence")
        statuses.append(
            FieldStatus(
                name=name,
                accepted=entry.get("normalized_value") is not None,
                p_correct=float(confidence) if confidence is not None else None,
            )
        )
        if value is None:
            continue
        if name in ("supplier_name", "supplier_address", "buyer_name", "buyer_address"):
            untrusted.append(to_untrusted(str(value), f"invoice:{invoice_id}:field:{name}"))
        else:
            typed[name] = str(value)
    return tuple(statuses), typed, untrusted


class PgAgentToolbox:
    """The `AgentToolbox` implementation. Never imports a `Grant`, `PolicyGuard` or payment-state
    writer; its only dependencies are a `SqlReader` and a `ProposalSink` (ADR-0010 D2, K3)."""

    def __init__(self, reader: SqlReader, sink: ProposalSink) -> None:
        self._reader = reader
        self._sink = sink

    # -- scope helpers ------------------------------------------------------------------

    async def _scope_vendor_id(self, caller: ToolCaller) -> VendorId | None:
        """The vendor of the `agent` origin's scope invoice, or None if it has no vendor."""
        assert caller.scope_invoice_id is not None
        row = await self._fetch_invoice(caller.scope_invoice_id)
        if row is None:
            return None
        vendor_id = row.get("vendor_id")
        return VendorId(str(vendor_id)) if vendor_id else None

    async def _fetch_invoice(self, invoice_id: InvoiceId) -> dict[str, Any] | None:
        rows = _rows(await self._reader.fetch_all(SQL_GET_INVOICE, {"invoice_id": invoice_id}, 1))
        if not rows:
            return None
        row = rows[0]
        for json_column in ("fields_json", "proposals_json", "reasons_json"):
            row[json_column] = _json(row.get(json_column))
        return row

    # -- search_invoices ------------------------------------------------------------------

    async def search_invoices(
        self, params: SearchInvoicesParams, caller: ToolCaller
    ) -> InvoiceSearchResult:
        vendor_id = params.vendor_id
        if caller.origin == "agent":
            vendor_id = await self._scope_vendor_id(caller)
            if vendor_id is None:
                # The scope invoice has no vendor (the real P1 "vendor unknown" REVIEW path).
                # `vendor_id=None` bound into `_SEARCH_INVOICES_WHERE`'s
                # `(CAST(:vendor_id AS text) IS NULL OR vendor_id = ...)` clause disables the
                # vendor filter rather than matching nothing, which would hand an `agent` caller
                # an unscoped, cross-vendor search -- refuse instead of ever binding that NULL.
                raise ToolScopeError("search_invoices: scope invoice has no vendor")

        bound = {
            "vendor_id": vendor_id,
            "vendor_query": escape_like(params.vendor_query) if params.vendor_query else None,
            "invoice_number": params.invoice_number,
            "status": params.status,
            "route": params.route,
            "received_from": params.received_from,
            "received_to": params.received_to,
            "total_min": params.total_min,
            "total_max": params.total_max,
            "currency": params.currency,
        }
        rows = _rows(
            await self._reader.fetch_all(_SEARCH_SORT_SQL[params.sort], bound, params.limit + 1)
        )
        truncated = len(rows) > params.limit
        return InvoiceSearchResult(
            rows=tuple(_invoice_row(r) for r in rows[: params.limit]),
            truncated=truncated,
        )

    # -- get_invoice ------------------------------------------------------------------

    async def get_invoice(self, params: GetInvoiceParams, caller: ToolCaller) -> InvoiceView:
        if caller.origin == "agent" and params.invoice_id != caller.scope_invoice_id:
            raise ToolScopeError("get_invoice: outside the agent's scope invoice")

        row = await self._fetch_invoice(params.invoice_id)
        if row is None:
            raise ToolNotFoundError(f"no invoice {params.invoice_id!r}")

        invoice_id = params.invoice_id
        fields, typed, untrusted = _field_rows(invoice_id, row.get("fields_json"))
        supplier_trn, trn_untrusted = _typed_or_untrusted(
            typed.get("supplier_trn"), _TRN_RE, f"invoice:{invoice_id}:field:supplier_trn"
        )
        buyer_trn, buyer_trn_untrusted = _typed_or_untrusted(
            typed.get("buyer_trn"), _TRN_RE, f"invoice:{invoice_id}:field:buyer_trn"
        )
        for block in (trn_untrusted, buyer_trn_untrusted):
            if block is not None:
                untrusted.append(block)

        reasons = tuple(
            # `PolicyRule` values such as "P1" are lower-cased into the `Code` shape.
            ReasonView(rule=str(r["rule"]).lower(), route=r["route"], message=r["message"])
            for r in (row.get("reasons_json") or [])
        )
        proposals = tuple(
            ProposalView(
                proposal_id=str(p["id"]),
                kind=str(p["kind"]),
                status=p["status"],
                created_by=str(p["created_by"]),
                created_at=p["created_at"],
            )
            for p in (row.get("proposals_json") or [])
            if p.get("status") == "pending"
        )

        return InvoiceView(
            row=_invoice_row(row),
            due_date=row.get("due_date"),
            po_number=_typed_or_untrusted(
                row.get("po_number_raw"),
                _INVOICE_NUMBER_RE,
                f"invoice:{invoice_id}:field:po_number",
            )[0],
            supplier_trn=supplier_trn,
            buyer_trn=buyer_trn,
            subtotal=_money(row.get("subtotal"), str(row.get("currency") or "AED")),
            vat_total=_money(row.get("vat_total"), str(row.get("currency") or "AED")),
            payee=row.get("payee_masked"),
            document_iban_matches_payee=row.get("document_iban_matches_payee"),
            payee_match=row.get("payee_match"),
            payee_refusal=row.get("payee_refusal"),
            # Always 0: there is no invoice line-item table yet, so there is nothing to count
            # (extraction stores only invoice-level `fields`, never per-line rows). Tracked as an
            # M9.3 follow-up in ADR-0010: needs a line-items table plus a `v_agent_invoices`
            # column or join before this can be exact.
            line_count=0,
            fields=fields,
            reasons=reasons,
            flags=(),
            pending_proposals=proposals,
            untrusted=tuple(untrusted),
        )

    # -- explain_decision ------------------------------------------------------------------

    async def explain_decision(
        self, params: ExplainDecisionParams, caller: ToolCaller
    ) -> DecisionExplanation:
        if caller.origin == "agent" and params.invoice_id != caller.scope_invoice_id:
            raise ToolScopeError("explain_decision: outside the agent's scope invoice")

        header = await self._fetch_invoice(params.invoice_id)
        if header is None:
            raise ToolNotFoundError(f"no invoice {params.invoice_id!r}")

        event_rows = _rows(
            await self._reader.fetch_all(
                SQL_EXPLAIN_DECISION_EVENTS,
                {"invoice_id": params.invoice_id},
                EXPLAIN_MAX_EVENTS + 1,
            )
        )
        truncated = len(event_rows) > EXPLAIN_MAX_EVENTS
        event_rows = event_rows[:EXPLAIN_MAX_EVENTS]

        prefixes = _TOPIC_ACTION_PREFIXES[params.topic]
        if prefixes:
            event_rows = [r for r in event_rows if str(r["action"]).startswith(prefixes)]

        events = tuple(
            AuditEventView(
                audit_id=r["audit_id"],
                ts=r["ts"],
                actor=r["actor"],
                action=r["action"],
                facts=_scalar_facts(_json(r["payload_json"]) or {}),
            )
            for r in event_rows
        )

        raw_reasons = header.get("reasons_json")

        cited: list[CitedReason] = []
        for reason in raw_reasons or []:
            ids = tuple(
                sorted(
                    {
                        e.audit_id
                        for e in events
                        if e.facts.get("rule") == reason["rule"]
                        and e.facts.get("route") == reason["route"]
                    }
                )
            )
            if ids:
                cited.append(
                    CitedReason(
                        rule=str(reason["rule"]).lower(),
                        route=reason["route"],
                        message=reason["message"],
                        audit_ids=ids,
                    )
                )

        return DecisionExplanation(
            invoice_id=params.invoice_id,
            status=str(header["status"]),
            route=header.get("route"),
            reasons=tuple(cited),
            events=events,
            truncated=truncated,
        )

    # -- vendor_history ------------------------------------------------------------------

    async def vendor_history(
        self, params: VendorHistoryParams, caller: ToolCaller
    ) -> VendorHistory:
        vendor_id = params.vendor_id
        if caller.origin == "agent":
            scope_vendor = await self._scope_vendor_id(caller)
            if scope_vendor is None or vendor_id != scope_vendor:
                raise ToolScopeError("vendor_history: outside the agent's scope vendor")

        header_rows = _rows(
            await self._reader.fetch_all(SQL_VENDOR_HEADER, {"vendor_id": vendor_id}, 1)
        )
        if not header_rows:
            raise ToolNotFoundError(f"no vendor {vendor_id!r}")
        header = header_rows[0]

        account_rows = _rows(
            await self._reader.fetch_all(
                SQL_VENDOR_ACCOUNTS, {"vendor_id": vendor_id}, VENDOR_MAX_ACCOUNTS + 1
            )
        )
        accounts = tuple(
            AccountView(
                iban=r["masked_iban"],
                verified=r["verified"],
                source=r["source"],
                verified_at=r.get("verified_at"),
            )
            for r in account_rows[:VENDOR_MAX_ACCOUNTS]
        )

        change_rows = _rows(
            await self._reader.fetch_all(
                SQL_VENDOR_BANK_CHANGES, {"vendor_id": vendor_id}, VENDOR_MAX_BANK_CHANGES + 1
            )
        )
        bank_changes = tuple(
            BankChangeView(
                changed_at=r["changed_at"],
                changed_by=r["changed_by"],
                old_iban=r.get("old_masked_iban"),
                new_iban=r["masked_iban"],
                # Always None today: `bank_details_history` (the source of
                # `v_agent_vendor_accounts`'s "change" rows) has no column linking a row back to
                # the `audit_log` id that recorded the change, so there is no join key to select
                # here. ADR-0010 D6 promises a citable bank-change audit id; closing this gap
                # needs a schema change (an `audit_id`/`change_id` column on
                # `bank_details_history`, written alongside the audit event) -- tracked as an
                # M9.3 follow-up in ADR-0010, not something `apx.agent`'s read-only SQL can fix.
                audit_id=None,
            )
            for r in change_rows[:VENDOR_MAX_BANK_CHANGES]
        )

        invoice_rows = _rows(
            await self._reader.fetch_all(
                SQL_VENDOR_INVOICES,
                {"vendor_id": vendor_id, "since": params.since},
                params.limit + 1,
            )
        )
        truncated = (
            len(account_rows) > VENDOR_MAX_ACCOUNTS
            or len(change_rows) > VENDOR_MAX_BANK_CHANGES
            or len(invoice_rows) > params.limit
        )
        invoices = tuple(_invoice_row(r) for r in invoice_rows[: params.limit])

        return VendorHistory(
            vendor_id=vendor_id,
            name=str(header["vendor_name"]),
            trn=header.get("vendor_trn"),
            payee=header.get("payee_masked"),
            accounts=accounts,
            bank_changes=bank_changes,
            invoices=invoices,
            invoice_count=len(invoices),
            flagged_count=0,
            truncated=truncated,
        )

    # -- request_approval ------------------------------------------------------------------

    async def request_approval(
        self, params: RequestApprovalParams, caller: ToolCaller
    ) -> ProposalReceipt:
        header = await self._fetch_invoice(params.invoice_id)
        if header is None:
            raise ToolNotFoundError(f"no invoice {params.invoice_id!r}")

        for existing in header.get("proposals_json") or []:
            if existing.get("kind") == "approve_payment" and existing.get("status") == "pending":
                return ProposalReceipt(
                    proposal_id=str(existing["id"]),
                    kind="approve_payment",
                    invoice_id=params.invoice_id,
                    note="A pending approval request already exists for this invoice.",
                )

        if header.get("route") != Route.REVIEW:
            raise ToolPreconditionError(f"invoice {params.invoice_id!r} is not routed to review")

        event_rows = _rows(
            await self._reader.fetch_all(
                SQL_EXPLAIN_DECISION_EVENTS,
                {"invoice_id": params.invoice_id},
                EXPLAIN_MAX_EVENTS + 1,
            )
        )
        known_ids = {r["audit_id"] for r in event_rows}
        if not set(params.cited_audit_ids) <= known_ids:
            raise ToolNotFoundError("one or more cited_audit_ids are not this invoice's own")

        proposal_id = f"appr_{uuid4().hex}"
        proposal = Proposal(
            id=proposal_id,
            invoice_id=params.invoice_id,
            kind="approve_payment",
            payload={
                "rationale": params.rationale,
                "cited_audit_ids": list(params.cited_audit_ids),
                "requested_by": _requested_by(caller),
            },
            created_by=caller.origin,
            created_at=_utcnow(),
        )
        await self._sink.submit(proposal)
        return ProposalReceipt(
            proposal_id=proposal_id,
            kind="approve_payment",
            invoice_id=params.invoice_id,
            note="Pending approval created; a reviewer or approver must execute it.",
        )

    # -- draft_vendor_email ------------------------------------------------------------------

    async def draft_vendor_email(
        self, params: DraftVendorEmailParams, caller: ToolCaller
    ) -> EmailDraftReceipt:
        if caller.origin == "agent" and params.invoice_id != caller.scope_invoice_id:
            raise ToolScopeError("draft_vendor_email: outside the agent's scope invoice")

        if params.details and _EMAIL_FORBIDDEN_RE.search(params.details):
            raise _invalid_email_details()

        header = await self._fetch_invoice(params.invoice_id)
        if header is None:
            raise ToolNotFoundError(f"no invoice {params.invoice_id!r}")
        vendor_id = header.get("vendor_id")
        if vendor_id is None:
            raise ToolPreconditionError(f"invoice {params.invoice_id!r} has no vendor on file")

        lang = "ar" if params.language.value == "ar" else "en"
        subject_line, opening = _EMAIL_TEMPLATES[params.reason][lang]
        body = opening if not params.details else f"{opening}\n\n{params.details}"

        proposal_id = f"mail_{uuid4().hex}"
        proposal = Proposal(
            id=proposal_id,
            invoice_id=params.invoice_id,
            kind="request_info",
            payload={
                "reason": params.reason,
                "language": params.language.value,
                "subject": subject_line,
                "body": body,
                "to_vendor_id": vendor_id,
            },
            created_by=caller.origin,
            created_at=_utcnow(),
        )
        await self._sink.submit(proposal)
        return EmailDraftReceipt(
            proposal_id=proposal_id,
            invoice_id=params.invoice_id,
            to_vendor_id=VendorId(str(vendor_id)),
            subject=subject_line,
            body=body,
        )


def _scalar_facts(payload: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    facts: dict[str, str | int | float | bool | None] = {}
    for key in AUDIT_FACT_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None or isinstance(value, (str, int, float, bool)):
            facts[key] = value
    return facts


def _requested_by(caller: ToolCaller) -> str:
    return caller.origin if caller.user is None else str(caller.user.id)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _invalid_email_details() -> ToolArgumentError:
    return ToolArgumentError(
        "details may not contain a URL, email address, long digit run or IBAN-like token"
    )
