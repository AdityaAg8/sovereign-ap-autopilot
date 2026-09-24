"""Unit tests for `PgToolDispatcher` (PRD §10 M9.2, ADR-0010 D1, D3, D7): every `ToolRefusalCode`
is reachable, non-tool errors (a `PolicyDeniedError` maps to `policy_denied`; anything else
propagates) behave as documented, and `specs()` only lists what a caller may actually use."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apx.agent.dispatch import PgToolDispatcher
from apx.agent.interfaces import (
    MCP_MAX_CALLS_PER_MINUTE,
    TOOL_RESULT_MAX_BYTES,
    AgentToolbox,
    AuditEventView,
    BankChangeView,
    CitedReason,
    DecisionExplanation,
    ExplainDecisionParams,
    GetInvoiceParams,
    InvoiceRow,
    InvoiceSearchResult,
    InvoiceView,
    ProposalReceipt,
    ProposalView,
    ReasonView,
    RequestApprovalParams,
    SearchInvoicesParams,
    ToolCaller,
    ToolNotFoundError,
    ToolPreconditionError,
    VendorHistory,
    VendorHistoryParams,
)
from apx.common.errors import ConfigError, PolicyDeniedError
from apx.common.types import InvoiceId, Route, UserId
from apx.llm.interfaces import ToolCall
from tests.unit._agent_factories import (
    admin,
    agent_caller,
    approver,
    chat_caller,
    mcp_caller,
    reviewer,
)


class _StubToolbox:
    """A minimal `AgentToolbox` whose methods either return a canned result or raise, so
    dispatch-level behaviour can be tested without a database."""

    def __init__(self) -> None:
        self.get_invoice_error: Exception | None = None
        self.request_approval_error: Exception | None = None

    async def search_invoices(
        self, params: SearchInvoicesParams, caller: ToolCaller
    ) -> InvoiceSearchResult:
        return InvoiceSearchResult(rows=())

    async def get_invoice(self, params: GetInvoiceParams, caller: ToolCaller) -> object:
        if self.get_invoice_error is not None:
            raise self.get_invoice_error
        row = InvoiceRow(
            invoice_id=params.invoice_id,
            status="awaiting_review",
            route=None,
            vendor_id=None,
            vendor_name=None,
            invoice_number=None,
            issue_date=None,
            total=None,
            received_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return InvoiceView(
            row=row,
            due_date=None,
            po_number=None,
            supplier_trn=None,
            buyer_trn=None,
            subtotal=None,
            vat_total=None,
            payee=None,
            document_iban_matches_payee=None,
            line_count=0,
            fields=(),
            reasons=(),
            flags=(),
            pending_proposals=(),
        )

    async def explain_decision(
        self, params: ExplainDecisionParams, caller: ToolCaller
    ) -> DecisionExplanation:
        return DecisionExplanation(
            invoice_id=params.invoice_id,
            status="awaiting_review",
            route=None,
            reasons=(),
            events=(),
        )

    async def vendor_history(
        self, params: VendorHistoryParams, caller: ToolCaller
    ) -> VendorHistory:
        return VendorHistory(
            vendor_id=params.vendor_id,
            name="Acme",
            trn=None,
            payee=None,
            accounts=(),
            bank_changes=(),
            invoices=(),
            invoice_count=0,
            flagged_count=0,
        )

    async def request_approval(
        self, params: RequestApprovalParams, caller: ToolCaller
    ) -> ProposalReceipt:
        if self.request_approval_error is not None:
            raise self.request_approval_error
        return ProposalReceipt(
            proposal_id="appr_1", kind="approve_payment", invoice_id=params.invoice_id, note="n"
        )

    async def draft_vendor_email(self, params: object, caller: ToolCaller) -> object:
        raise AssertionError("not exercised by these tests")


def _dispatcher(toolbox: AgentToolbox | None = None) -> PgToolDispatcher:
    return PgToolDispatcher(toolbox=toolbox or _StubToolbox())  # type: ignore[arg-type]


def _call(name: str, arguments: dict[str, object], call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


# -- unknown_tool -----------------------------------------------------------------------------


async def test_unknown_tool_is_refused() -> None:
    execution = await _dispatcher().dispatch(_call("delete_invoice", {}), chat_caller(reviewer()))
    assert execution.trace.outcome == "refused"
    assert execution.trace.refusal == "unknown_tool"
    assert execution.trace.tool is None


# -- role_denied / out_of_scope (authorize_tool) -----------------------------------------------


async def test_role_denied_for_admin_alone_on_a_proposal_tool() -> None:
    execution = await _dispatcher().dispatch(
        _call(
            "request_approval",
            {"invoice_id": "inv-1", "rationale": "ok", "cited_audit_ids": [1]},
        ),
        chat_caller(admin()),
    )
    assert execution.trace.refusal == "role_denied"


async def test_out_of_scope_for_agent_origin_on_request_approval() -> None:
    execution = await _dispatcher().dispatch(
        _call(
            "request_approval",
            {"invoice_id": "inv-1", "rationale": "ok", "cited_audit_ids": [1]},
        ),
        agent_caller("inv-1"),
    )
    assert execution.trace.refusal == "out_of_scope"


# -- invalid_arguments --------------------------------------------------------------------------


async def test_invalid_arguments_for_a_missing_required_field() -> None:
    execution = await _dispatcher().dispatch(_call("get_invoice", {}), chat_caller(reviewer()))
    assert execution.trace.refusal == "invalid_arguments"


async def test_invalid_arguments_for_an_extra_field() -> None:
    execution = await _dispatcher().dispatch(
        _call("get_invoice", {"invoice_id": "inv-1", "unexpected": "x"}), chat_caller(reviewer())
    )
    assert execution.trace.refusal == "invalid_arguments"


# -- not_found / precondition_failed (AgentToolError from the toolbox) --------------------------


async def test_not_found_propagates_from_the_toolbox_as_a_refusal() -> None:
    toolbox = _StubToolbox()
    toolbox.get_invoice_error = ToolNotFoundError("no such invoice")
    execution = await _dispatcher(toolbox).dispatch(
        _call("get_invoice", {"invoice_id": "inv-1"}), chat_caller(reviewer())
    )
    assert execution.trace.refusal == "not_found"


async def test_precondition_failed_propagates_from_the_toolbox_as_a_refusal() -> None:
    toolbox = _StubToolbox()
    toolbox.request_approval_error = ToolPreconditionError("not in review")
    execution = await _dispatcher(toolbox).dispatch(
        _call(
            "request_approval",
            {"invoice_id": "inv-1", "rationale": "ok", "cited_audit_ids": [1]},
        ),
        chat_caller(reviewer()),
    )
    assert execution.trace.refusal == "precondition_failed"


# -- policy_denied (PolicyDeniedError is not an AgentToolError) --------------------------------


async def test_policy_denied_maps_a_policy_denied_error_to_a_refusal() -> None:
    toolbox = _StubToolbox()
    toolbox.request_approval_error = PolicyDeniedError("P5: not an agent proposal kind")
    execution = await _dispatcher(toolbox).dispatch(
        _call(
            "request_approval",
            {"invoice_id": "inv-1", "rationale": "ok", "cited_audit_ids": [1]},
        ),
        chat_caller(reviewer()),
    )
    assert execution.trace.refusal == "policy_denied"


# -- rate_limited (mcp only) --------------------------------------------------------------------


async def test_rate_limited_after_too_many_mcp_calls_in_one_session() -> None:
    dispatcher = _dispatcher()
    caller = mcp_caller(reviewer())
    call = _call("get_invoice", {"invoice_id": "inv-1"})

    for _ in range(MCP_MAX_CALLS_PER_MINUTE):
        execution = await dispatcher.dispatch(call, caller)
        assert execution.trace.outcome == "ok"

    over_limit = await dispatcher.dispatch(call, caller)
    assert over_limit.trace.refusal == "rate_limited"


async def test_rate_limit_does_not_apply_to_chat_origin() -> None:
    dispatcher = _dispatcher()
    caller = chat_caller(reviewer())
    call = _call("get_invoice", {"invoice_id": "inv-1"})

    for _ in range(MCP_MAX_CALLS_PER_MINUTE + 1):
        execution = await dispatcher.dispatch(call, caller)
        assert execution.trace.outcome == "ok"


# -- result_too_large -----------------------------------------------------------------------


async def test_result_too_large_when_content_exceeds_the_byte_cap() -> None:
    class _HugeToolbox(_StubToolbox):
        async def search_invoices(
            self, params: SearchInvoicesParams, caller: ToolCaller
        ) -> InvoiceSearchResult:
            huge_name = "x" * (TOOL_RESULT_MAX_BYTES + 1000)
            row = InvoiceRow(
                invoice_id=InvoiceId("inv-1"),
                status="awaiting_review",
                route=None,
                vendor_id=None,
                vendor_name=huge_name,
                invoice_number=None,
                issue_date=None,
                total=None,
                received_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            return InvoiceSearchResult(rows=(row,))

    execution = await _dispatcher(_HugeToolbox()).dispatch(
        _call("search_invoices", {}), chat_caller(reviewer())
    )
    assert execution.trace.refusal == "result_too_large"


# -- a non-tool error propagates instead of becoming a refusal ---------------------------------


async def test_a_plain_runtime_error_propagates_instead_of_becoming_a_refusal() -> None:
    toolbox = _StubToolbox()
    toolbox.get_invoice_error = RuntimeError("the database exploded")
    with pytest.raises(RuntimeError):
        await _dispatcher(toolbox).dispatch(
            _call("get_invoice", {"invoice_id": "inv-1"}), chat_caller(reviewer())
        )


async def test_a_malformed_caller_propagates_as_a_config_error_not_a_refusal() -> None:
    """A ConfigError from `authorize_tool` (the runtime built an invalid `ToolCaller`) is a bug,
    never a model-facing refusal (ADR-0010 D3)."""
    with pytest.raises(ConfigError):
        await _dispatcher().dispatch(
            _call("get_invoice", {"invoice_id": "inv-1"}),
            ToolCaller(origin="chat", user=None, session_id="s" * 8),
        )


# -- successful dispatch produces refs and an ok trace -----------------------------------------


async def test_successful_dispatch_has_ok_outcome_and_a_proposal_ref() -> None:
    execution = await _dispatcher().dispatch(
        _call(
            "request_approval",
            {"invoice_id": "inv-1", "rationale": "ok", "cited_audit_ids": [1]},
        ),
        chat_caller(approver()),
    )
    assert execution.trace.outcome == "ok"
    assert execution.trace.proposal_id == "appr_1"
    assert ("proposal", "appr_1") in {(c.source, c.ref) for c in execution.trace.refs}


async def test_refs_for_covers_every_non_empty_branch_through_dispatch() -> None:
    """F6: `_refs_for` (apx/agent/dispatch.py) has a branch per tool; exercise the ones a
    fixture with all-empty collections never reaches -- explain_decision's audit-id-per-event,
    get_invoice's reason/proposal refs, and vendor_history's bank-change audit ids -- by
    dispatching through a stub toolbox that returns rich results, so a regression to any one
    branch shows up as a missing ref rather than surviving on branch coverage alone."""

    class _RichToolbox(_StubToolbox):
        async def explain_decision(
            self, params: ExplainDecisionParams, caller: ToolCaller
        ) -> DecisionExplanation:
            return DecisionExplanation(
                invoice_id=params.invoice_id,
                status="blocked",
                route=Route.BLOCK,
                reasons=(
                    CitedReason(
                        rule="fraud_score",
                        route=Route.BLOCK,
                        message="High.",
                        audit_ids=(501,),
                    ),
                ),
                events=(
                    AuditEventView(
                        audit_id=501,
                        ts=datetime(2026, 1, 1, tzinfo=UTC),
                        actor="system",
                        action="policy.route",
                        facts={"rule": "fraud_score"},
                    ),
                    AuditEventView(
                        audit_id=502,
                        ts=datetime(2026, 1, 1, tzinfo=UTC),
                        actor="system",
                        action="fraud.flag",
                        facts={"rule": "fraud_score"},
                    ),
                ),
            )

        async def get_invoice(self, params: GetInvoiceParams, caller: ToolCaller) -> InvoiceView:
            row = InvoiceRow(
                invoice_id=params.invoice_id,
                status="awaiting_review",
                route=Route.REVIEW,
                vendor_id=None,
                vendor_name=None,
                invoice_number=None,
                issue_date=None,
                total=None,
                received_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            return InvoiceView(
                row=row,
                due_date=None,
                po_number=None,
                supplier_trn=None,
                buyer_trn=None,
                subtotal=None,
                vat_total=None,
                payee=None,
                document_iban_matches_payee=None,
                line_count=0,
                fields=(),
                reasons=(ReasonView(rule="missing_po", route=Route.REVIEW, message="No PO."),),
                flags=(),
                pending_proposals=(
                    ProposalView(
                        proposal_id="appr_pending",
                        kind="approve_payment",
                        status="pending",
                        created_by="chat:u1",
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                ),
            )

        async def vendor_history(
            self, params: VendorHistoryParams, caller: ToolCaller
        ) -> VendorHistory:
            invoice = InvoiceRow(
                invoice_id=InvoiceId("inv-9"),
                status="awaiting_review",
                route=Route.REVIEW,
                vendor_id=params.vendor_id,
                vendor_name="Acme",
                invoice_number=None,
                issue_date=None,
                total=None,
                received_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            return VendorHistory(
                vendor_id=params.vendor_id,
                name="Acme",
                trn=None,
                payee=None,
                accounts=(),
                bank_changes=(
                    BankChangeView(
                        changed_at=datetime(2026, 1, 1, tzinfo=UTC),
                        changed_by=UserId("u1"),
                        old_iban="****1111",
                        new_iban="****2222",
                        audit_id=777,
                    ),
                ),
                invoices=(invoice,),
                invoice_count=1,
                flagged_count=0,
            )

    dispatcher = _dispatcher(_RichToolbox())
    caller = chat_caller(reviewer())

    explain = await dispatcher.dispatch(
        _call("explain_decision", {"invoice_id": "inv-1"}, call_id="c1"), caller
    )
    refs = {(c.source, c.ref) for c in explain.trace.refs}
    assert {("audit", "501"), ("audit", "502"), ("invoice", "inv-1")} <= refs

    get_invoice = await dispatcher.dispatch(
        _call("get_invoice", {"invoice_id": "inv-1"}, call_id="c2"), caller
    )
    refs = {(c.source, c.ref) for c in get_invoice.trace.refs}
    assert {("invoice", "inv-1"), ("policy", "missing_po"), ("proposal", "appr_pending")} <= refs

    vendor_history = await dispatcher.dispatch(
        _call("vendor_history", {"vendor_id": "v-1"}, call_id="c3"), caller
    )
    refs = {(c.source, c.ref) for c in vendor_history.trace.refs}
    assert {("vendor", "v-1"), ("invoice", "inv-9"), ("audit", "777")} <= refs


# -- specs() only lists what a caller may use ---------------------------------------------------


def test_specs_excludes_request_approval_for_agent_origin() -> None:
    specs = _dispatcher().specs(agent_caller("inv-1"))
    assert "request_approval" not in {s.name for s in specs}


def test_specs_excludes_proposal_tools_for_a_role_without_them() -> None:
    specs = _dispatcher().specs(chat_caller(admin()))
    names = {s.name for s in specs}
    assert "request_approval" not in names
    assert "get_invoice" in names


def test_specs_includes_every_read_tool_for_any_authenticated_role() -> None:
    specs = _dispatcher().specs(mcp_caller(admin()))
    names = {s.name for s in specs}
    assert {"search_invoices", "get_invoice", "explain_decision", "vendor_history"} <= names
