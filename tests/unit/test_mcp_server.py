"""An MCP client (the SDK's in-memory `Client`) lists and calls the tools of
`ApxMcpServer` (ADR-0010 D8): exactly MCP_TOOL_NAMES with the D8 annotations, calls routed
through the real `PgToolDispatcher` as origin mcp, reads audited as `mcp.read`, refusals as
`mcp.refusal`, and the token checked at start-up and on every call."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from mcp.client import Client

from apx.agent.dispatch import PgToolDispatcher
from apx.agent.interfaces import (
    MCP_TOOL_NAMES,
    GetInvoiceParams,
    InvoiceRow,
    InvoiceSearchResult,
    InvoiceView,
    ProposalReceipt,
    RequestApprovalParams,
    SearchInvoicesParams,
    ToolCaller,
)
from apx.agent.mcp_server import ApxMcpServer, JwtMcpAuthenticator
from apx.api.interfaces import TokenClaims
from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import PolicyDeniedError
from apx.common.master import User
from apx.common.types import Role, UserId
from tests.unit._agent_factories import reviewer


class _Toolbox:
    def __init__(self) -> None:
        self.callers: list[ToolCaller] = []

    async def search_invoices(
        self, params: SearchInvoicesParams, caller: ToolCaller
    ) -> InvoiceSearchResult:
        self.callers.append(caller)
        return InvoiceSearchResult(rows=())

    async def get_invoice(self, params: GetInvoiceParams, caller: ToolCaller) -> InvoiceView:
        self.callers.append(caller)
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

    async def request_approval(
        self, params: RequestApprovalParams, caller: ToolCaller
    ) -> ProposalReceipt:
        self.callers.append(caller)
        return ProposalReceipt(
            proposal_id="appr_1", kind="approve_payment", invoice_id=params.invoice_id, note="n"
        )


class _Audit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry.model_construct(id=len(self.events))


class _Auth:
    def __init__(self, user: User) -> None:
        self.user = user
        self.valid = True

    def authenticate(self, token: str) -> User:
        if not self.valid or token != "good":
            raise PolicyDeniedError("invalid or expired token")
        return self.user


def _app(auth: _Auth | None = None) -> tuple[ApxMcpServer, _Toolbox, _Audit, _Auth]:
    toolbox, audit = _Toolbox(), _Audit()
    auth = auth or _Auth(reviewer())
    app = ApxMcpServer(PgToolDispatcher(toolbox), auth, audit, "good")  # type: ignore[arg-type]
    return app, toolbox, audit, auth


async def test_client_lists_exactly_the_mcp_tools_with_d8_annotations() -> None:
    app, *_ = _app()
    async with Client(app.server) as client:
        listed = await client.list_tools()
    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == set(MCP_TOOL_NAMES)
    read = tools["get_invoice"].annotations
    assert read is not None and read.read_only_hint is True and read.open_world_hint is False
    propose = tools["request_approval"].annotations
    assert propose is not None and propose.read_only_hint is False
    assert propose.destructive_hint is False and propose.idempotent_hint is True
    email = tools["draft_vendor_email"].annotations
    assert email is not None and email.idempotent_hint is False
    assert tools["get_invoice"].input_schema["type"] == "object"


async def test_client_calls_a_read_tool_as_origin_mcp_and_it_is_audited() -> None:
    app, toolbox, audit, _ = _app()
    async with Client(app.server) as client:
        result = await client.call_tool("get_invoice", {"invoice_id": "inv_1"})
    assert result.is_error is False
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["row"]["invoice_id"] == "inv_1"
    assert [c.origin for c in toolbox.callers] == ["mcp"]
    assert [(e.action, e.actor, e.entity) for e in audit.events] == [
        ("mcp.read", "mcp:u1", "tool:get_invoice")
    ]
    assert "inv_1" not in json.dumps(audit.events[0].payload)  # arguments are hashed


async def test_proposal_tool_returns_an_unexecuted_receipt_and_is_not_a_read() -> None:
    app, _, audit, _ = _app()
    async with Client(app.server) as client:
        result = await client.call_tool(
            "request_approval",
            {"invoice_id": "inv_1", "rationale": "ok", "cited_audit_ids": [1]},
        )
    assert result.is_error is False
    receipt = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert receipt["proposal_id"] == "appr_1" and receipt["executed"] is False
    assert audit.events == []  # the proposal sink audits proposals, not the MCP layer


async def test_invalid_arguments_are_a_refusal_and_audited() -> None:
    app, toolbox, audit, _ = _app()
    async with Client(app.server) as client:
        result = await client.call_tool("get_invoice", {"invoice_id": 7, "extra": True})
    assert result.is_error is True
    assert toolbox.callers == []
    assert [e.action for e in audit.events] == ["mcp.refusal"]


async def test_unknown_tool_is_rejected_without_dispatch() -> None:
    app, toolbox, _, _ = _app()
    async with Client(app.server) as client:
        result = await client.call_tool("pay_invoice", {"invoice_id": "inv_1"})
    assert result.is_error is True and toolbox.callers == []


async def test_token_is_checked_at_start_up_and_on_every_call() -> None:
    auth = _Auth(reviewer())
    auth.valid = False
    with pytest.raises(PolicyDeniedError):
        _app(auth)
    app, toolbox, _, auth = _app()
    async with Client(app.server) as client:
        auth.valid = False  # the token expired mid-session
        result = await client.call_tool("get_invoice", {"invoice_id": "inv_1"})
    assert result.is_error is True and toolbox.callers == []
    assert "authentication failed" in result.content[0].text  # type: ignore[union-attr]


def test_jwt_authenticator_maps_claims_and_rejects_an_empty_token() -> None:
    class _Verifier:
        def verify(self, token: str) -> TokenClaims:
            return TokenClaims(
                sub=UserId("u9"),
                roles=frozenset({Role.APPROVER}),
                exp=datetime.now(UTC) + timedelta(minutes=5),
            )

    authenticator = JwtMcpAuthenticator(_Verifier())
    user = authenticator.authenticate("t")
    assert user.id == "u9" and user.roles == frozenset({Role.APPROVER})
    with pytest.raises(PolicyDeniedError):
        authenticator.authenticate("")
