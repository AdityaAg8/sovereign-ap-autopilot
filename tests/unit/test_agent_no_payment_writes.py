"""Behavioural proof that no `apx.agent` tool changes payment state (P5, ADR-0008 K3, ADR-0010
D2): with a spy `SqlReader`/`ProposalSink`, every one of the six tools performs only `fetch_all`
reads and, for the two propose tools, exactly one `submit` of a pending proposal whose kind is in
`TOOL_PROPOSAL_KINDS`. A read tool given a `DenyingSink` must never call it at all."""

from __future__ import annotations

from apx.agent.interfaces import (
    TOOL_PROPOSAL_KINDS,
    DraftVendorEmailParams,
    ExplainDecisionParams,
    GetInvoiceParams,
    RequestApprovalParams,
    SearchInvoicesParams,
    VendorHistoryParams,
)
from apx.agent.sql import SQL_EXPLAIN_DECISION_EVENTS, SQL_GET_INVOICE, SQL_VENDOR_HEADER
from apx.agent.toolbox import PgAgentToolbox
from apx.common.types import InvoiceId, VendorId
from tests.unit._agent_factories import (
    DenyingSink,
    FakeReader,
    SpySink,
    chat_caller,
    invoice_row,
    reviewer,
)


async def test_search_invoices_never_calls_the_sink() -> None:
    reader = FakeReader()
    toolbox = PgAgentToolbox(reader=reader, sink=DenyingSink())
    await toolbox.search_invoices(SearchInvoicesParams(), chat_caller(reviewer()))
    # DenyingSink.submit raises AssertionError if called; reaching here proves it was not.


async def test_get_invoice_never_calls_the_sink() -> None:
    reader = FakeReader({SQL_GET_INVOICE: [invoice_row()]})
    toolbox = PgAgentToolbox(reader=reader, sink=DenyingSink())
    await toolbox.get_invoice(
        GetInvoiceParams(invoice_id=InvoiceId("inv-1")), chat_caller(reviewer())
    )


async def test_explain_decision_never_calls_the_sink() -> None:
    reader = FakeReader({SQL_GET_INVOICE: [invoice_row()], SQL_EXPLAIN_DECISION_EVENTS: []})
    toolbox = PgAgentToolbox(reader=reader, sink=DenyingSink())
    await toolbox.explain_decision(
        ExplainDecisionParams(invoice_id=InvoiceId("inv-1")), chat_caller(reviewer())
    )


async def test_vendor_history_never_calls_the_sink() -> None:
    reader = FakeReader(
        {
            SQL_VENDOR_HEADER: [
                {
                    "vendor_id": VendorId("v1"),
                    "vendor_name": "Acme",
                    "vendor_trn": None,
                    "payee_masked": None,
                }
            ]
        }
    )
    toolbox = PgAgentToolbox(reader=reader, sink=DenyingSink())
    await toolbox.vendor_history(
        VendorHistoryParams(vendor_id=VendorId("v1")), chat_caller(reviewer())
    )


async def test_read_tools_issue_only_select_statements() -> None:
    reader = FakeReader({SQL_GET_INVOICE: [invoice_row()], SQL_EXPLAIN_DECISION_EVENTS: []})
    toolbox = PgAgentToolbox(reader=reader, sink=DenyingSink())

    await toolbox.get_invoice(
        GetInvoiceParams(invoice_id=InvoiceId("inv-1")), chat_caller(reviewer())
    )
    await toolbox.explain_decision(
        ExplainDecisionParams(invoice_id=InvoiceId("inv-1")), chat_caller(reviewer())
    )

    assert reader.calls, "expected at least one fetch_all call"
    for statement, _params, _limit in reader.calls:
        assert statement.strip().upper().startswith("SELECT")


async def test_request_approval_submits_exactly_one_pending_proposal_of_an_agent_kind() -> None:
    reader = FakeReader(
        {
            SQL_GET_INVOICE: [invoice_row(route="review")],
            SQL_EXPLAIN_DECISION_EVENTS: [
                {
                    "audit_id": 501,
                    "ts": invoice_row()["received_at"],
                    "actor": "system",
                    "action": "policy.route",
                    "payload_json": {},
                }
            ],
        }
    )
    sink = SpySink()
    toolbox = PgAgentToolbox(reader=reader, sink=sink)

    await toolbox.request_approval(
        RequestApprovalParams(
            invoice_id=InvoiceId("inv-1"), rationale="ok", cited_audit_ids=(501,)
        ),
        chat_caller(reviewer()),
    )

    assert len(sink.submitted) == 1
    proposal = sink.submitted[0]
    assert proposal.status == "pending"
    assert proposal.kind in TOOL_PROPOSAL_KINDS


async def test_draft_vendor_email_submits_exactly_one_pending_proposal_of_an_agent_kind() -> None:
    reader = FakeReader({SQL_GET_INVOICE: [invoice_row(vendor_id="v1")]})
    sink = SpySink()
    toolbox = PgAgentToolbox(reader=reader, sink=sink)

    await toolbox.draft_vendor_email(
        DraftVendorEmailParams(invoice_id=InvoiceId("inv-1"), reason="missing_po"),
        chat_caller(reviewer()),
    )

    assert len(sink.submitted) == 1
    proposal = sink.submitted[0]
    assert proposal.status == "pending"
    assert proposal.kind in TOOL_PROPOSAL_KINDS


def test_every_proposal_kind_in_toolbox_py_is_a_fixed_string_literal() -> None:
    """AST proof that `kind=` on every `Proposal(...)` construction in `toolbox.py` is a
    constant string (`"approve_payment"` or `"request_info"`), never a name or attribute that
    could carry a caller-supplied value such as `change_bank` (ADR-0010 D4)."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "apx" / "agent" / "toolbox.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    kinds_seen: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "Proposal":
            continue
        for kw in node.keywords:
            if kw.arg != "kind":
                continue
            assert isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str), (
                f"Proposal(kind=...) at line {node.lineno} is not a fixed string literal"
            )
            kinds_seen.append(kw.value.value)

    assert kinds_seen, "expected at least one Proposal(kind=...) construction"
    assert set(kinds_seen) <= TOOL_PROPOSAL_KINDS
    assert "change_bank" not in kinds_seen
