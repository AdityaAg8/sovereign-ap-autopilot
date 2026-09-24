"""Integration test for the ADR-0010 D2 read-only boundary against a real, throwaway Postgres
container: `apx_agent_ro` can SELECT only the three masked views and a write through its
session fails at the database level; every one of the six tools is run through the real
`PgSqlReader` and `PgAgentToolbox`, proving their SQL is parameterised (a SQL-injection-style
`vendor_query` neither errors nor changes any row) and that no tool call changes payment or bank
state (row counts and full-row snapshots of `invoices`/`bank_details` are identical before and
after, and only the propose tools add exactly one `proposals` row) (PRD §10 M9.2). Marked
`integration` (needs Docker), so it is excluded from the default `make test`/CI selection and
runs via `pytest -m integration`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

from apx.agent.interfaces import (
    DraftVendorEmailParams,
    ExplainDecisionParams,
    GetInvoiceParams,
    RequestApprovalParams,
    SearchInvoicesParams,
    ToolCaller,
    VendorHistoryParams,
)
from apx.agent.sql import PgSqlReader
from apx.agent.toolbox import PgAgentToolbox
from apx.common.master import User
from apx.common.types import InvoiceId, Role, UserId, VendorId
from apx.policy.interfaces import Proposal

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-m9-2"
HOST_PORT = 55439
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"
AGENT_PASSWORD = "apx-agent-ro-dev-only"  # matches db/migrations/versions/0002_*.py

APP_DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
APP_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
AGENT_DSN = f"postgresql://apx_agent_ro:{AGENT_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
AGENT_DATABASE_URL = (
    f"postgresql+asyncpg://apx_agent_ro:{AGENT_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
)

INV1_ID = "1" * 64
"""`invoices.id` must itself look like a sha256 digest (`ck_invoices_id_is_sha256`); it equals
`document_sha256` for ingested invoices, so the test uses the same fixed digest for both."""
INV2_ID = "2" * 64
_IBAN = "AE070331234567890123456"


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    return cfg


async def _wait_for_postgres(dsn: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return
    raise TimeoutError(f"Postgres did not become ready within {timeout_s}s") from last_error


async def _seed(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            "INSERT INTO users (id, email, display_name, password_hash) VALUES ($1, $2, $3, $4)",
            "u1",
            "u1@example.com",
            "Test User",
            "hash",
        )
        await conn.execute(
            "INSERT INTO vendors (id, name, trn) VALUES ($1, $2, $3)",
            "v1",
            "Acme Supplies LLC",
            "123456789012345",
        )
        verification_id = await conn.fetchval(
            "INSERT INTO bank_verifications (vendor_id, iban, verified_by, method) "
            "VALUES ($1, $2, $3, $4) RETURNING id",
            "v1",
            _IBAN,
            "u1",
            "admin_callback",
        )
        await conn.execute(
            "INSERT INTO bank_details "
            "(vendor_id, iban, source, verification_id, is_current, changed_by) "
            "VALUES ($1, $2, 'admin_callback', $3, true, $4)",
            "v1",
            _IBAN,
            verification_id,
            "u1",
        )
        for invoice_id in (INV1_ID, INV2_ID):
            await conn.execute(
                "INSERT INTO invoices "
                "(id, document_sha256, vendor_id, language, number, issue_date, currency, "
                " subtotal, vat_total, total, iban, status, route) "
                "VALUES ($1, $1, $2, 'en', $3, '2026-01-05', 'AED', 100.00, 5.00, 105.00, "
                "$4, 'awaiting_review', 'review')",
                invoice_id,
                "v1",
                f"INV-{invoice_id[:8]}",
                _IBAN,
            )
        await conn.execute(
            "INSERT INTO fields (invoice_id, name, value, normalized_value, confidence, source) "
            "VALUES ($1, 'supplier_name', $2, $2, 0.95, 'arbiter'), "
            "($1, 'supplier_trn', '123456789012345', '123456789012345', 0.9, 'arbiter')",
            INV1_ID,
            "ACME SUPPLIES CO <script>alert(1)</script>",
        )
        await conn.execute(
            "INSERT INTO decisions (invoice_id, action, route, reasons) "
            "VALUES ($1, 'policy_route', 'review', $2::jsonb)",
            INV1_ID,
            '[{"rule": "fraud_score", "route": "review", "message": "Fraud score exceeded."}, '
            '{"payee": {"masked_iban": "****3456", "match": "matched", "verified_count": 1}}]',
        )
        await conn.execute(
            "INSERT INTO audit_log (actor, action, entity, payload_json, prev_hash, hash) "
            "VALUES ('system', 'policy.route', $1, $2::jsonb, $3, $4)",
            f"invoice:{INV1_ID}",
            '{"rule": "fraud_score", "route": "review", "score": 0.91}',
            "0" * 64,
            "a" * 64,
        )
    finally:
        await conn.close()


async def _row_counts(dsn: str, tables: tuple[str, ...]) -> dict[str, int]:
    conn = await asyncpg.connect(dsn)
    try:
        return {t: await conn.fetchval(f"SELECT count(*) FROM {t}") for t in tables}
    finally:
        await conn.close()


async def _table_snapshot(dsn: str, table: str, order_by: str) -> list[dict[str, Any]]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(f"SELECT * FROM {table} ORDER BY {order_by}")
        return [dict(r) for r in rows]
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def postgres_ready() -> Iterator[None]:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{HOST_PORT}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    try:
        asyncio.run(_wait_for_postgres(APP_DSN))
        yield
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture(scope="module")
def migrated_and_seeded(
    postgres_ready: None, monkeypatch_module: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch_module.setenv("DATABASE_URL", APP_DATABASE_URL)
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    asyncio.run(_seed(APP_DSN))
    try:
        yield
    finally:
        command.downgrade(cfg, "base")


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    try:
        yield mp
    finally:
        mp.undo()


class _DirectSink:
    """A `ProposalSink` for this test only: inserts through the *privileged* app role (never
    `apx_agent_ro`), mirroring `DefaultProposalSink` closely enough to prove the toolbox writes
    nothing but a pending proposal (and one audit event) per call."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def submit(self, proposal: Proposal) -> Proposal:
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(
                "INSERT INTO proposals (id, invoice_id, kind, payload, created_by, status) "
                "VALUES ($1, $2, $3, $4::jsonb, $5, 'pending')",
                proposal.id,
                proposal.invoice_id,
                proposal.kind,
                _json_payload(proposal.payload),
                str(proposal.created_by),
            )
            await conn.execute(
                "INSERT INTO audit_log (actor, action, entity, payload_json, prev_hash, hash) "
                "VALUES ($1, 'proposal.submit', $2, '{}'::jsonb, $3, $4)",
                str(proposal.created_by),
                f"proposal:{proposal.id}",
                "0" * 64,
                hashlib.sha256(proposal.id.encode()).hexdigest(),
            )
        finally:
            await conn.close()
        return proposal


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


@pytest.fixture
def agent_engine(migrated_and_seeded: None) -> Iterator[Any]:
    engine = create_async_engine(AGENT_DATABASE_URL)
    yield engine
    asyncio.run(engine.dispose())


def _reviewer() -> User:
    return User(id=UserId("rev1"), display_name="Reviewer", roles=frozenset({Role.REVIEWER}))


def _chat_caller() -> ToolCaller:
    return ToolCaller(origin="chat", user=_reviewer(), session_id="s" * 8)


# -- D2: the read-only role ---------------------------------------------------------------


def test_apx_agent_ro_can_select_the_three_views(migrated_and_seeded: None) -> None:
    async def _check() -> None:
        conn = await asyncpg.connect(AGENT_DSN)
        try:
            for view in ("v_agent_invoices", "v_agent_vendor_accounts", "v_agent_audit"):
                await conn.fetch(f"SELECT * FROM {view} LIMIT 1")
        finally:
            await conn.close()

    asyncio.run(_check())


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO invoices (id, document_sha256, vendor_id, language, number, issue_date, "
        "currency, subtotal, vat_total, total, status) VALUES "
        "('hack', repeat('0', 64), 'v1', 'en', 'X', '2026-01-01', 'AED', 1, 0, 1, 'received')",
        "UPDATE invoices SET status = 'approved' WHERE id = 'inv1'",
        "DELETE FROM invoices WHERE id = 'inv1'",
        "UPDATE bank_details SET iban = 'AE000000000000000000000'",
        "INSERT INTO proposals (id, kind, payload, created_by) "
        "VALUES ('x', 'approve_payment', '{}'::jsonb, 'attacker')",
        "SELECT iban FROM bank_details",
        "SELECT * FROM fields",
        "SELECT * FROM audit_log",
    ],
)
def test_apx_agent_ro_cannot_write_or_read_outside_the_views(
    migrated_and_seeded: None, statement: str
) -> None:
    async def _attempt() -> None:
        conn = await asyncpg.connect(AGENT_DSN)
        try:
            with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                await conn.execute(statement)
        finally:
            await conn.close()

    asyncio.run(_attempt())


# -- read tools return correct, masked, typed data -----------------------------------------


def test_search_invoices_finds_the_seeded_invoice_by_vendor_query(agent_engine: Any) -> None:
    async def _run() -> None:
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        result = await toolbox.search_invoices(
            SearchInvoicesParams(vendor_query="Acme"), _chat_caller()
        )
        assert {row.invoice_id for row in result.rows} >= {INV1_ID, INV2_ID}

    asyncio.run(_run())


def test_search_invoices_with_a_sql_injection_style_vendor_query_changes_nothing(
    agent_engine: Any,
) -> None:
    async def _run() -> None:
        before = await _row_counts(APP_DSN, ("invoices", "vendors", "bank_details"))
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        result = await toolbox.search_invoices(
            SearchInvoicesParams(vendor_query="x' OR '1'='1"), _chat_caller()
        )
        after = await _row_counts(APP_DSN, ("invoices", "vendors", "bank_details"))
        assert before == after
        assert result.rows == ()

    asyncio.run(_run())


def test_get_invoice_masks_the_payee_and_matches_the_document_iban(agent_engine: Any) -> None:
    async def _run() -> None:
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        result = await toolbox.get_invoice(
            GetInvoiceParams(invoice_id=InvoiceId(INV1_ID)), _chat_caller()
        )
        assert result.payee == "****3456"
        assert result.document_iban_matches_payee is True
        assert result.supplier_trn == "123456789012345"
        # The raw IBAN must never appear anywhere in the typed content.
        content = result.model_dump(mode="json", exclude={"untrusted"})
        assert _IBAN not in str(content)

    asyncio.run(_run())


def test_get_invoice_payee_is_the_routed_selection_not_the_newest_account(
    agent_engine: Any,
) -> None:
    """ADR-0014 §7 (migration 0008): the view's payee is the one `select_payee` pinned in the
    `policy_route` decision, even when it differs from the vendor's newest verified account
    (****3456 here); the payee element never leaks into `reasons`, and an unrouted invoice has
    no payee at all."""
    inv3 = "3" * 64

    async def _seed_inv3() -> None:
        conn = await asyncpg.connect(APP_DSN)
        try:
            await conn.execute(
                "INSERT INTO invoices "
                "(id, document_sha256, vendor_id, language, number, issue_date, currency, "
                " subtotal, vat_total, total, iban, status, route) "
                "VALUES ($1, $1, 'v1', 'en', 'INV-3', '2026-01-05', 'AED', 100.00, 5.00, "
                "105.00, $2, 'awaiting_review', 'review')",
                inv3,
                _IBAN,
            )
            await conn.execute(
                "INSERT INTO decisions (invoice_id, action, route, reasons) "
                "VALUES ($1, 'policy_route', 'review', $2::jsonb)",
                inv3,
                '[{"rule": "P1", "route": "review", "message": "payee selected"}, '
                '{"payee": {"masked_iban": "****4321", "match": "sole_mismatch", '
                '"verified_count": 1}}]',
            )
        finally:
            await conn.close()

    async def _drop_inv3() -> None:
        conn = await asyncpg.connect(APP_DSN)
        try:
            await conn.execute("DELETE FROM invoices WHERE id = $1", inv3)
        finally:
            await conn.close()

    async def _run() -> None:
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        routed = await toolbox.get_invoice(
            GetInvoiceParams(invoice_id=InvoiceId(inv3)), _chat_caller()
        )
        assert routed.payee == "****4321"
        assert routed.payee_match == "sole_mismatch"
        assert routed.document_iban_matches_payee is False
        assert [r.rule for r in routed.reasons] == ["p1"]
        unrouted = await toolbox.get_invoice(
            GetInvoiceParams(invoice_id=InvoiceId(INV2_ID)), _chat_caller()
        )
        assert unrouted.payee is None and unrouted.document_iban_matches_payee is None

    asyncio.run(_seed_inv3())
    try:
        asyncio.run(_run())
    finally:
        asyncio.run(_drop_inv3())


def test_get_invoice_puts_the_script_tag_only_in_untrusted(agent_engine: Any) -> None:
    async def _run() -> None:
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        result = await toolbox.get_invoice(
            GetInvoiceParams(invoice_id=InvoiceId(INV1_ID)), _chat_caller()
        )
        content = result.model_dump(mode="json", exclude={"untrusted"})
        assert "<script>" not in str(content)
        assert any("<script>" in block.text for block in result.untrusted)

    asyncio.run(_run())


def test_explain_decision_cites_the_seeded_audit_event(agent_engine: Any) -> None:
    async def _run() -> None:
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        result = await toolbox.explain_decision(
            ExplainDecisionParams(invoice_id=InvoiceId(INV1_ID)), _chat_caller()
        )
        assert len(result.reasons) == 1
        assert result.reasons[0].rule == "fraud_score"
        cited_ids = set(result.reasons[0].audit_ids)
        assert cited_ids
        assert cited_ids <= {e.audit_id for e in result.events}

    asyncio.run(_run())


def test_vendor_history_returns_the_masked_account_and_seeded_invoices(agent_engine: Any) -> None:
    async def _run() -> None:
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        result = await toolbox.vendor_history(
            VendorHistoryParams(vendor_id=VendorId("v1")), _chat_caller()
        )
        assert result.name == "Acme Supplies LLC"
        assert result.payee == "****3456"
        assert all(a.iban.startswith("****") for a in result.accounts)
        assert {i.invoice_id for i in result.invoices} >= {INV1_ID, INV2_ID}

    asyncio.run(_run())


# -- propose tools write exactly one proposal, nothing else --------------------------------


def test_request_approval_writes_exactly_one_proposal_and_nothing_else(agent_engine: Any) -> None:
    async def _run() -> None:
        tables = ("invoices", "vendors", "bank_details", "proposals")
        before_counts = await _row_counts(APP_DSN, tables)
        before_invoices = await _table_snapshot(APP_DSN, "invoices", "id")
        before_bank = await _table_snapshot(APP_DSN, "bank_details", "id")

        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        receipt = await toolbox.request_approval(
            RequestApprovalParams(
                invoice_id=InvoiceId(INV1_ID), rationale="looks fine", cited_audit_ids=(1,)
            ),
            _chat_caller(),
        )

        after_counts = await _row_counts(APP_DSN, tables)
        after_invoices = await _table_snapshot(APP_DSN, "invoices", "id")
        after_bank = await _table_snapshot(APP_DSN, "bank_details", "id")

        assert receipt.executed is False
        assert after_counts["proposals"] == before_counts["proposals"] + 1
        assert after_counts["invoices"] == before_counts["invoices"]
        assert after_counts["bank_details"] == before_counts["bank_details"]
        assert after_invoices == before_invoices
        assert after_bank == before_bank

    asyncio.run(_run())


def test_draft_vendor_email_writes_exactly_one_proposal_and_never_sends(agent_engine: Any) -> None:
    async def _run() -> None:
        tables = ("invoices", "vendors", "bank_details", "proposals")
        before_counts = await _row_counts(APP_DSN, tables)

        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=_DirectSink(APP_DSN))
        receipt = await toolbox.draft_vendor_email(
            DraftVendorEmailParams(invoice_id=InvoiceId(INV2_ID), reason="missing_po"),
            _chat_caller(),
        )

        after_counts = await _row_counts(APP_DSN, tables)

        assert receipt.sent is False
        assert after_counts["proposals"] == before_counts["proposals"] + 1
        assert after_counts["invoices"] == before_counts["invoices"]
        assert after_counts["bank_details"] == before_counts["bank_details"]

    asyncio.run(_run())


# -- M9.4: an MCP client against the real stack ----------------------------------------------


def test_mcp_client_reads_and_proposes_through_the_real_stack(agent_engine: Any) -> None:
    """ADR-0010 D8 end to end: the SDK client lists and calls tools on `ApxMcpServer` wired like
    production (read-only role, `DefaultProposalSink` over `PgProposalStore`, `PgAuditLog`). The
    read is audited as `mcp.read`; the proposal lands as exactly one pending row."""
    from mcp.client import Client

    from apx.agent.dispatch import PgToolDispatcher
    from apx.agent.mcp_server import ApxMcpServer
    from apx.audit.chain import PgAuditLog
    from apx.db import session as db_session
    from apx.policy.proposals import DefaultProposalSink, PgProposalStore

    class _Auth:
        def authenticate(self, token: str) -> User:
            return _reviewer()

    async def _run() -> None:
        db_session.get_engine.cache_clear()  # bind the app engine to this event loop
        db_session._session_factory.cache_clear()
        audit = PgAuditLog()
        sink = DefaultProposalSink(PgProposalStore(), audit)
        toolbox = PgAgentToolbox(reader=PgSqlReader(agent_engine), sink=sink)
        app = ApxMcpServer(PgToolDispatcher(toolbox), _Auth(), audit, "token")
        before = await _row_counts(APP_DSN, ("proposals", "invoices"))
        try:
            async with Client(app.server) as client:
                names = {tool.name for tool in (await client.list_tools()).tools}
                read = await client.call_tool("get_invoice", {"invoice_id": INV1_ID})
                proposed = await client.call_tool(
                    "draft_vendor_email", {"invoice_id": INV2_ID, "reason": "missing_po"}
                )
        finally:
            await db_session.get_engine().dispose()
        after = await _row_counts(APP_DSN, ("proposals", "invoices"))
        conn = await asyncpg.connect(APP_DSN)
        try:
            actions = [r["action"] for r in await conn.fetch("SELECT action FROM audit_log")]
            status = await conn.fetchval(
                "SELECT status::text FROM proposals ORDER BY created_at DESC LIMIT 1"
            )
        finally:
            await conn.close()
        assert "get_invoice" in names and "request_approval" in names
        assert read.is_error is False and proposed.is_error is False
        assert json.loads(proposed.content[0].text)["sent"] is False  # type: ignore[union-attr]
        assert after["proposals"] == before["proposals"] + 1
        assert after["invoices"] == before["invoices"]
        assert "mcp.read" in actions and status == "pending"

    asyncio.run(_run())


# -- M9.3: ops chat over the real tools ----------------------------------------------------


def test_ops_chat_cites_real_audit_ids_and_links_them_to_the_audit_tab(agent_engine: Any) -> None:
    """PRD §10 M9.3: "why was X routed" is answered through the real dispatcher and toolbox;
    the reply cites the audit id explain_decision returned, the link resolves to the invoice's
    audit tab, and a citation no tool returned is replaced by the fixed refusal."""
    from apx.agent.dispatch import PgToolDispatcher
    from apx.agent.interfaces import AgentConfig, ChatTurn, Citation
    from apx.agent.prompts import DefaultAgentPromptBuilder
    from apx.agent.runner import LlmOpsChat, ToolLoop
    from apx.api.routers.chat import invoice_audit, reply_payload
    from apx.audit.chain import PgAuditLog
    from apx.db import session as db_session
    from apx.llm.interfaces import CompletionResult, ToolCall, ToolRequest

    class _Scripted:
        def __init__(self, final: str) -> None:
            self._replies = [
                CompletionResult(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="c1", name="explain_decision", arguments={"invoice_id": INV1_ID}
                        ),
                    ),
                    finish_reason="tool_calls",
                    model="ops-chat",
                    latency_ms=1.0,
                ),
                CompletionResult(
                    content=final, finish_reason="stop", model="ops-chat", latency_ms=1.0
                ),
            ]

        async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
            return self._replies.pop(0)

        async def complete_structured(self, request: Any) -> CompletionResult:
            raise AssertionError("not used")

        async def embed(self, model: str, texts: Any) -> list[list[float]]:
            raise AssertionError("not used")

    async def _run() -> None:
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()
        conn = await asyncpg.connect(APP_DSN)
        try:
            audit_id = await conn.fetchval(
                "SELECT id FROM audit_log WHERE entity = $1 AND action = 'policy.route'",
                f"invoice:{INV1_ID}",
            )
        finally:
            await conn.close()
        dispatcher = PgToolDispatcher(
            PgAgentToolbox(PgSqlReader(agent_engine), _DirectSink(APP_DSN))
        )
        prompts = DefaultAgentPromptBuilder()
        question = [ChatTurn(role="user", text="Why was this invoice sent to review?")]

        def _chat(final: str) -> LlmOpsChat:
            loop = ToolLoop(_Scripted(final), dispatcher, prompts, AgentConfig(model="ops-chat"))
            return LlmOpsChat(loop, dispatcher, prompts, PgAuditLog(), "ops-chat")

        try:
            reply = await _chat(f"The fraud score was over its limit [audit:{audit_id}].").respond(
                question, _reviewer()
            )
            payload = await reply_payload(reply)
            made_up = await _chat(f"Fraud [audit:{audit_id + 1000}].").respond(
                question, _reviewer()
            )
            events = await invoice_audit(INV1_ID, _reviewer())
        finally:
            await db_session.get_engine().dispose()

        assert reply.refusal is None
        assert reply.citations == (Citation(source="audit", ref=str(audit_id)),)
        assert payload["citations"][0]["href"] == f"/invoices/{INV1_ID}?tab=audit&event={audit_id}"
        assert made_up.refusal == "unverifiable"
        routed = next(e for e in events if e.id == audit_id)
        assert routed.action == "policy.route"
        assert routed.facts == {"rule": "fraud_score", "route": "review", "score": 0.91}

    asyncio.run(_run())
