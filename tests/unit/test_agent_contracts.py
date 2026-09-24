"""Unit tests for apx.agent.contracts: role/origin scoping (authorize_tool), IBAN masking and
the audit-actor id normalisation reviewer M9.1 flagged as F4 (PRD §10 M9.2, ADR-0010 D1, D3, D4).
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from apx.agent.contracts import audit_actor, authorize_tool, mask_iban, tool_spec
from apx.agent.interfaces import (
    TOOL_CONTRACTS,
    MaskedIban,
    ToolArgumentError,
    ToolCaller,
    ToolRoleError,
    ToolScopeError,
)
from apx.common.errors import ConfigError
from apx.common.master import User
from apx.common.types import InvoiceId, Role, UserId

_GET_INVOICE = TOOL_CONTRACTS["get_invoice"]
_REQUEST_APPROVAL = TOOL_CONTRACTS["request_approval"]


def _reviewer(user_id: str = "u1") -> User:
    return User(id=UserId(user_id), display_name="Reviewer", roles=frozenset({Role.REVIEWER}))


def _admin(user_id: str = "u2") -> User:
    return User(id=UserId(user_id), display_name="Admin", roles=frozenset({Role.ADMIN}))


# -- tool_spec ---------------------------------------------------------------------------


def test_tool_spec_builds_name_description_and_schema() -> None:
    spec = tool_spec(_GET_INVOICE)
    assert spec.name == "get_invoice"
    assert spec.description == _GET_INVOICE.description
    assert spec.parameters == _GET_INVOICE.params.model_json_schema()


# -- authorize_tool: origin scoping -------------------------------------------------------


def test_authorize_tool_allows_agent_within_its_contract_origins() -> None:
    caller = ToolCaller(
        origin="agent", user=None, session_id="s" * 8, scope_invoice_id=InvoiceId("inv1")
    )
    authorize_tool(_GET_INVOICE, caller)  # does not raise


def test_authorize_tool_denies_agent_for_request_approval_origin() -> None:
    """ADR-0010 D3: request_approval is not offered to the agent origin at all."""
    caller = ToolCaller(
        origin="agent", user=None, session_id="s" * 8, scope_invoice_id=InvoiceId("inv1")
    )
    with pytest.raises(ToolScopeError):
        authorize_tool(_REQUEST_APPROVAL, caller)


def test_authorize_tool_denies_chat_origin_when_tool_excludes_it() -> None:
    only_agent = TOOL_CONTRACTS["get_invoice"].model_copy(update={"origins": frozenset({"agent"})})
    caller = ToolCaller(origin="chat", user=_reviewer(), session_id="s" * 8)
    with pytest.raises(ToolScopeError):
        authorize_tool(only_agent, caller)


# -- authorize_tool: role scoping ----------------------------------------------------------


def test_authorize_tool_allows_reviewer_to_request_approval() -> None:
    caller = ToolCaller(origin="chat", user=_reviewer(), session_id="s" * 8)
    authorize_tool(_REQUEST_APPROVAL, caller)  # does not raise


def test_authorize_tool_denies_admin_alone_for_request_approval() -> None:
    """ADR-0010 D3: an admin alone gets role_denied for a proposal tool."""
    caller = ToolCaller(origin="chat", user=_admin(), session_id="s" * 8)
    with pytest.raises(ToolRoleError):
        authorize_tool(_REQUEST_APPROVAL, caller)


def test_authorize_tool_allows_any_role_for_a_read_tool() -> None:
    caller = ToolCaller(origin="mcp", user=_admin(), session_id="s" * 8)
    authorize_tool(_GET_INVOICE, caller)  # does not raise


# -- authorize_tool: caller invariants (runtime bugs, not model refusals) -----------------


def test_authorize_tool_rejects_agent_caller_carrying_a_user() -> None:
    caller = ToolCaller(
        origin="agent", user=_reviewer(), session_id="s" * 8, scope_invoice_id=InvoiceId("inv1")
    )
    with pytest.raises(ConfigError):
        authorize_tool(_GET_INVOICE, caller)


def test_authorize_tool_rejects_agent_caller_without_scope() -> None:
    caller = ToolCaller(origin="agent", user=None, session_id="s" * 8)
    with pytest.raises(ConfigError):
        authorize_tool(_GET_INVOICE, caller)


def test_authorize_tool_rejects_chat_caller_without_user() -> None:
    caller = ToolCaller(origin="chat", user=None, session_id="s" * 8)
    with pytest.raises(ConfigError):
        authorize_tool(_GET_INVOICE, caller)


# -- mask_iban -----------------------------------------------------------------------------


def test_mask_iban_keeps_last_four_alphanumerics() -> None:
    assert mask_iban(" ae07 0331 2345 6789 0123 456 ") == "****3456"


def test_mask_iban_rejects_too_short_a_value() -> None:
    with pytest.raises(ToolArgumentError):
        mask_iban("ab")


@given(st.text(alphabet=string.ascii_uppercase + string.digits, min_size=4, max_size=34))
def test_mask_iban_output_always_satisfies_masked_iban(raw: str) -> None:
    masked = mask_iban(raw)
    TypeAdapter(MaskedIban).validate_python(masked)


@given(st.text(max_size=3))
def test_masked_iban_type_rejects_a_raw_iban_string(raw: str) -> None:
    """ADR-0010 M9.2 gate: MaskedIban rejects raw IBANs."""
    with pytest.raises(ValidationError):
        TypeAdapter(MaskedIban).validate_python(f"AE07{raw}0331234567890123456")


# -- audit_actor (reviewer M9.1 F4) --------------------------------------------------------


def test_audit_actor_for_agent_origin_is_the_literal_agent() -> None:
    caller = ToolCaller(
        origin="agent", user=None, session_id="s" * 8, scope_invoice_id=InvoiceId("inv1")
    )
    assert audit_actor(caller) == "agent"


def test_audit_actor_for_chat_origin_prefixes_the_user_id() -> None:
    caller = ToolCaller(origin="chat", user=_reviewer("u-123"), session_id="s" * 8)
    assert audit_actor(caller) == "chat:u-123"


def test_audit_actor_for_mcp_origin_prefixes_the_user_id() -> None:
    caller = ToolCaller(origin="mcp", user=_reviewer("u.123"), session_id="s" * 8)
    assert audit_actor(caller) == "mcp:u.123"


def test_audit_actor_strips_surrounding_whitespace() -> None:
    caller = ToolCaller(origin="chat", user=_reviewer("  u1  "), session_id="s" * 8)
    assert audit_actor(caller) == "chat:u1"


def test_audit_actor_rejects_a_user_id_with_illegal_characters() -> None:
    """A JWT `sub` can be anything (UserId has no format constraint); an id like an email
    address would corrupt the AuditActor pattern if silently rewritten, so this must fail
    loudly instead (reviewer M9.1 F4)."""
    caller = ToolCaller(origin="chat", user=_reviewer("someone@example.com"), session_id="s" * 8)
    with pytest.raises(ConfigError):
        audit_actor(caller)


def test_audit_actor_rejects_an_empty_user_id() -> None:
    caller = ToolCaller(origin="chat", user=_reviewer("   "), session_id="s" * 8)
    with pytest.raises(ConfigError):
        audit_actor(caller)


def test_audit_actor_rejects_agent_origin_carrying_a_user() -> None:
    caller = ToolCaller(
        origin="agent", user=_reviewer(), session_id="s" * 8, scope_invoice_id=InvoiceId("inv1")
    )
    with pytest.raises(ConfigError):
        audit_actor(caller)


def test_audit_actor_rejects_chat_origin_without_a_user() -> None:
    caller = ToolCaller(origin="chat", user=None, session_id="s" * 8)
    with pytest.raises(ConfigError):
        audit_actor(caller)
