"""Unit tests for the ADR-0010 D1/D3 tool contract table: there are exactly six tools with the
documented effects/origins/roles, MCP lists exactly the read and propose tools, every parameter
model is closed and schema-producing, `MaskedIban` rejects a raw IBAN, and `executed`/`sent`
accept only `False` (ADR-0010 gate)."""

from __future__ import annotations

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from apx.agent.interfaces import (
    MCP_TOOL_NAMES,
    TOOL_CONTRACTS,
    EmailDraftReceipt,
    MaskedIban,
    ProposalReceipt,
)
from apx.common.types import InvoiceId, Role, VendorId

_APPROVING_ROLES = frozenset({Role.REVIEWER, Role.APPROVER})
_ALL_ROLES = frozenset(Role)


def test_there_are_exactly_six_tools() -> None:
    assert set(TOOL_CONTRACTS) == {
        "search_invoices",
        "get_invoice",
        "explain_decision",
        "vendor_history",
        "request_approval",
        "draft_vendor_email",
    }


@pytest.mark.parametrize(
    "name", ["search_invoices", "get_invoice", "explain_decision", "vendor_history"]
)
def test_read_tools_are_free_for_every_role_and_origin(name: str) -> None:
    contract = TOOL_CONTRACTS[name]
    assert contract.effect == "read"
    assert contract.roles == _ALL_ROLES
    assert contract.origins == {"agent", "chat", "mcp"}


def test_request_approval_is_reviewer_and_approver_only_and_excludes_agent_origin() -> None:
    contract = TOOL_CONTRACTS["request_approval"]
    assert contract.effect == "propose"
    assert contract.roles == _APPROVING_ROLES
    assert contract.origins == {"chat", "mcp"}
    assert contract.proposal_kind == "approve_payment"


def test_draft_vendor_email_is_reviewer_and_approver_only_but_all_origins() -> None:
    contract = TOOL_CONTRACTS["draft_vendor_email"]
    assert contract.effect == "propose"
    assert contract.roles == _APPROVING_ROLES
    assert contract.origins == {"agent", "chat", "mcp"}
    assert contract.proposal_kind == "request_info"


def test_mcp_tool_names_equals_every_read_and_propose_tool() -> None:
    assert frozenset(TOOL_CONTRACTS) == MCP_TOOL_NAMES


@pytest.mark.parametrize("name", list(TOOL_CONTRACTS))
def test_every_param_model_forbids_extra_keys_and_has_a_schema(name: str) -> None:
    params_cls = TOOL_CONTRACTS[name].params
    assert params_cls.model_config.get("extra") == "forbid"
    schema = params_cls.model_json_schema()
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("name", list(TOOL_CONTRACTS))
def test_every_result_model_inherits_tool_output_and_has_a_schema(name: str) -> None:
    result_cls = TOOL_CONTRACTS[name].result
    schema = result_cls.model_json_schema()
    assert "truncated" in schema["properties"]


# -- MaskedIban rejects a raw IBAN (hypothesis) --------------------------------------------


@given(st.text(alphabet=string.ascii_uppercase + string.digits, min_size=15, max_size=34))
def test_masked_iban_rejects_a_plausible_raw_iban(raw: str) -> None:
    candidate = f"AE07{raw}"
    with pytest.raises(ValidationError):
        TypeAdapter(MaskedIban).validate_python(candidate)


def test_masked_iban_accepts_the_masked_form() -> None:
    assert TypeAdapter(MaskedIban).validate_python("****1234") == "****1234"


# -- executed/sent accept only False --------------------------------------------------------


def test_proposal_receipt_executed_rejects_true() -> None:
    with pytest.raises(ValidationError):
        ProposalReceipt(
            proposal_id="p1",
            kind="approve_payment",
            invoice_id=InvoiceId("inv1"),
            note="n",
            executed=True,  # type: ignore[arg-type]
        )


def test_proposal_receipt_executed_defaults_false() -> None:
    receipt = ProposalReceipt(
        proposal_id="p1", kind="approve_payment", invoice_id=InvoiceId("inv1"), note="n"
    )
    assert receipt.executed is False


def test_email_draft_receipt_sent_rejects_true() -> None:
    with pytest.raises(ValidationError):
        EmailDraftReceipt(
            proposal_id="p1",
            invoice_id=InvoiceId("inv1"),
            to_vendor_id=VendorId("v1"),
            subject="s",
            body="b",
            sent=True,  # type: ignore[arg-type]
        )


def test_email_draft_receipt_sent_defaults_false() -> None:
    receipt = EmailDraftReceipt(
        proposal_id="p1",
        invoice_id=InvoiceId("inv1"),
        to_vendor_id=VendorId("v1"),
        subject="s",
        body="b",
    )
    assert receipt.sent is False
