"""Unit and property tests for `apx.policy.interfaces.is_payment_affecting` (M6 part-1 review
finding F4): every direct capability and every EXECUTE_PROPOSAL proposal kind must be classified
exactly as ADR-0008's definition of I3's "payment-affecting change" requires (PRD §6.2 I3,
ADR-0008, M6.5)."""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from apx.policy.interfaces import (
    PAYMENT_AFFECTING,
    PROPOSAL_CAPABILITY,
    ActionRequest,
    Capability,
    Proposal,
    ProposalKind,
    is_payment_affecting,
)
from tests.unit._policy_factories import make_user

_AT = datetime(2026, 1, 1, tzinfo=UTC)
_ALL_PROPOSAL_KINDS: tuple[ProposalKind, ...] = (
    "edit_field",
    "route",
    "request_info",
    "approve_payment",
    "change_bank",
)


def _request(action: Capability, proposal: Proposal | None = None) -> ActionRequest:
    return ActionRequest(
        actor=make_user(), action=action, invoice=None, vendor=None, at=_AT, proposal=proposal
    )


def _proposal(kind: ProposalKind) -> Proposal:
    return Proposal(
        id="p1",
        invoice_id=None,
        kind=kind,
        payload={},
        created_by="agent",
        created_at=_AT,
    )


def test_every_direct_capability_is_classified_per_adr_0008() -> None:
    """PAYMENT_AFFECTING is exactly {APPROVE_PAYMENT, APPROVE_ABOVE_LIMIT, CHANGE_BANK_DETAILS,
    VERIFY_BANK_DETAILS} (ADR-0008 I3); every other capability, used directly (no proposal), is
    not payment-affecting."""
    for capability in Capability:
        request = _request(capability)
        expected = capability in PAYMENT_AFFECTING
        assert is_payment_affecting(request) is expected, capability


def test_execute_proposal_without_a_proposal_is_never_payment_affecting() -> None:
    """Fail-closed shape: EXECUTE_PROPOSAL with no attached proposal cannot be resolved to a
    mapped capability, so it must never be treated as payment-affecting."""
    request = _request(Capability.EXECUTE_PROPOSAL, proposal=None)
    assert is_payment_affecting(request) is False


def test_every_proposal_kind_via_execute_proposal_is_classified_per_adr_0008() -> None:
    """Only `approve_payment` and `change_bank` proposals are payment-affecting when executed;
    `edit_field`, `route` and `request_info` are not (ADR-0008 I3)."""
    for kind in _ALL_PROPOSAL_KINDS:
        request = _request(Capability.EXECUTE_PROPOSAL, proposal=_proposal(kind))
        expected = PROPOSAL_CAPABILITY[kind] in PAYMENT_AFFECTING
        assert is_payment_affecting(request) is expected, kind


@given(
    capability=st.sampled_from(list(Capability)),
    kind=st.sampled_from(_ALL_PROPOSAL_KINDS),
    attach_proposal=st.booleans(),
)
def test_is_payment_affecting_matches_the_adr_0008_definition(
    capability: Capability, kind: ProposalKind, attach_proposal: bool
) -> None:
    """Property: for any capability and any optional proposal, `is_payment_affecting` is true iff
    the direct capability is in PAYMENT_AFFECTING, or it is EXECUTE_PROPOSAL with an attached
    proposal whose mapped capability is in PAYMENT_AFFECTING."""
    proposal = _proposal(kind) if attach_proposal else None
    request = _request(capability, proposal=proposal)

    expected = capability in PAYMENT_AFFECTING or (
        capability is Capability.EXECUTE_PROPOSAL
        and proposal is not None
        and PROPOSAL_CAPABILITY.get(kind) in PAYMENT_AFFECTING
    )
    assert is_payment_affecting(request) is expected
