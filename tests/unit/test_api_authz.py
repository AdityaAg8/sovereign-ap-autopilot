"""Unit tests for the ADR-0008 K1 role capability matrix and the K4 proposal-kind capability
map: pure functions, no database needed. The database-backed parts of authorisation (P1
verified payee, P2 SoD) are covered end to end in `tests/integration/test_api_authz.py`
against a real Postgres container."""

from __future__ import annotations

from apx.common.types import Role
from apx.db.enums import ProposalKind
from apx.policy.interfaces import PROPOSAL_CAPABILITY, Capability, capabilities_for


def test_reviewer_can_edit_fields_and_approve_but_not_above_limit() -> None:
    caps = capabilities_for(frozenset({Role.REVIEWER}))
    assert Capability.EDIT_FIELD in caps
    assert Capability.APPROVE_PAYMENT in caps
    assert Capability.APPROVE_ABOVE_LIMIT not in caps


def test_approver_can_approve_above_limit_but_not_edit_fields() -> None:
    caps = capabilities_for(frozenset({Role.APPROVER}))
    assert Capability.APPROVE_PAYMENT in caps
    assert Capability.APPROVE_ABOVE_LIMIT in caps
    assert Capability.EDIT_FIELD not in caps


def test_admin_alone_cannot_approve_payments() -> None:
    caps = capabilities_for(frozenset({Role.ADMIN}))
    assert Capability.APPROVE_PAYMENT not in caps
    assert Capability.APPROVE_ABOVE_LIMIT not in caps
    assert Capability.CHANGE_BANK_DETAILS in caps


def test_admin_plus_approver_can_approve_payments() -> None:
    caps = capabilities_for(frozenset({Role.ADMIN, Role.APPROVER}))
    assert Capability.APPROVE_PAYMENT in caps
    assert Capability.APPROVE_ABOVE_LIMIT in caps
    assert Capability.CHANGE_BANK_DETAILS in caps


def test_capabilities_are_the_union_of_every_held_role() -> None:
    caps = capabilities_for(frozenset({Role.REVIEWER, Role.ADMIN}))
    assert Capability.EDIT_FIELD in caps
    assert Capability.MANAGE_USERS in caps


def test_no_roles_grants_no_capabilities() -> None:
    assert capabilities_for(frozenset()) == frozenset()


def test_every_proposal_kind_maps_to_a_capability() -> None:
    for kind in ProposalKind:
        assert kind.value in PROPOSAL_CAPABILITY


def test_change_bank_proposal_requires_the_admin_only_capability() -> None:
    required = PROPOSAL_CAPABILITY["change_bank"]
    assert required == Capability.CHANGE_BANK_DETAILS
    assert required not in capabilities_for(frozenset({Role.REVIEWER}))
    assert required not in capabilities_for(frozenset({Role.APPROVER}))
    assert required in capabilities_for(frozenset({Role.ADMIN}))
