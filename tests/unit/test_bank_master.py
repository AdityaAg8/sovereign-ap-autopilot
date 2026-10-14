"""Unit tests for the bank-master path: IBAN mod-97 validation at write time
(table plus Hypothesis properties), the route-level authz matrix and error mapping of
`apx.api.bank_routes` over a recording fake service, and the static proof that `apx.agent`
(agent, chat, MCP) never reaches the bank service. The database path is covered by
`tests/integration/test_bank_master.py`."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from apx.api.bank_routes import get_bank_service
from apx.api.bank_routes import router as bank_router
from apx.api.deps import get_current_user
from apx.api.errors import install_exception_handlers
from apx.bank.errors import (
    BankMasterConflictError,
    BankVendorNotFoundError,
    InvalidBankRequestError,
    InvalidIbanError,
)
from apx.bank.iban import mod97_valid, validated_iban
from apx.bank.service import (
    BankAccountRecord,
    BankAccountStatus,
    VerificationMethod,
    mask_iban,
)
from apx.common.errors import SegregationOfDutiesError
from apx.common.master import User
from apx.common.types import Role, UserId, VendorId
from apx.policy.interfaces import AGENT_PROPOSAL_KINDS

VALID_AE = "AE070331234567890123456"

# ---------------------------------------------------------------------------------------------
# IBAN validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iban",
    [VALID_AE, "GB82WEST12345698765432", "DE89370400440532013000", "SA0380000000608010167519"],
)
def test_known_valid_ibans_pass_mod97(iban: str) -> None:
    assert mod97_valid(iban)
    assert validated_iban(iban) == iban


@pytest.mark.parametrize(
    "iban",
    ["AE070331234567890123457", "GB82WEST12345698765433", "AE070331234567890000001"],
)
def test_checksum_failures_are_refused(iban: str) -> None:
    assert not mod97_valid(iban)
    with pytest.raises(InvalidIbanError, match="mod-97"):
        validated_iban(iban)


def test_print_form_is_normalized_to_electronic_form() -> None:
    assert validated_iban("ae07 0331 2345 6789 0123 456") == VALID_AE


@pytest.mark.parametrize(
    "raw",
    [
        "AE07" + chr(0x00A0) + "0331234567890123456",  # NBSP
        "AE07-0331-2345-6789-0123-456",
        "AE07\t0331234567890123456",
        chr(0x0410) + "E070331234567890123456",  # Cyrillic A
        "AE07033123456789012345" + chr(0x0666),  # Arabic-Indic digit
        "",
        "AE07",
    ],
)
def test_malformed_ibans_are_refused_without_echoing_the_input(raw: str) -> None:
    with pytest.raises(InvalidIbanError) as info:
        validated_iban(raw)
    assert raw not in str(info.value) or raw == ""


def _with_check_digits(country: str, bban: str) -> str:
    numeric = "".join(str(int(c, 36)) for c in bban + country + "00")
    check = 98 - int(numeric) % 97
    return f"{country}{check:02d}{bban}"


_COUNTRY = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=2)
_BBAN = st.text(alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=11, max_size=30)


@given(country=_COUNTRY, bban=_BBAN)
def test_every_iban_with_correct_check_digits_is_accepted(country: str, bban: str) -> None:
    iban = _with_check_digits(country, bban)
    assert validated_iban(iban) == iban


@given(country=_COUNTRY, bban=_BBAN, data=st.data())
def test_any_single_digit_substitution_is_refused(
    country: str, bban: str, data: st.DataObject
) -> None:
    iban = _with_check_digits(country, bban)
    digit_positions = [i for i, c in enumerate(iban) if c.isdigit() and i >= 2]
    position = data.draw(st.sampled_from(digit_positions))
    replacement = data.draw(st.sampled_from([d for d in "0123456789" if d != iban[position]]))
    mutated = iban[:position] + replacement + iban[position + 1 :]
    with pytest.raises(InvalidIbanError):
        validated_iban(mutated)


def test_mask_keeps_only_the_last_four_characters() -> None:
    assert mask_iban(VALID_AE) == "*" * 19 + "3456"


# ---------------------------------------------------------------------------------------------
# Route authz matrix and error mapping
# ---------------------------------------------------------------------------------------------


def _record(status: BankAccountStatus) -> BankAccountRecord:
    return BankAccountRecord(
        vendor_id=VendorId("v1"),
        masked_iban=mask_iban(VALID_AE),
        status=status,
        changed_by=UserId("admin1"),
        changed_at=datetime(2026, 9, 1, tzinfo=UTC),
        verified_by=None,
        verified_at=None,
        verification_method=None,
    )


class _FakeBankService:
    """Records calls; raises `error` when set. Stands in for `BankMasterService`, which is not
    the unit under test here (the routes are)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    async def propose_change(
        self,
        actor: User,
        vendor_id: VendorId,
        new_iban: str,
        *,
        expected_current_iban: str | None,
    ) -> BankAccountRecord:
        self.calls.append(("propose", str(actor.id)))
        if self.error is not None:
            raise self.error
        return _record(BankAccountStatus.PENDING_VERIFICATION)

    async def verify(
        self,
        actor: User,
        vendor_id: VendorId,
        iban: str,
        *,
        method: VerificationMethod,
        reference: str,
    ) -> BankAccountRecord:
        self.calls.append(("verify", str(actor.id)))
        if self.error is not None:
            raise self.error
        return _record(BankAccountStatus.VERIFIED)


def _client(service: _FakeBankService, roles: frozenset[Role]) -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(bank_router)
    user = User(id=UserId("u1"), display_name="u1", roles=roles)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_bank_service] = lambda: service
    return TestClient(app)


_CHANGE = {"new_iban": VALID_AE, "expected_current_iban": None}
_VERIFY = {"iban": VALID_AE, "method": "phone_callback_on_file", "reference": "CALL-2026-0917"}


@pytest.mark.parametrize(
    ("roles", "allowed"),
    [
        (frozenset({Role.REVIEWER}), False),
        (frozenset({Role.APPROVER}), False),
        (frozenset({Role.REVIEWER, Role.APPROVER}), False),
        (frozenset(), False),
        (frozenset({Role.ADMIN}), True),
        (frozenset({Role.ADMIN, Role.APPROVER}), True),
    ],
)
@pytest.mark.parametrize(("path", "body"), [("changes", _CHANGE), ("verifications", _VERIFY)])
def test_only_roles_holding_the_bank_capability_reach_the_service(
    roles: frozenset[Role], allowed: bool, path: str, body: dict[str, object]
) -> None:
    service = _FakeBankService()
    response = _client(service, roles).post(f"/vendors/v1/bank-details/{path}", json=body)
    if allowed:
        assert response.status_code == 200, response.text
        assert len(service.calls) == 1
    else:
        assert response.status_code == 403, response.text
        assert service.calls == []


def test_a_proposal_is_returned_pending_and_masked() -> None:
    response = _client(_FakeBankService(), frozenset({Role.ADMIN})).post(
        "/vendors/v1/bank-details/changes", json=_CHANGE
    )
    body = response.json()
    assert body["status"] == "pending_verification"
    assert VALID_AE not in response.text
    assert body["masked_iban"].endswith("3456")


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (SegregationOfDutiesError("verifier is changer"), 403),
        (BankVendorNotFoundError("no vendor"), 404),
        (BankMasterConflictError("stale"), 409),
        (InvalidIbanError("bad iban"), 422),
        (InvalidBankRequestError("bad reference"), 422),
    ],
)
def test_typed_service_errors_map_to_http_statuses(error: Exception, status: int) -> None:
    service = _FakeBankService()
    service.error = error
    response = _client(service, frozenset({Role.ADMIN})).post(
        "/vendors/v1/bank-details/verifications", json=_VERIFY
    )
    assert response.status_code == status, response.text


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("changes", {**_CHANGE, "verified": True}),
        ("changes", {"new_iban": "A" * 65}),
        ("verifications", {**_VERIFY, "method": "email_from_vendor"}),
        ("verifications", {**_VERIFY, "reference": ""}),
        ("verifications", {"iban": VALID_AE, "method": "in_person"}),
    ],
)
def test_malformed_bodies_never_reach_the_service(path: str, body: dict[str, object]) -> None:
    service = _FakeBankService()
    response = _client(service, frozenset({Role.ADMIN})).post(
        f"/vendors/v1/bank-details/{path}", json=body
    )
    assert response.status_code == 422, response.text
    assert service.calls == []


# ---------------------------------------------------------------------------------------------
# Agent, chat and MCP cannot reach the bank master
# ---------------------------------------------------------------------------------------------

_AGENT_PACKAGE = Path(__file__).resolve().parents[2] / "apx" / "agent"
_BANK_MODULES = ("apx.bank", "apx.api.bank_routes", "apx.api.app", "apx.policy.writer")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_apx_agent_never_imports_the_bank_master() -> None:
    files = sorted(_AGENT_PACKAGE.rglob("*.py"))
    assert files
    offending = {
        f"{path.name}: {module}"
        for path in files
        for module in _imported_modules(path)
        if any(module == banned or module.startswith(f"{banned}.") for banned in _BANK_MODULES)
    }
    assert not offending


def test_models_cannot_propose_a_bank_change() -> None:
    assert "change_bank" not in AGENT_PROPOSAL_KINDS
