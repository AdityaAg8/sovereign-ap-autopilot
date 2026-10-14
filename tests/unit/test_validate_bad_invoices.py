"""Loads every crafted bad invoice in `tests/fixtures/bad_invoices/` and asserts it is caught by
exactly the one `ViolationCode` the fixture was built for, no more and no fewer
(acceptance: "30 crafted bad invoices each caught by exactly the expected rule")."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from apx.common.invoice import Invoice
from apx.validate.interfaces import ViolationCode, validate_invoice

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "bad_invoices"
_TODAY = date(2026, 9, 22)  # fixed clock: fixtures use 2025 issue/due dates well in the past


def _load_fixtures() -> list[tuple[str, dict[str, Any]]]:
    paths = sorted(_FIXTURE_DIR.glob("*.json"))
    return [(path.name, json.loads(path.read_text())) for path in paths]


_FIXTURES = _load_fixtures()


def test_fixture_directory_has_thirty_cases() -> None:
    assert len(_FIXTURES) == 30


@pytest.mark.parametrize("name,payload", _FIXTURES, ids=[name for name, _ in _FIXTURES])
def test_bad_invoice_triggers_exactly_expected_rule(name: str, payload: dict[str, Any]) -> None:
    invoice = Invoice.model_validate(payload["invoice"])
    expected = ViolationCode(payload["rule_id"])

    report = validate_invoice(invoice, today=_TODAY)

    codes = {violation.code for violation in report.violations}
    assert codes == {expected}, (
        f"{name}: expected exactly {{{expected}}}, got {codes} "
        f"({[(v.code, v.field, v.message) for v in report.violations]})"
    )
    # `passed` only tracks error-severity violations; a warning-only fixture (e.g. a
    # non-AED invoice) is still "caught" by its rule_id without failing the invoice outright.
    if any(v.severity == "error" for v in report.violations):
        assert not report.passed
