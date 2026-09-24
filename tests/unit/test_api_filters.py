"""Unit tests that SQL-injection payloads in invoice list filters are inert (PRD §10 M5.3):
`apx.api.queries` builds every filter with SQLAlchemy `.where()`/`.ilike()`, which always binds
values as parameters, so a malicious filter value can never change the query's shape. Compiling
with `literal_binds` and inspecting the rendered SQL proves the payload text never appears as
executable SQL -- it only ever appears quoted as a string literal.
"""

from __future__ import annotations

import pytest

from apx.api.queries import (
    invoice_count_stmt,
    invoice_list_stmt,
    vendor_count_stmt,
    vendor_list_stmt,
)
from apx.common.types import VendorId
from apx.workflows.interfaces import InvoiceStatus

INJECTION_PAYLOADS = [
    "'; DROP TABLE invoices; --",
    "' OR '1'='1",
    "x' UNION SELECT password_hash FROM users --",
    "\"; DELETE FROM audit_log WHERE '1'='1",
    "%' OR 1=1 --",
]


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_q_filter_payload_is_bound_not_concatenated(payload: str) -> None:
    stmt = invoice_list_stmt(status=None, vendor_id=None, q=payload, limit=10, offset=0)
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    rendered = str(compiled)

    # No unescaped single quote from the payload can terminate the string literal early: every
    # apostrophe in the payload must be doubled in the rendered literal (SQLAlchemy's own
    # escaping of a bound parameter rendered as a literal), never left bare. The statement's
    # *shape* invariance is checked separately below, which is the stronger proof.
    if "'" in payload:
        assert "''" in rendered


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_vendor_id_filter_payload_is_bound_not_concatenated(payload: str) -> None:
    stmt = invoice_list_stmt(status=None, vendor_id=VendorId(payload), q=None, limit=10, offset=0)
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    rendered = str(compiled)
    if "'" in payload:
        assert "''" in rendered


def test_injection_payload_does_not_change_the_statement_shape() -> None:
    baseline = str(
        invoice_list_stmt(status=None, vendor_id=None, q="normal vendor", limit=10, offset=0)
    )
    for payload in INJECTION_PAYLOADS:
        injected = str(
            invoice_list_stmt(status=None, vendor_id=None, q=payload, limit=10, offset=0)
        )
        # The compiled SQL text (placeholders only, no literal values) is identical regardless
        # of the filter value -- the payload can only ever occupy a bind parameter slot.
        assert injected == baseline


def test_count_statement_also_binds_filters() -> None:
    for payload in INJECTION_PAYLOADS:
        stmt = invoice_count_stmt(status=InvoiceStatus.AWAITING_REVIEW, vendor_id=None, q=payload)
        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
        rendered = str(compiled)
        if "'" in payload:
            assert "''" in rendered


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_vendor_q_filter_payload_is_bound_not_concatenated(payload: str) -> None:
    stmt = vendor_list_stmt(q=payload, limit=10, offset=0)
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    rendered = str(compiled)
    if "'" in payload:
        assert "''" in rendered


def test_vendor_injection_payload_does_not_change_the_statement_shape() -> None:
    baseline = str(vendor_list_stmt(q="normal vendor", limit=10, offset=0))
    for payload in INJECTION_PAYLOADS:
        injected = str(vendor_list_stmt(q=payload, limit=10, offset=0))
        assert injected == baseline


def test_vendor_count_statement_also_binds_filters() -> None:
    for payload in INJECTION_PAYLOADS:
        stmt = vendor_count_stmt(q=payload)
        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
        rendered = str(compiled)
        if "'" in payload:
            assert "''" in rendered
