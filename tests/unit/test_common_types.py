"""Unit tests for `apx.common.types.make_idempotency_key`: the deterministic key every
side-effecting activity in `apx.workflows` carries."""

from __future__ import annotations

import pytest

from apx.common.errors import IdempotencyKeyError
from apx.common.types import InvoiceId, make_idempotency_key


def test_builds_invoice_id_colon_step() -> None:
    key = make_idempotency_key(InvoiceId("abc123"), "asp_send")
    assert key == "abc123:asp_send"


def test_deterministic_for_same_inputs() -> None:
    first = make_idempotency_key(InvoiceId("abc123"), "archive")
    second = make_idempotency_key(InvoiceId("abc123"), "archive")
    assert first == second


@pytest.mark.parametrize("step", ["ASP_SEND", "1step", "-step", "step name", ""])
def test_rejects_non_lowercase_identifier_step(step: str) -> None:
    with pytest.raises(IdempotencyKeyError):
        make_idempotency_key(InvoiceId("abc123"), step)


@pytest.mark.parametrize("invoice_id", ["abc:123", "abc 123", "abc\t123"])
def test_rejects_invoice_id_with_colon_or_whitespace(invoice_id: str) -> None:
    with pytest.raises(IdempotencyKeyError):
        make_idempotency_key(InvoiceId(invoice_id), "archive")
