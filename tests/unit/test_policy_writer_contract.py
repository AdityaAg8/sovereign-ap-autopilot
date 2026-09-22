"""Structural test: every PaymentStateWriter method accepts a `Grant`, so no payment-state
mutator can be called without one (ADR-0008 K6, M6.5)."""

from __future__ import annotations

import typing

from apx.policy.interfaces import Grant, PaymentStateWriter


def _protocol_methods() -> list[str]:
    return [
        name
        for name in vars(PaymentStateWriter)
        if not name.startswith("_") and callable(getattr(PaymentStateWriter, name))
    ]


def test_payment_state_writer_has_methods() -> None:
    assert _protocol_methods(), "expected at least one PaymentStateWriter method"


def test_every_writer_method_takes_a_grant() -> None:
    for name in _protocol_methods():
        func = getattr(PaymentStateWriter, name)
        hints = typing.get_type_hints(func)
        hints.pop("return", None)
        assert Grant in hints.values(), f"PaymentStateWriter.{name} must take a `grant: Grant`"
