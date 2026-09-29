"""Unit tests for evals.real.normalise (ADR-0019 §3): amount, count and text rules applied alike to
gold and prediction, and the rules hash the reports print. Pure CPU."""

from __future__ import annotations

import re

import pytest

from evals.real.normalise import RULES_SHA256, normalise


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("16,500", "16500"),
        ("45.500", "45500"),
        ("Rp. 16.500", "16500"),
        ("1,234,567", "1234567"),
        ("16,500.00", "16500"),
        ("12.50", "12.5"),
        ("12,50", "12.5"),
        ("-1,000", "-1000"),
        ("AED 1,050.00", "1050"),
        ("0", "0"),
        ("0.00", "0"),
        ("16.500,-", "16500"),
    ],
)
def test_amounts(raw: str, expected: str) -> None:
    assert normalise("amount", raw) == expected


def test_amount_without_a_number_falls_back_to_text() -> None:
    assert normalise("amount", "  FREE  ") == "FREE"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2", "2"), ("2x", "2"), ("2 X", "2"), ("1.00", "1"), ("1.5", "1.5"), ("two", "two")],
)
def test_counts(raw: str, expected: str) -> None:
    assert normalise("count", raw) == expected


def test_count_infinity_is_text_not_a_crash() -> None:
    assert normalise("count", "Infinity") == "Infinity"


def test_text_is_nfkc_collapsed_and_case_sensitive() -> None:
    no_break_space = chr(0xA0)
    fullwidth_tea = "".join(map(chr, (0xFF34, 0xFF45, 0xFF41)))
    assert normalise("text", f"  REAL{no_break_space} GANACHE\n") == "REAL GANACHE"
    assert normalise("text", fullwidth_tea) == "Tea"
    assert normalise("text", "tea") != normalise("text", "TEA")


def test_none_and_blank_are_none() -> None:
    assert normalise("text", None) is None
    assert normalise("text", "   ") is None
    assert normalise("count", "") is None


def test_rules_sha256_is_a_hex_digest() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", RULES_SHA256)
