"""Value normalisation for scoring CORD and DocILE in the real-data track (ADR-0019 §3), applied to
gold and prediction alike: amounts to a canonical Decimal string, counts to integers, text to NFKC
with collapsed whitespace. Rules were fixed on CORD validation; `RULES_SHA256` names them."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, Literal

ValueKind = Literal["amount", "count", "text"]

_NUMBER: Final = re.compile(r"-?\d(?:[\d.,]*\d)?")
_THOUSANDS: Final = re.compile(r"[.,](?=\d{3}(?:$|[.,]))")
_TRAILING_X: Final = re.compile("\\s*[xX\u00d7]\\s*$")
_SPACE: Final = re.compile(r"\s+")


def _rules_sha256() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(source).hexdigest()


RULES_SHA256: Final[str] = _rules_sha256()
"""sha256 of this module's source (LF endings): any rule change, however small, changes it."""


def normalise_text(value: str) -> str | None:
    """NFKC, whitespace collapsed and trimmed, case kept; None when nothing is left."""
    text = _SPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    return text or None


def _decimal_text(number: Decimal) -> str:
    return "0" if number == 0 else format(number.normalize(), "f")


def normalise_amount(value: str) -> str | None:
    """The first number in `value` (currency marks and other text dropped), thousands separators
    removed (a `.` or `,` followed by exactly three digits, then the end or another separator),
    a remaining `,` read as the decimal point; falls back to text when no number parses."""
    text = unicodedata.normalize("NFKC", value)
    match = _NUMBER.search(text)
    if match is None:
        return normalise_text(value)
    digits = _THOUSANDS.sub("", match.group(0))
    if "," in digits and "." not in digits:
        digits = digits.replace(",", ".", 1)
    try:
        return _decimal_text(Decimal(digits))
    except InvalidOperation:
        return normalise_text(value)


def normalise_count(value: str) -> str | None:
    """An integer after stripping a trailing `x` (`2x` -> `2`, `1.00` -> `1`); text otherwise."""
    text = _TRAILING_X.sub("", unicodedata.normalize("NFKC", value)).strip()
    try:
        number = Decimal(text.replace(",", "."))
    except InvalidOperation:
        return normalise_text(text)
    if not number.is_finite():
        return normalise_text(text)
    if number == number.to_integral_value():
        return str(int(number))
    return _decimal_text(number)


def normalise(kind: ValueKind, value: str | None) -> str | None:
    """ADR-0019 §3 rules, frozen after CORD validation."""
    if value is None:
        return None
    if kind == "amount":
        return normalise_amount(value)
    if kind == "count":
        return normalise_count(value)
    return normalise_text(value)
