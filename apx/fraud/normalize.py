"""Pure normalization helpers for fraud matching: invoice numbers and IBANs, so duplicate,
split and shared-IBAN rules compare like with like regardless of source formatting."""

from __future__ import annotations

import re

_ARABIC_INDIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
"""Arabic-Indic (U+0660-U+0669) and Extended Arabic-Indic / Persian (U+06F0-U+06F9) digits."""

_SEPARATOR_RE = re.compile(r"[\s\-_/\\.,#:]+")
_DIGIT_RUN_RE = re.compile(r"\d+")


def normalize_invoice_number(raw: str) -> str:
    """Canonical form of an invoice number for duplicate matching.

    Converts Arabic-Indic digits to ASCII, strips separators (spaces, hyphens, slashes,
    underscores, dots, commas, colons, hashes), upper-cases the remainder, then drops leading
    zeros from each run of digits (an all-zero run collapses to a single "0"). "INV-0007",
    "inv0007" and an Arabic-Indic-digit "INV-0007" all normalize to "INV7", as does "INV-7"."""
    text = raw.translate(_ARABIC_INDIC_DIGITS)
    text = _SEPARATOR_RE.sub("", text)
    text = text.upper()
    return _DIGIT_RUN_RE.sub(lambda m: m.group(0).lstrip("0") or "0", text)


def normalize_iban(raw: str) -> str:
    """Upper-case, separator-stripped IBAN for equality comparisons across free-text sources."""
    return _SEPARATOR_RE.sub("", raw).upper()
