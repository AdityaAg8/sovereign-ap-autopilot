"""Normalizers that turn raw `InvoiceFields` text (whatever script and format an extractor read
off the document) into canonical ASCII forms: digits, dates, amounts, TRNs and IBANs. These do
representation cleanup only; format and checksum validation is `apx.validate`'s job."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Final

from apx.common.errors import ApxError

# All non-ASCII literals below are written as `chr()`/`\uXXXX` code points rather than raw
# glyphs, so the source stays plain ASCII (readable in any editor/terminal encoding) and ruff's
# ambiguous-Unicode checks (RUF001-003) have nothing to flag.

_ARABIC_INDIC_DIGITS: Final[str] = "".join(chr(0x0660 + n) for n in range(10))
"""Arabic-Indic digits 0-9 (U+0660-0669), used in Arabic-script documents."""

_EXTENDED_ARABIC_INDIC_DIGITS: Final[str] = "".join(chr(0x06F0 + n) for n in range(10))
"""Eastern Arabic-Indic / Persian digits 0-9 (U+06F0-06F9), also seen in Arabic-script OCR."""

_DIGIT_TRANSLATION: Final[dict[int, str]] = {
    ord(glyph): str(value)
    for digits in (_ARABIC_INDIC_DIGITS, _EXTENDED_ARABIC_INDIC_DIGITS)
    for value, glyph in enumerate(digits)
}

_ARABIC_THOUSANDS_SEP: Final[str] = chr(0x066C)  # ARABIC THOUSANDS SEPARATOR
_ARABIC_DECIMAL_SEP: Final[str] = chr(0x066B)  # ARABIC DECIMAL SEPARATOR
_ARABIC_NBSP: Final[str] = chr(0x00A0)  # NO-BREAK SPACE, sometimes left by PDF/OCR extraction
_UNICODE_MINUS: Final[str] = chr(0x2212)  # MINUS SIGN, distinct from ASCII hyphen-minus

_AED_ARABIC: Final[str] = chr(0x062F) + "." + chr(0x0625)
"""Arabic abbreviation for AED: dal, full stop, hamza-on-alif (U+062F, U+0625)."""

_CURRENCY_TOKENS: Final[tuple[str, ...]] = ("AED", "aed", "Aed", _AED_ARABIC, "Dhs", "DHS", "dhs")

_ARABIC_MONTHS: Final[dict[str, int]] = {
    "يناير": 1,  # January
    "فبراير": 2,  # February
    "مارس": 3,  # March
    "أبريل": 4,  # April
    "ابريل": 4,  # April, spelled without the hamza
    "مايو": 5,  # May
    "يونيو": 6,  # June
    "يوليو": 7,  # July
    "أغسطس": 8,  # August
    "اغسطس": 8,  # August, spelled without the hamza
    "سبتمبر": 9,  # September
    "أكتوبر": 10,  # October
    "اكتوبر": 10,  # October, spelled without the hamza
    "نوفمبر": 11,  # November
    "ديسمبر": 12,  # December
}
"""Gregorian month names as written in formal Modern Standard Arabic invoices."""

_ENGLISH_MONTHS: Final[dict[str, int]] = {
    name: index
    for index, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}

_MONTH_NAMES: Final[dict[str, int]] = {**_ARABIC_MONTHS, **_ENGLISH_MONTHS}

_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
_DAY_FIRST_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$")
_NAMED_MONTH_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{1,2})\s+([^\d\s]+)\s+(\d{4})$")

_TRN_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"[\s\-]")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_AMOUNT_RE: Final[re.Pattern[str]] = re.compile(r"^\d+(\.\d+)?$")
MAX_AMOUNT_INTEGER_DIGITS: Final[int] = 26
"""Integer digits an amount may have and still quantize to cents in the default 28-digit decimal
context. A longer digit run (account or IBAN digits the grounder joined into one word run) is not
an amount: `NormalizationError`, never `decimal.InvalidOperation` at the caller's quantize."""


class NormalizationError(ApxError):
    """Raw field text could not be parsed into its normalized form."""


def convert_digits(text: str) -> str:
    """Convert Arabic-Indic (U+0660-0669) and Eastern Arabic-Indic (U+06F0-06F9) digits to ASCII
    0-9. ASCII digits and every other character pass through unchanged, so this is idempotent:
    `convert_digits(convert_digits(s)) == convert_digits(s)`."""
    return text.translate(_DIGIT_TRANSLATION)


def normalize_trn(raw: str) -> str:
    """Convert digits to ASCII and strip whitespace/dashes from a UAE TRN. Returns the resulting
    digit string; whether it is exactly 15 digits is `apx.validate.is_valid_trn`'s job. Raises
    `NormalizationError` if a non-digit character remains."""
    text = _TRN_STRIP_RE.sub("", convert_digits(raw))
    if not text or not text.isdigit():
        raise NormalizationError(f"TRN is not purely digits after normalization: {raw!r}")
    return text


def normalize_iban(raw: str) -> str:
    """Convert digits to ASCII, strip whitespace and uppercase an IBAN. Structural and mod-97
    checksum validation is `apx.validate.is_valid_iban`'s job. Raises `NormalizationError` on
    empty input."""
    text = _WHITESPACE_RE.sub("", convert_digits(raw)).upper()
    if not text:
        raise NormalizationError(f"empty IBAN: {raw!r}")
    return text


def normalize_amount(raw: str) -> Decimal:
    """Parse an amount in ASCII or Arabic-Indic digits, with Western (comma thousands / period
    decimal) or Arabic (U+066C thousands / U+066B decimal) separators and an optional AED /
    Arabic-AED currency marker, into an exact `Decimal`. A lone comma is always read as a
    thousands separator (this codebase never emits comma-decimal notation), so formatting a
    `Decimal` either way and normalizing it back is idempotent. Raises `NormalizationError` if no
    number can be recovered."""
    text = convert_digits(raw)
    text = text.translate({ord(_ARABIC_THOUSANDS_SEP): ",", ord(_ARABIC_DECIMAL_SEP): "."})
    for token in _CURRENCY_TOKENS:
        text = text.replace(token, "")
    text = text.replace(_ARABIC_NBSP, " ").strip()

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if text.startswith(("-", _UNICODE_MINUS)):
        negative = True
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()

    text = _WHITESPACE_RE.sub("", text)
    if not text:
        raise NormalizationError(f"no digits in amount: {raw!r}")

    if "," in text and "." in text:
        if text.rindex(",") > text.rindex("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", "")

    if not _AMOUNT_RE.fullmatch(text):
        raise NormalizationError(f"unparseable amount: {raw!r}")

    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise NormalizationError(f"unparseable amount: {raw!r}") from exc
    if value.adjusted() >= MAX_AMOUNT_INTEGER_DIGITS:
        raise NormalizationError(f"amount out of range: {raw!r}")
    return -value if negative else value


def normalize_date(raw: str) -> date:
    """Parse a date in ASCII or Arabic-Indic digits: ISO `YYYY-MM-DD` (or `/`), day-first numeric
    `DD-MM-YYYY` / `DD.MM.YYYY` / `DD/MM/YYYY`, or `DD <month name> YYYY` with an English or
    Arabic Gregorian month name. Day-first is assumed for numeric dates, matching UAE invoice
    convention. Idempotent: `normalize_date(d.isoformat()) == d` for any result `d`. Raises
    `NormalizationError` if no format matches or the date is invalid."""
    text = convert_digits(raw).strip()

    if match := _ISO_DATE_RE.match(text):
        year, month, day = (int(part) for part in match.groups())
        return _build_date(year, month, day, raw)

    if match := _DAY_FIRST_DATE_RE.match(text):
        day, month, year = (int(part) for part in match.groups())
        return _build_date(year, month, day, raw)

    if match := _NAMED_MONTH_DATE_RE.match(text):
        day_text, month_name, year_text = match.groups()
        named_month = _MONTH_NAMES.get(month_name.strip().lower())
        if named_month is None:
            raise NormalizationError(f"unrecognized month name in date: {raw!r}")
        return _build_date(int(year_text), named_month, int(day_text), raw)

    raise NormalizationError(f"unrecognized date format: {raw!r}")


def _build_date(year: int, month: int, day: int, raw: str) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise NormalizationError(f"invalid date: {raw!r}") from exc


_DATE_FIELDS: Final[frozenset[str]] = frozenset({"issue_date", "due_date"})
_AMOUNT_FIELDS: Final[frozenset[str]] = frozenset({"subtotal", "vat_total", "total"})
_TRN_FIELDS: Final[frozenset[str]] = frozenset({"supplier_trn", "buyer_trn"})
_CENTS: Final[Decimal] = Decimal("0.01")
_TEXT_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[\s,.;:\-_/()\[\]'\"#]+")


def normalize_text(raw: str) -> str:
    """Comparison form of free text (names, addresses, numbers-as-identifiers): NFKC, ASCII
    digits, casefolded, with whitespace and common punctuation collapsed to single spaces."""
    text = convert_digits(unicodedata.normalize("NFKC", raw)).casefold()
    return _TEXT_PUNCT_RE.sub(" ", text).strip()


def normalize_field(name: str, raw: str) -> str:
    """Canonical string form of header field `name` (an `apx.extract.interfaces.FieldName`
    value): ISO date, 2-dp amount, digit-only TRN, compact IBAN, ISO currency code, or
    `normalize_text` for everything else. Raises `NormalizationError` when a typed field does
    not parse; callers decide whether that is a disagreement or an ungroundable value."""
    if name in _DATE_FIELDS:
        return normalize_date(raw).isoformat()
    if name in _AMOUNT_FIELDS:
        return format(normalize_amount(raw).quantize(_CENTS), "f")
    if name in _TRN_FIELDS:
        return normalize_trn(raw)
    if name == "iban":
        return normalize_iban(raw)
    if name == "currency":
        text = _WHITESPACE_RE.sub("", raw)
        if text in _CURRENCY_TOKENS:
            return "AED"
        if not text:
            raise NormalizationError(f"empty currency: {raw!r}")
        return text.upper()
    return normalize_text(raw)
