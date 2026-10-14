"""The single canonical field-value comparator (ADR-0009 §4.1): `FieldKind`, `normalize_for_compare`
and the plain-string `FIELD_KIND_BY_NAME` lookup. `apx.trust.features` (trust labels and the LTT
gate) and `evals.scorers.field_f1` (the F1 scoreboard, including Extractor B's promotion gate) both
delegate to this module, so "correct" means the same thing on both sides of ADR-0009 §4.1's
equivalence requirement; it has no dependency on `apx.trust`, so `evals` never imports trust/cal
code."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum

from apx.common.normalize import MAX_AMOUNT_INTEGER_DIGITS


class FieldKind(StrEnum):
    """Comparison family of a header field: selects its normaliser and format rule (ADR-0009
    §4.1)."""

    TRN = "trn"
    IBAN = "iban"
    AMOUNT = "amount"
    DATE = "date"
    CURRENCY = "currency"
    IDENTIFIER = "identifier"
    TEXT = "text"


FIELD_KIND_BY_NAME: Mapping[str, FieldKind] = {
    "invoice_number": FieldKind.IDENTIFIER,
    "issue_date": FieldKind.DATE,
    "due_date": FieldKind.DATE,
    "supplier_name": FieldKind.TEXT,
    "supplier_trn": FieldKind.TRN,
    "supplier_address": FieldKind.TEXT,
    "buyer_name": FieldKind.TEXT,
    "buyer_trn": FieldKind.TRN,
    "currency": FieldKind.CURRENCY,
    "po_number": FieldKind.IDENTIFIER,
    "subtotal": FieldKind.AMOUNT,
    "vat_total": FieldKind.AMOUNT,
    "total": FieldKind.AMOUNT,
    "iban": FieldKind.IBAN,
}
"""Plain-string mirror of `apx.trust.interfaces.FIELD_KINDS`, keyed by `FieldName.value` rather
than the enum itself so this module stays free of any `apx.extract`/`apx.trust` dependency.
`apx.trust.interfaces.FIELD_KINDS` is built from this dict (a test pins the two in sync); the eval
scorers (FATURA, ExtractBench) use field names outside this set, which is why the lookup is a
plain, best-effort dict rather than a total mapping over some enum."""

_DIGIT_MAP: dict[int, str] = {}
for _i, _digit in enumerate("٠١٢٣٤٥٦٧٨٩"):  # U+0660-0669 Arabic-Indic
    _DIGIT_MAP[ord(_digit)] = str(_i)
for _i, _digit in enumerate("۰۱۲۳۴۵۶۷۸۹"):  # U+06F0-06F9 Eastern Arabic-Indic
    _DIGIT_MAP[ord(_digit)] = str(_i)

_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}  # noqa: RUF001
_TATWEEL = "ـ"
_CURRENCY_ALIASES: Mapping[str, str] = {
    "AED": "AED",
    "DHS": "AED",
    "DH": "AED",
    "د.إ": "AED",
    "درهم": "AED",
}


def _is_arabic_diacritic(cp: int) -> bool:
    return 0x064B <= cp <= 0x065F or cp == 0x0670


def _common_normalize(raw: str) -> str:
    """NFKC, Arabic/Eastern-Arabic digits -> ASCII, NBSP -> space, strip (ADR-0009 §4.1)."""
    text = unicodedata.normalize("NFKC", raw).translate(_DIGIT_MAP)
    text = text.replace(chr(0xA0), " ")  # NBSP -> space
    return text.strip()


def _text_canonical(s: str) -> str:
    """TEXT canonical form: casefold, drop tatweel/diacritics, fold alef/ya/ta-marbuta variants,
    map every P*/S* character to a space, collapse whitespace, then re-apply NFKC (ADR-0009 §4.1).

    The final NFKC pass is required for idempotency, not cosmetic: casefold() can itself introduce
    combining marks that NFKC would have composed with a *preceding* character (e.g. Greek
    dialytika+tonos casefolding), and deleting tatweel/diacritics or collapsing a NBSP/space can
    remove a character that was *blocking* a composition (e.g. "H" + TATWEEL + COMBINING DIAERESIS
    -> deleting TATWEEL leaves an adjacent base+mark pair NFKC would combine on the next pass). Both
    cases change the string's length on a second pass unless the composition happens here, in this
    same call, before the value is ever handed back out."""
    folded = s.casefold()
    out: list[str] = []
    for ch in folded:
        cp = ord(ch)
        if ch == _TATWEEL or _is_arabic_diacritic(cp):
            continue
        ch = _ALEF_VARIANTS.get(ch, ch)
        category = unicodedata.category(ch)
        out.append(" " if category[0] in "PS" else ch)
    collapsed = re.sub(r"\s+", " ", "".join(out)).strip()
    return unicodedata.normalize("NFKC", collapsed)


def _fold_upper(s: str) -> str:
    """Upper-case then re-apply NFKC (ADR-0009 §4.1 idempotency fix, same pattern as CURRENCY's
    alpha filter below): str.upper() can introduce combining marks that a later NFKC pass would
    compose with a preceding character (e.g. Greek iota-with-dialytika-and-varia), so the result
    must already be NFKC-normal before it is returned, not left for the next pass to normalize."""
    return unicodedata.normalize("NFKC", s.upper())


def _parse_amount(body: str) -> Decimal | None:
    """Parse a header amount into a Decimal, handling thousands/decimal separator ambiguity with
    the common heuristic: with both separators present the last one is decimal; with one appearing
    once and exactly 3 trailing digits it is thousands grouping; otherwise it is decimal."""
    text = body.strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")") and len(text) > 2:
        negative = True
        text = text[1:-1].strip()
    if text and text[0] in "+-":
        negative = negative != (text[0] == "-")
        text = text[1:].strip()
    text = text.replace(" ", "")
    if not text or not re.fullmatch(r"[0-9.,]+", text):
        return None

    dot_count, comma_count = text.count("."), text.count(",")
    integer_part: str
    frac_part: str
    if dot_count and comma_count:
        if text.rfind(".") > text.rfind(","):
            integer_part, _, frac_part = text.replace(",", "").rpartition(".")
        else:
            integer_part, _, frac_part = text.replace(".", "").rpartition(",")
    elif dot_count == 1 or comma_count == 1:
        sep = "." if dot_count == 1 else ","
        integer_part, frac_part = text.split(sep)
        if len(frac_part) == 3 and len(integer_part) <= 3:
            integer_part, frac_part = integer_part + frac_part, ""
    elif dot_count > 1 or comma_count > 1:
        integer_part, frac_part = text.replace(".", "").replace(",", ""), ""
    else:
        integer_part, frac_part = text, ""

    integer_part = integer_part or "0"
    if not integer_part.isdigit() or (frac_part and not frac_part.isdigit()):
        return None
    try:
        value = Decimal(f"{integer_part}.{frac_part or '0'}")
    except InvalidOperation:
        return None
    if value.adjusted() >= MAX_AMOUNT_INTEGER_DIGITS:  # not an amount: cents would overflow
        return None
    if negative:
        value = -value
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%Y%m%d",
)
"""Tried in order; day-first formats are prioritised (UAE/EN invoices), ADR-0009 §4.1."""


def _parse_date(body: str) -> date | None:
    text = body.strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _currency_alias(common: str) -> str | None:
    return _CURRENCY_ALIASES.get(common.strip().upper()) or _CURRENCY_ALIASES.get(common.strip())


def normalize_for_compare(kind: FieldKind, raw: str | None) -> str | None:
    """Canonical comparison form (ADR-0009 §4.1). None or blank -> None. Never raises: an
    unparseable structured value becomes "~" + its TEXT canonical form, which cannot equal a
    parsed value. The UAE field-F1 scorer and `is_correct` use this same comparator."""
    if raw is None:
        return None
    common = _common_normalize(raw)
    if not common:
        return None

    def or_tilde_fallback(candidate: str) -> str:
        # A non-blank `raw` must never canonicalise to "": that would make it indistinguishable,
        # on a second pass through this same function, from a genuinely blank/None value (the
        # `if not common: return None` check above), breaking idempotency (proof). The
        # leading "~" guarantees a non-empty result even when `_text_canonical(common)` is also
        # empty (an all-punctuation value), and it can never equal a validly-parsed canonical form.
        return candidate if candidate else "~" + _text_canonical(common)

    if kind is FieldKind.TRN:
        # NFKC after stripping: removing whitespace/hyphens can un-block a composition (e.g.
        # "H" + TATWEEL-like separator + a combining mark), so re-normalize before returning
        # (ADR-0009 §4.1 idempotency fix).
        return or_tilde_fallback(unicodedata.normalize("NFKC", re.sub(r"[\s-]", "", common)))
    if kind is FieldKind.IBAN:
        return or_tilde_fallback(_fold_upper(re.sub(r"\s", "", common)))
    if kind is FieldKind.AMOUNT:
        parsed = _parse_amount(common)
        return format(parsed, "f") if parsed is not None else "~" + _text_canonical(common)
    if kind is FieldKind.DATE:
        parsed_date = _parse_date(common)
        if parsed_date is not None:
            return parsed_date.isoformat()
        return "~" + _text_canonical(common)
    if kind is FieldKind.CURRENCY:
        # Upper-case and re-NFKC (via _fold_upper) BEFORE filtering: str.upper() can itself
        # introduce combining marks (e.g. Greek upsilon-with-psili -> "Y" + COMBINING COMMA ABOVE)
        # that .isalpha() rejects; folding case first, composing, and filtering the result keeps
        # normalize_for_compare idempotent on re-feed (ADR-0009 §4.1).
        alpha_only = "".join(ch for ch in _fold_upper(common) if ch.isalpha())
        # Check the alias table against the raw common form first (the Arabic aliases, e.g.
        # "د.إ", need their punctuation intact) and, failing that, against the alpha-filtered
        # form: noise stripped by the filter (e.g. a leading OCR digit in "0DH") can reveal an
        # alias that the unfiltered value did not match. Checking both *here* -- rather than
        # letting a second call through or_tilde_fallback discover it -- is what keeps this
        # idempotent: "0DH" -> "DH" -> "AED" in two calls would otherwise never stabilise after
        # a single one.
        alias = _currency_alias(common) or _currency_alias(alpha_only)
        if alias is not None:
            return alias
        return or_tilde_fallback(alpha_only)
    if kind is FieldKind.IDENTIFIER:
        return or_tilde_fallback(_fold_upper(re.sub(r"\s", "", common)))
    if kind is FieldKind.TEXT:
        return or_tilde_fallback(_text_canonical(common))
    raise AssertionError(f"unhandled FieldKind {kind!r}")  # pragma: no cover - exhaustive enum
