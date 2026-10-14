"""Grounds extracted header values onto parse words (I1): finds the run of consecutive words
(reading order, one page) whose normalised text equals the normalised value, prefers the run
nearest a field label (EN/AR), and takes boxes from those words only. Model output never
supplies a box; an unmatched value keeps empty `word_ids` and `boxes`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apx.common.types import FieldValue
from apx.extract.interfaces import FieldName, RawField
from apx.extract.normalize import NormalizationError, normalize_field, normalize_text
from apx.parse.interfaces import ParsedDocument, Word

_MAX_WORDS: dict[FieldName, int] = {
    FieldName.ISSUE_DATE: 5,
    FieldName.DUE_DATE: 5,
    FieldName.SUBTOTAL: 3,
    FieldName.VAT_TOTAL: 3,
    FieldName.TOTAL: 3,
    FieldName.SUPPLIER_TRN: 5,
    FieldName.BUYER_TRN: 5,
    FieldName.IBAN: 9,
    FieldName.CURRENCY: 2,
}
_MAX_TEXT_WORDS = 40
_EDGE_PUNCTUATION = " .,;:()[]"
"""Sentence punctuation a typed value can carry on the page, e.g. `IBAN AE07...456.`"""
_LABEL_WINDOW = 8
"""How many preceding words (same page) are searched for a field label."""

LABEL_HINTS: dict[FieldName, tuple[str, ...]] = {
    FieldName.INVOICE_NUMBER: ("invoice no", "invoice number", "invoice #", "رقم الفاتورة"),
    FieldName.ISSUE_DATE: ("invoice date", "issue date", "date", "تاريخ الفاتورة", "تاريخ الإصدار"),
    FieldName.DUE_DATE: ("due date", "payment due", "تاريخ الاستحقاق"),
    FieldName.SUPPLIER_NAME: ("supplier", "seller", "from", "المورد", "البائع"),
    FieldName.SUPPLIER_TRN: ("trn", "tax registration", "الرقم الضريبي", "رقم التسجيل الضريبي"),
    FieldName.SUPPLIER_ADDRESS: ("address", "العنوان"),
    FieldName.BUYER_NAME: ("bill to", "buyer", "customer", "المشتري", "العميل"),
    FieldName.BUYER_TRN: ("trn", "tax registration", "الرقم الضريبي", "رقم التسجيل الضريبي"),
    FieldName.CURRENCY: ("currency", "العملة"),
    FieldName.PO_NUMBER: ("po", "purchase order", "أمر الشراء"),
    FieldName.SUBTOTAL: ("subtotal", "sub total", "total excluding vat", "المجموع الفرعي"),
    FieldName.VAT_TOTAL: ("vat", "tax amount", "ضريبة القيمة المضافة"),
    FieldName.TOTAL: ("total", "amount due", "grand total", "الإجمالي", "المبلغ المستحق"),
    FieldName.IBAN: ("iban", "آيبان", "رقم الحساب"),
}


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    label_score: int


def _comparable(name: FieldName, text: str) -> str | None:
    try:
        return normalize_field(name.value, text)
    except NormalizationError:
        return None


def _ordered_words(document: ParsedDocument) -> list[Word]:
    """All line words in reading order, ignoring any word not on a line."""
    words = {word.id: word for page in document.pages for word in page.words}
    line_words = {line.id: line.word_ids for page in document.pages for line in page.lines}
    return [words[word_id] for line_id in document.reading_order for word_id in line_words[line_id]]


class WordGrounder:
    """`Grounder` by exact normalised match over word runs. Text fields also accept a run with
    the same words in another order, which absorbs RTL word-order differences."""

    def ground(
        self, document: ParsedDocument, fields: Sequence[RawField]
    ) -> dict[FieldName, FieldValue]:
        words = _ordered_words(document)
        folded = [normalize_text(word.text) for word in words]
        grounded: dict[FieldName, FieldValue] = {}
        for field in fields:
            grounded[field.name] = self._ground_one(field, words, folded)
        return grounded

    def _ground_one(self, field: RawField, words: list[Word], folded: list[str]) -> FieldValue:
        raw = field.value
        if raw is None or not raw.strip():
            return FieldValue(name=field.name.value, raw=raw, normalized=None)
        target = _comparable(field.name, raw)
        match = self._best_match(field.name, target, raw, words, folded)
        if match is None:
            return FieldValue(name=field.name.value, raw=raw, normalized=target)
        run = words[match.start : match.end]
        return FieldValue(
            name=field.name.value,
            raw=raw,
            normalized=target,
            word_ids=tuple(word.id for word in run),
            boxes=tuple(word.bbox for word in run),
        )

    def _best_match(
        self,
        name: FieldName,
        target: str | None,
        raw: str,
        words: list[Word],
        folded: list[str],
    ) -> _Match | None:
        typed = name in _MAX_WORDS
        if typed and target is None:
            return None
        text_target = normalize_text(raw)
        max_words = _MAX_WORDS.get(name, _MAX_TEXT_WORDS)
        target_tokens = sorted(text_target.split())
        matches: list[_Match] = []
        for start in range(len(words)):
            page = words[start].bbox.page
            for end in range(start + 1, min(start + max_words, len(words)) + 1):
                if words[end - 1].bbox.page != page:
                    break
                run_text = " ".join(folded[start:end])
                if typed:
                    run_raw = " ".join(w.text for w in words[start:end]).strip(_EDGE_PUNCTUATION)
                    hit = _comparable(name, run_raw) == target
                else:
                    if len(run_text) > len(text_target) + 1:
                        break
                    hit = run_text == text_target or (
                        sorted(run_text.split()) == target_tokens and len(target_tokens) > 1
                    )
                if hit:
                    matches.append(
                        _Match(start, end, self._label_score(name, start, words, folded))
                    )
                    break
        if not matches:
            return None
        return max(matches, key=lambda m: (m.label_score, -(m.end - m.start), -m.start))

    @staticmethod
    def _label_score(name: FieldName, start: int, words: list[Word], folded: list[str]) -> int:
        page = words[start].bbox.page
        window_start = start
        while window_start > 0 and start - window_start < _LABEL_WINDOW:
            if words[window_start - 1].bbox.page != page:
                break
            window_start -= 1
        context = " " + " ".join(folded[window_start:start]) + " "
        return sum(1 for hint in LABEL_HINTS[name] if f" {normalize_text(hint)} " in context)
