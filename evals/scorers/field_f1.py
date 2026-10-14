"""Shared field-level precision/recall/F1 core used by the FATURA and UAE scorers:
both compare a predicted (doc_id, field_name) -> value mapping against gold on
normalized exact match. Kept as one pure, typed function so both scorers report identically
comparable numbers for the scoreboard. For a UAE header field (`apx.common.compare.
FIELD_KIND_BY_NAME`), normalization delegates to the same comparator as the trust gate
(ADR-0009 §4.1), so the F1 promotion gate and the trust labels agree on what "correct" means;
other fields (FATURA, ExtractBench) keep the plain NFKC + whitespace-collapse rule unchanged."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from apx.common.compare import FIELD_KIND_BY_NAME, normalize_for_compare


@dataclass(frozen=True)
class FieldRecord:
    """One (document, field, value) triple, e.g. gold or predicted extraction."""

    doc_id: str
    field: str
    value: str


@dataclass(frozen=True)
class FieldF1Result:
    """Precision/recall/F1 for a set of predicted field records against gold, plus the raw
    counts they were computed from. Precision and recall are defined as 1.0 when their
    denominator is zero (nothing predicted / nothing to find), a standard vacuous-truth
    convention that keeps a scorer with no fields to grade from reporting a spurious 0."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def normalize_value(value: str, field: str | None = None) -> str:
    """Normalizes a field value for exact-match comparison. When `field` is a UAE header field
    (a `FieldName` value, e.g. "supplier_trn" or "iban"), delegates to the same per-kind
    comparator the trust gate uses (`apx.common.compare.normalize_for_compare`), so Extractor
    B's F1 promotion gate and the trust labels can never disagree on what counts as a match
    (ADR-0009 §4.1, F4). `normalize_for_compare`'s None (blank input) becomes "", matching this
    function's plain-string contract. Otherwise (FATURA, ExtractBench, or no field given): NFKC
    unicode normalization (folds Arabic-Indic digit variants and presentation forms to their
    canonical form) then collapses surrounding/internal whitespace, unchanged from before."""
    if field is not None and field in FIELD_KIND_BY_NAME:
        canonical = normalize_for_compare(FIELD_KIND_BY_NAME[field], value)
        return canonical if canonical is not None else ""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _record_key(record: FieldRecord) -> tuple[str, str, str]:
    return (record.doc_id, record.field, normalize_value(record.value, record.field))


def field_f1(gold: Iterable[FieldRecord], predicted: Iterable[FieldRecord]) -> FieldF1Result:
    """Micro-averaged field F1 over normalized exact match of (doc_id, field, value)."""
    gold_set = {_record_key(r) for r in gold}
    predicted_set = {_record_key(r) for r in predicted}

    true_positives = len(gold_set & predicted_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)

    precision = (
        1.0
        if true_positives + false_positives == 0
        else true_positives / (true_positives + false_positives)
    )
    recall = (
        1.0
        if true_positives + false_negatives == 0
        else true_positives / (true_positives + false_negatives)
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return FieldF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def records_from_field_maps(
    doc_to_fields: Mapping[str, Mapping[str, str]],
) -> list[FieldRecord]:
    """Flattens a `{doc_id: {field: value}}` mapping (the natural shape of a predictor's or a
    gold set's per-document field output) into `FieldRecord`s for `field_f1`."""
    return [
        FieldRecord(doc_id=doc_id, field=field, value=value)
        for doc_id, fields in doc_to_fields.items()
        for field, value in fields.items()
    ]
