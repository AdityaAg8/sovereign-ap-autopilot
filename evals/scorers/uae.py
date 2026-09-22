"""UAE factory field F1 scorer, reported separately for English and Arabic invoices (PRD §5
scoreboard row "Field F1, EN and AR separately", §10 M1.3). Built on the same normalized-exact-
match field F1 core as the FATURA scorer; the only UAE-specific piece is partitioning documents
by language before scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from evals.scorers.field_f1 import FieldF1Result, field_f1, records_from_field_maps

Language = Literal["en", "ar"]


@dataclass(frozen=True)
class UaeDocument:
    """A gold UAE factory document: its id, source language, and gold field values."""

    doc_id: str
    lang: Language
    fields: Mapping[str, str]


@dataclass(frozen=True)
class UaeFieldF1Result:
    """Field F1 for a UAE test set, split by language plus the pooled result over both."""

    en: FieldF1Result
    ar: FieldF1Result
    overall: FieldF1Result


def score_uae(
    gold: list[UaeDocument],
    predicted: Mapping[str, Mapping[str, str]],
) -> UaeFieldF1Result:
    """Scores UAE field extraction split by language. `gold` carries each document's language
    and gold fields; `predicted` maps doc_id to `{field_name: value}`, independent of
    language (a predictor sees the same document regardless of which language it is in)."""
    gold_by_lang: dict[Language, dict[str, Mapping[str, str]]] = {"en": {}, "ar": {}}
    predicted_by_lang: dict[Language, dict[str, Mapping[str, str]]] = {"en": {}, "ar": {}}
    for doc in gold:
        gold_by_lang[doc.lang][doc.doc_id] = doc.fields
        if doc.doc_id in predicted:
            predicted_by_lang[doc.lang][doc.doc_id] = predicted[doc.doc_id]

    en_result = field_f1(
        records_from_field_maps(gold_by_lang["en"]),
        records_from_field_maps(predicted_by_lang["en"]),
    )
    ar_result = field_f1(
        records_from_field_maps(gold_by_lang["ar"]),
        records_from_field_maps(predicted_by_lang["ar"]),
    )
    overall_gold = {doc.doc_id: doc.fields for doc in gold}
    overall_result = field_f1(
        records_from_field_maps(overall_gold), records_from_field_maps(predicted)
    )
    return UaeFieldF1Result(en=en_result, ar=ar_result, overall=overall_result)
