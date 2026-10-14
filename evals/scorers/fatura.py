"""FATURA field F1 scorer: compares predicted per-document field extractions
against the FATURA (Zenodo record 8261508) gold annotations using the shared normalized-exact-
match field F1 core."""

from __future__ import annotations

from collections.abc import Mapping

from evals.scorers.field_f1 import FieldF1Result, field_f1, records_from_field_maps


def score_fatura(
    gold: Mapping[str, Mapping[str, str]],
    predicted: Mapping[str, Mapping[str, str]],
) -> FieldF1Result:
    """Scores FATURA field extraction. Both `gold` and `predicted` map document id to a
    `{field_name: value}` dict; a document or field missing from `predicted` counts as a false
    negative, and one present but not in `gold` counts as a false positive."""
    return field_f1(records_from_field_maps(gold), records_from_field_maps(predicted))
