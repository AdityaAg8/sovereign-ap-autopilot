"""Unit tests for evals/scorers/field_f1.py, the shared normalized-exact-match field F1 core,
on hand-computed cases, plus the ADR-0009 §4.1 (F4) equivalence property: for a
UAE header field, this module's comparator and the M7 trust gate's must agree on every match."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from apx.common.compare import FIELD_KIND_BY_NAME
from apx.extract.interfaces import FieldName
from apx.trust.features import is_correct
from evals.scorers.field_f1 import FieldRecord, field_f1, normalize_value


def test_normalize_value_collapses_whitespace() -> None:
    assert normalize_value("  Acme   Corp \n") == "Acme Corp"


def test_normalize_value_folds_unicode_compatibility_forms() -> None:
    # U+FF11 FULLWIDTH DIGIT ONE NFKC-normalizes to ASCII "1".
    assert normalize_value("\uff11") == "1"


def test_field_f1_perfect_match() -> None:
    gold = [FieldRecord("doc1", "total", "100.00"), FieldRecord("doc1", "vat", "5.00")]
    predicted = [FieldRecord("doc1", "total", "100.00"), FieldRecord("doc1", "vat", "5.00")]

    result = field_f1(gold, predicted)

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_field_f1_hand_computed_mixed_case() -> None:
    # 2 docs x 2 fields = 4 gold fields. Predictions: 2 correct, 1 wrong value (false positive
    # + false negative), 1 field not predicted at all (false negative only).
    gold = [
        FieldRecord("doc1", "total", "100.00"),
        FieldRecord("doc1", "vat", "5.00"),
        FieldRecord("doc2", "total", "200.00"),
        FieldRecord("doc2", "vat", "10.00"),
    ]
    predicted = [
        FieldRecord("doc1", "total", "100.00"),  # correct
        FieldRecord("doc1", "vat", "999.00"),  # wrong value
        FieldRecord("doc2", "total", "200.00"),  # correct
        # doc2/vat not predicted at all
    ]

    result = field_f1(gold, predicted)

    # TP = {(doc1,total,100.00), (doc2,total,200.00)} = 2
    # FP = {(doc1,vat,999.00)} = 1
    # FN = {(doc1,vat,5.00), (doc2,vat,10.00)} = 2
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 2
    assert result.precision == 2 / 3
    assert result.recall == 2 / 4
    expected_f1 = 2 * (2 / 3) * (1 / 2) / ((2 / 3) + (1 / 2))
    assert abs(result.f1 - expected_f1) < 1e-9


def test_field_f1_empty_gold_and_predicted_is_vacuously_perfect() -> None:
    result = field_f1([], [])
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_field_f1_all_false_positives_when_gold_is_empty() -> None:
    predicted = [FieldRecord("doc1", "total", "100.00")]
    result = field_f1([], predicted)
    assert result.precision == 0.0
    assert result.recall == 1.0  # nothing to find, so recall is vacuously perfect
    assert result.f1 == 0.0


def test_field_f1_normalizes_before_matching() -> None:
    gold = [FieldRecord("doc1", "name", "  Acme   Corp  ")]
    predicted = [FieldRecord("doc1", "name", "Acme Corp")]
    result = field_f1(gold, predicted)
    assert result.true_positives == 1
    assert result.f1 == 1.0


# --------------------------------------------------------------------------------------------
# ADR-0009 §4.1 (F4): UAE header fields share one comparator with the M7 trust gate
# --------------------------------------------------------------------------------------------


def test_field_kind_by_name_covers_exactly_the_uae_header_fields() -> None:
    assert set(FIELD_KIND_BY_NAME) == {name.value for name in FieldName}


def test_normalize_value_delegates_to_the_trust_comparator_for_a_header_field() -> None:
    # IBAN: spacing differs, must still match (ADR-0009 §4.1's own worked example).
    assert normalize_value("ae07 0331 2345 6789 0123 456", "iban") == normalize_value(
        "AE070331234567890123456", "iban"
    )
    # TRN: dashes vs none.
    assert normalize_value("100-234-567-890-003", "supplier_trn") == normalize_value(
        "100234567890003", "supplier_trn"
    )
    # A field with no FieldKind (e.g. a FATURA/ExtractBench field, or a line-item column) keeps
    # the plain NFKC + whitespace-collapse rule, unaffected by the delegation.
    assert normalize_value("  Acme   Corp  ", "vendor") == "Acme Corp"
    assert normalize_value("  Acme   Corp  ") == "Acme Corp"  # field omitted entirely


@settings(
    max_examples=1000,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(st.sampled_from(list(FieldName)), st.text(max_size=30), st.text(max_size=30))
def test_normalize_value_agrees_with_the_trust_comparator_for_every_header_field(
    field: FieldName, a: str, b: str
) -> None:
    """For every UAE header field and arbitrary raw values, evals' `normalize_value` and the
    trust gate's `is_correct` must reach the same match/no-match verdict. They now share the
    same underlying `apx.common.compare.normalize_for_compare` call (ADR-0009 §4.1, F4), so this
    proves the delegation is wired correctly end to end rather than merely present in the diff."""
    matches_in_eval = normalize_value(a, field.value) == normalize_value(b, field.value)
    matches_in_trust = is_correct(field, a, b)
    assert matches_in_eval == matches_in_trust
