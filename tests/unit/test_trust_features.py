"""Unit and property tests for the ADR-0009 §4 trust features: the comparator, agreement, format
validity, OCR support, the arbiter vote, hard-reject rules and the 28-column encoder (M7.4 proof:
comparator idempotent, agreement symmetric and within [0, 1], encode length 28, lexicon covers
EN and AR for every field)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from apx.common.errors import TrustGateError
from apx.common.types import BBox, FieldValue, Language
from apx.extract.interfaces import (
    ExtractionResult,
    ExtractorId,
    ExtractorOutput,
    FieldLogprobs,
    FieldName,
    RawField,
)
from apx.parse.interfaces import Line, Page, ParsedDocument, Word
from apx.trust.features import (
    ExtractionFeatureBuilder,
    _is_valid_iban,
    agreement,
    arbiter_vote,
    encode_features,
    format_valid,
    hard_reject,
    is_correct,
    normalize_for_compare,
    ocr_support,
)
from apx.trust.interfaces import (
    FEATURE_NAMES,
    FIELD_KINDS,
    ArbiterVote,
    FieldFeatures,
    FieldKind,
)
from apx.trust.labels import LABELS

DOC_SHA = "a" * 64
VALID_UAE_IBAN = "AE070331234567890123456"
INVALID_UAE_IBAN = "AE070331234567890123457"

ALL_KINDS = list(FieldKind)


# --------------------------------------------------------------------------------------------
# normalize_for_compare
# --------------------------------------------------------------------------------------------


def test_normalize_for_compare_none_and_blank_is_none() -> None:
    for kind in ALL_KINDS:
        assert normalize_for_compare(kind, None) is None
        assert normalize_for_compare(kind, "   ") is None
        assert normalize_for_compare(kind, "") is None


def test_normalize_trn_strips_whitespace_and_dashes() -> None:
    assert normalize_for_compare(FieldKind.TRN, "100 234 567 890 003") == "100234567890003"
    assert normalize_for_compare(FieldKind.TRN, "100-234-567-890-003") == "100234567890003"


def test_normalize_trn_maps_arabic_indic_digits() -> None:
    assert normalize_for_compare(FieldKind.TRN, "١٠٠٢٣٤٥٦٧٨٩٠٠٠٣") == "100234567890003"


def test_normalize_iban_uppercases_and_strips_spaces() -> None:
    spaced = "ae07 0331 2345 6789 0123 456"
    assert normalize_for_compare(FieldKind.IBAN, spaced) == VALID_UAE_IBAN


def test_normalize_amount_thousands_and_decimal() -> None:
    assert normalize_for_compare(FieldKind.AMOUNT, "1,234.50") == "1234.50"
    assert normalize_for_compare(FieldKind.AMOUNT, "1.234,50") == "1234.50"
    assert normalize_for_compare(FieldKind.AMOUNT, "1,234") == "1234.00"
    assert normalize_for_compare(FieldKind.AMOUNT, "(1234.5)") == "-1234.50"
    assert normalize_for_compare(FieldKind.AMOUNT, "-1234.505") == "-1234.51"  # half-up


def test_normalize_amount_unparseable_gets_tilde_text_fallback() -> None:
    result = normalize_for_compare(FieldKind.AMOUNT, "N/A")
    assert result is not None
    assert result.startswith("~")


def test_normalize_date_common_formats_agree() -> None:
    assert normalize_for_compare(FieldKind.DATE, "2025-01-31") == "2025-01-31"
    assert normalize_for_compare(FieldKind.DATE, "31/01/2025") == "2025-01-31"
    assert normalize_for_compare(FieldKind.DATE, "31-01-2025") == "2025-01-31"
    assert normalize_for_compare(FieldKind.DATE, "31 Jan 2025") == "2025-01-31"


def test_normalize_date_unparseable_gets_tilde_text_fallback() -> None:
    result = normalize_for_compare(FieldKind.DATE, "not a date")
    assert result is not None
    assert result.startswith("~")


def test_normalize_currency_aliases_to_aed() -> None:
    for alias in ("AED", "DHS", "DH", "د.إ", "درهم", "aed"):
        assert normalize_for_compare(FieldKind.CURRENCY, alias) == "AED"
    assert normalize_for_compare(FieldKind.CURRENCY, "USD") == "USD"


def test_normalize_identifier_uppercases_and_strips_whitespace() -> None:
    assert normalize_for_compare(FieldKind.IDENTIFIER, " inv 001 ") == "INV001"


def test_normalize_text_folds_arabic_variants_and_punctuation() -> None:
    assert normalize_for_compare(FieldKind.TEXT, "أحمد!") == normalize_for_compare(
        FieldKind.TEXT, "احمد"
    )
    assert normalize_for_compare(FieldKind.TEXT, "Acme, Inc.") == normalize_for_compare(
        FieldKind.TEXT, "acme inc"
    )


@given(st.sampled_from(ALL_KINDS), st.text(min_size=0, max_size=40))
def test_normalize_for_compare_never_raises(kind: FieldKind, raw: str) -> None:
    normalize_for_compare(kind, raw)


# normalize_for_compare(kind, normalize_for_compare(kind, x)) must equal normalize_for_compare(kind,
# x) for every FieldKind and *arbitrary* Unicode input, not just letters/digits: a case-folding or
# upper-casing step can itself introduce combining marks (or delete a character that was blocking a
# composition) that a later NFKC pass would combine differently, changing the string on a second
# pass through the comparator (ADR-0009 §4.1, M7.4 proof). The strategy below is biased toward the
# character classes where that actually happens -- combining marks, Greek Extended (dialytika/tonos
# precomposed letters), Arabic diacritics/tatweel, NBSP/hyphen -- since uniform sampling over the
# full codepoint space essentially never reaches them. `derandomize=True` and `database=None` make
# the run deterministic in CI: a flaky pass/fail on this property would be worse than a fixed set of
# hard examples, so no persisted example database and no per-run random seed.
_idempotency_targeted_chars = st.sampled_from(
    list("H¨ـ -  ًٌٍَُِّْٰ")  # noqa: RUF001 -- NBSP is deliberate test data
    + [chr(c) for c in range(0x0300, 0x0370)]  # combining diacritical marks
    + [chr(c) for c in range(0x1F00, 0x2000)]  # Greek Extended
    + [chr(c) for c in range(0x0620, 0x0660)]  # Arabic letters
    + list("aAeEiIoOuUsSbBßİı0123456789")  # noqa: RUF001 -- dotless i is deliberate test data
)
_idempotency_any_char = st.characters(
    min_codepoint=0x20, max_codepoint=0x2FFFF, blacklist_categories=("Cs", "Co")
)
_idempotency_strategy = st.text(
    alphabet=st.one_of(_idempotency_any_char, _idempotency_targeted_chars), min_size=0, max_size=20
)


@settings(
    max_examples=2000,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@example(FieldKind.IBAN, "ῒ")  # F1: A/B upper() introduces marks NFKC recomposes differently
@example(FieldKind.IDENTIFIER, "ῒ")  # same _fold_upper path as IBAN
@example(FieldKind.TRN, "H¨")  # NFKC(raw) inserts a blocking space that stripping removes
@example(FieldKind.CURRENCY, "0DH")  # alias lookup on the unfiltered value misses "DH"
@example(FieldKind.CURRENCY, "ῒ")
@example(FieldKind.TEXT, "Hـ̈")  # deleting TATWEEL un-blocks a composition
@example(FieldKind.TEXT, "ΐ")  # casefold() of iota-dialytika-tonos introduces marks
@given(st.sampled_from(ALL_KINDS), _idempotency_strategy)
def test_normalize_for_compare_is_idempotent(kind: FieldKind, raw: str) -> None:
    once = normalize_for_compare(kind, raw)
    if once is None:
        return
    twice = normalize_for_compare(kind, once)
    assert once == twice, (kind, raw, once, twice)


def test_normalize_amount_and_date_are_idempotent_on_their_own_canonical_form() -> None:
    for raw in ("1,234.50", "-1234.505", "31/01/2025"):
        kind = FieldKind.AMOUNT if "/" not in raw else FieldKind.DATE
        once = normalize_for_compare(kind, raw)
        assert once is not None
        twice = normalize_for_compare(kind, once)
        assert once == twice


# --------------------------------------------------------------------------------------------
# is_correct
# --------------------------------------------------------------------------------------------


def test_is_correct_null_equals_null() -> None:
    assert is_correct(FieldName.DUE_DATE, None, None) is True


def test_is_correct_matches_after_normalisation() -> None:
    assert is_correct(FieldName.SUPPLIER_TRN, "100-234-567-890-003", "100234567890003") is True
    assert is_correct(FieldName.SUPPLIER_TRN, "100234567890003", "999999999999999") is False


# --------------------------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------------------------


def test_agreement_both_null_is_one() -> None:
    assert agreement(FieldKind.TEXT, None, None) == 1.0


def test_agreement_one_null_is_zero() -> None:
    assert agreement(FieldKind.TEXT, "Acme", None) == 0.0
    assert agreement(FieldKind.TEXT, None, "Acme") == 0.0


def test_agreement_non_text_is_binary() -> None:
    assert agreement(FieldKind.AMOUNT, "100.00", "100") == 1.0
    assert agreement(FieldKind.AMOUNT, "100.00", "200") == 0.0


def test_agreement_text_partial_similarity() -> None:
    value = agreement(FieldKind.TEXT, "Acme Trading LLC", "Acme Trading L.L.C")
    assert 0.0 < value < 1.0


@given(
    st.sampled_from(ALL_KINDS),
    st.one_of(st.none(), st.text(max_size=20)),
    st.one_of(st.none(), st.text(max_size=20)),
)
def test_agreement_is_symmetric_and_bounded(kind: FieldKind, a: str | None, b: str | None) -> None:
    forward = agreement(kind, a, b)
    backward = agreement(kind, b, a)
    assert forward == backward
    assert 0.0 <= forward <= 1.0


# --------------------------------------------------------------------------------------------
# format_valid
# --------------------------------------------------------------------------------------------


def test_format_valid_trn() -> None:
    assert format_valid(FieldKind.TRN, "100234567890003") is True
    assert format_valid(FieldKind.TRN, "12345") is False
    assert format_valid(FieldKind.TRN, None) is False


def test_format_valid_iban() -> None:
    assert format_valid(FieldKind.IBAN, VALID_UAE_IBAN) is True
    assert format_valid(FieldKind.IBAN, INVALID_UAE_IBAN) is False


def test_format_valid_amount() -> None:
    assert format_valid(FieldKind.AMOUNT, "1,234.50") is True
    assert format_valid(FieldKind.AMOUNT, "not a number") is False


def test_format_valid_date_range() -> None:
    assert format_valid(FieldKind.DATE, "2025-01-31") is True
    assert format_valid(FieldKind.DATE, "1999-01-01") is False  # before 2000
    assert format_valid(FieldKind.DATE, "2101-01-01") is False  # after 2100


def test_format_valid_currency() -> None:
    assert format_valid(FieldKind.CURRENCY, "AED") is True
    assert format_valid(FieldKind.CURRENCY, "usd") is True
    assert format_valid(FieldKind.CURRENCY, "1") is False


def test_format_valid_identifier() -> None:
    assert format_valid(FieldKind.IDENTIFIER, "INV-001") is True
    assert format_valid(FieldKind.IDENTIFIER, "---") is False


def test_format_valid_text() -> None:
    assert format_valid(FieldKind.TEXT, "Acme Trading LLC") is True
    assert format_valid(FieldKind.TEXT, "1") is False
    assert format_valid(FieldKind.TEXT, "12345") is False


def test_is_valid_iban_matches_the_mod97_checksum() -> None:
    assert _is_valid_iban(VALID_UAE_IBAN) is True
    assert _is_valid_iban(INVALID_UAE_IBAN) is False


# --------------------------------------------------------------------------------------------
# arbiter_vote / hard_reject
# --------------------------------------------------------------------------------------------


def test_arbiter_vote_not_run_when_not_disputed() -> None:
    result = arbiter_vote(
        FieldKind.TEXT, "Acme", "Acme", "Acme", disputed=False, arbiter_valid=True
    )
    assert result is ArbiterVote.NOT_RUN


def test_arbiter_vote_invalid_when_arbiter_output_invalid() -> None:
    result = arbiter_vote(FieldKind.TEXT, "Acme", "Acme2", None, disputed=True, arbiter_valid=False)
    assert result is ArbiterVote.INVALID


def test_arbiter_vote_agrees_a_b_or_neither() -> None:
    assert (
        arbiter_vote(FieldKind.TEXT, "Acme", "Beta", "Acme", disputed=True, arbiter_valid=True)
        is ArbiterVote.AGREES_A
    )
    assert (
        arbiter_vote(FieldKind.TEXT, "Acme", "Beta", "Beta", disputed=True, arbiter_valid=True)
        is ArbiterVote.AGREES_B
    )
    assert (
        arbiter_vote(FieldKind.TEXT, "Acme", "Beta", "Gamma", disputed=True, arbiter_valid=True)
        is ArbiterVote.AGREES_NEITHER
    )


def _features(**overrides: object) -> FieldFeatures:
    base: dict[str, object] = {
        "agreement": 1.0,
        "a_mean_logprob": -0.1,
        "a_min_logprob": -0.2,
        "b_mean_logprob": -0.1,
        "b_min_logprob": -0.2,
        "format_valid": True,
        "ocr_support": 0.8,
        "arbiter_vote": None,
        "arbiter": ArbiterVote.NOT_RUN,
    }
    base.update(overrides)
    return FieldFeatures(**base)  # type: ignore[arg-type]


def test_hard_reject_h1_a_or_b_invalid() -> None:
    assert hard_reject(_features(a_logprob_missing=True)) is True
    assert hard_reject(_features(b_logprob_missing=True)) is True
    assert hard_reject(_features()) is False


def test_hard_reject_h2_arbiter_invalid() -> None:
    feats = _features(arbiter=ArbiterVote.INVALID, arbiter_vote=False)
    assert hard_reject(feats) is True


def test_field_features_rejects_inconsistent_arbiter_vote() -> None:
    with pytest.raises(ValidationError):
        _features(arbiter=ArbiterVote.AGREES_A, arbiter_vote=None)


# --------------------------------------------------------------------------------------------
# encode_features
# --------------------------------------------------------------------------------------------


def test_encode_features_length_matches_feature_names() -> None:
    vector = encode_features(FieldName.TOTAL, _features())
    assert len(vector) == len(FEATURE_NAMES) == 28


def test_encode_features_field_one_hot_is_exclusive() -> None:
    vector = encode_features(FieldName.TOTAL, _features())
    one_hot = vector[14:]
    assert sum(one_hot) == 1.0
    total_index = list(FieldName).index(FieldName.TOTAL)
    assert one_hot[total_index] == 1.0


def test_encode_features_raises_for_hard_rejected_field() -> None:
    with pytest.raises(TrustGateError):
        encode_features(FieldName.TOTAL, _features(a_logprob_missing=True))


# --------------------------------------------------------------------------------------------
# labels lexicon
# --------------------------------------------------------------------------------------------


def test_labels_cover_every_field_with_en_and_ar_phrases() -> None:
    import unicodedata

    for field in FIELD_KINDS:
        phrases = LABELS.get(field, ())
        assert len(phrases) >= 4, field
        has_ar = any(
            any(unicodedata.name(ch, "").startswith("ARABIC") for ch in phrase)
            for phrase in phrases
        )
        has_en = any(phrase.isascii() for phrase in phrases)
        assert has_ar, f"{field} has no Arabic label phrase"
        assert has_en, f"{field} has no English label phrase"


# --------------------------------------------------------------------------------------------
# ocr_support (via a small synthetic ParsedDocument)
# --------------------------------------------------------------------------------------------


def _make_document() -> tuple[ParsedDocument, FieldValue]:
    label_word = Word(
        id="p0w0",
        text="Invoice",
        bbox=BBox(page=0, x0=0.05, y0=0.09, x1=0.15, y1=0.11),
        line_id="p0l0",
        confidence=0.99,
    )
    label_word2 = Word(
        id="p0w1",
        text="Number",
        bbox=BBox(page=0, x0=0.16, y0=0.09, x1=0.26, y1=0.11),
        line_id="p0l0",
        confidence=0.99,
    )
    value_word = Word(
        id="p0w2",
        text="INV-001",
        bbox=BBox(page=0, x0=0.20, y0=0.10, x1=0.30, y1=0.12),
        line_id="p0l1",
        confidence=0.98,
    )
    label_line = Line(
        id="p0l0",
        word_ids=("p0w0", "p0w1"),
        bbox=BBox(page=0, x0=0.05, y0=0.09, x1=0.26, y1=0.11),
    )
    value_line = Line(
        id="p0l1", word_ids=("p0w2",), bbox=BBox(page=0, x0=0.20, y0=0.10, x1=0.30, y1=0.12)
    )
    page = Page(
        index=0,
        width_px=1000,
        height_px=1400,
        words=(label_word, label_word2, value_word),
        lines=(label_line, value_line),
        tables=(),
        image_png=b"",
    )
    document = ParsedDocument(
        document_sha256=DOC_SHA,
        pages=(page,),
        reading_order=("p0l0", "p0l1"),
        languages=frozenset({Language.EN}),
        parser_version="test-1",
    )
    final = FieldValue(
        name="invoice_number", raw="INV-001", normalized="INV-001", word_ids=("p0w2",)
    )
    return document, final


def test_ocr_support_close_label_scores_high() -> None:
    document, final = _make_document()
    score = ocr_support(FieldName.INVOICE_NUMBER, final, document)
    assert score > 0.5


def test_ocr_support_null_value_is_zero() -> None:
    document, _final = _make_document()
    empty = FieldValue(name="invoice_number", raw=None, normalized=None)
    assert ocr_support(FieldName.INVOICE_NUMBER, empty, document) == 0.0


def test_ocr_support_no_matching_label_halves_score() -> None:
    document, final = _make_document()
    # PO_NUMBER's label phrases do not appear anywhere on the page.
    score = ocr_support(FieldName.PO_NUMBER, final, document)
    assert score == pytest.approx(1.0 * 0.5)


def _single_value_document(printed: str) -> ParsedDocument:
    """One line holding just the printed value, no label phrase (so ocr_support = m * 0.5)."""
    tokens = printed.split()
    words = tuple(
        Word(
            id=f"p0w{i}",
            text=token,
            bbox=BBox(page=0, x0=0.1 + 0.1 * i, y0=0.5, x1=0.19 + 0.1 * i, y1=0.52),
            line_id="p0l0",
            confidence=0.99,
        )
        for i, token in enumerate(tokens)
    )
    line = Line(
        id="p0l0",
        word_ids=tuple(w.id for w in words),
        bbox=BBox(page=0, x0=0.1, y0=0.5, x1=0.1 + 0.1 * len(words), y1=0.52),
    )
    page = Page(
        index=0,
        width_px=1000,
        height_px=1400,
        words=words,
        lines=(line,),
        tables=(),
        image_png=b"",
    )
    return ParsedDocument(
        document_sha256=DOC_SHA,
        pages=(page,),
        reading_order=("p0l0",),
        languages=frozenset({Language.EN, Language.AR}),
        parser_version="test-1",
    )


@pytest.mark.parametrize(
    ("field", "printed", "extracted"),
    [
        (FieldName.INVOICE_NUMBER, "INV-001", "INV-001"),
        (FieldName.INVOICE_NUMBER, "INV/2026/0042", "INV/2026/0042"),
        (FieldName.INVOICE_NUMBER, "فاتورة-١٢٣", "فاتورة-١٢٣"),
        (FieldName.INVOICE_NUMBER, "INV-١٢٣", "INV-123"),
        (FieldName.PO_NUMBER, "PO-77.A", "PO-77.A"),
        (FieldName.TOTAL, "1,050.00", "1,050.00"),
        (FieldName.TOTAL, "12,345.5", "12,345.50"),
        (FieldName.TOTAL, "\u0661,\u0660\u0665\u0660.\u0660\u0660", "1,050.00"),
        (FieldName.ISSUE_DATE, "15/01/2026", "15/01/2026"),
        (FieldName.ISSUE_DATE, "15 Jan 2026", "15 Jan 2026"),
        (FieldName.ISSUE_DATE, "\u0661\u0665/\u0660\u0661/\u0662\u0660\u0662\u0666", "15/01/2026"),
    ],
)
def test_ocr_support_matches_printed_words_under_the_value_kind(
    field: FieldName, printed: str, extracted: str
) -> None:
    """Regression (M7.5): the grounder's normalised form (e.g. `inv 001`) is not comparable
    with the printed words under the kind's rule, so the match uses the extracted raw value;
    both sides go through `normalize_for_compare(kind)`."""
    from apx.extract.grounding import WordGrounder
    from apx.extract.interfaces import RawField

    document = _single_value_document(printed)
    final = WordGrounder().ground(document, [RawField(name=field, value=extracted)])[field]
    assert final.word_ids, "the grounder must match the printed words"
    assert ocr_support(field, final, document) == pytest.approx(0.5)  # m = 1, no label line


def test_ocr_support_mismatched_value_scores_zero() -> None:
    document = _single_value_document("INV-001")
    final = FieldValue(
        name="invoice_number", raw="INV-002", normalized="inv 002", word_ids=("p0w0",)
    )
    assert ocr_support(FieldName.INVOICE_NUMBER, final, document) == 0.0


# --------------------------------------------------------------------------------------------
# ExtractionFeatureBuilder (end-to-end wiring)
# --------------------------------------------------------------------------------------------


def _extractor_output(
    extractor_id: ExtractorId, value: str | None, valid_json: bool = True
) -> ExtractorOutput:
    return ExtractorOutput(
        extractor=extractor_id,
        model="test-model",
        fields=(RawField(name=FieldName.INVOICE_NUMBER, value=value),),
        line_items=(),
        logprobs={FieldName.INVOICE_NUMBER: FieldLogprobs(mean=-0.1, min=-0.5)},
        valid_json=valid_json,
    )


def test_feature_builder_produces_all_header_fields() -> None:
    document, final = _make_document()
    a = _extractor_output(ExtractorId.A, "INV-001")
    b = _extractor_output(ExtractorId.B, "INV-001")
    result = ExtractionResult(
        document_sha256=DOC_SHA,
        a=a,
        b=b,
        arbiter=None,
        disputed=frozenset(),
        grounded={FieldName.INVOICE_NUMBER: final},
    )
    features = ExtractionFeatureBuilder().features(result, document)
    assert set(features.keys()) == set(FIELD_KINDS.keys())
    assert features[FieldName.INVOICE_NUMBER].agreement == 1.0
    assert features[FieldName.INVOICE_NUMBER].a_missing is False


def test_feature_builder_marks_h1_when_extractor_invalid() -> None:
    document, final = _make_document()
    a = _extractor_output(ExtractorId.A, "INV-001", valid_json=False)
    b = _extractor_output(ExtractorId.B, "INV-001")
    result = ExtractionResult(
        document_sha256=DOC_SHA,
        a=a,
        b=b,
        arbiter=None,
        disputed=frozenset({FieldName.INVOICE_NUMBER}),
        grounded={FieldName.INVOICE_NUMBER: final},
    )
    features = ExtractionFeatureBuilder().features(result, document)
    invoice_features = features[FieldName.INVOICE_NUMBER]
    assert invoice_features.a_logprob_missing is True
    assert hard_reject(invoice_features) is True
    assert invoice_features.arbiter is ArbiterVote.NOT_RUN
