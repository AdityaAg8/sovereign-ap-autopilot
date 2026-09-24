"""Tests for the ADR-0012 extraction contract: the enum-constrained BT-3 / BT-151 schema, the
printed-vs-classified line keys, A/B agreement resolution (disagreement or null gives None, which
routes to review) and the classification rules shared by the A and B prompts."""

from __future__ import annotations

from apx.common.invoice import VatCategory
from apx.extract.interfaces import (
    INVOICE_TYPE_CODE_KEY,
    LINE_ITEM_KEYS,
    PRINTED_LINE_ITEM_KEYS,
    ExtractorId,
    ExtractorOutput,
    FieldName,
    InvoiceTypeCode,
    RawField,
    RawLineItem,
    extraction_json_schema,
    resolve_classification,
)
from apx.extract.prompt import CLASSIFICATION_GUIDE, SYSTEM_PROMPT
from apx.extract.prompt_b import B_SYSTEM_PROMPT

S, Z, E = VatCategory.STANDARD, VatCategory.ZERO_RATED, VatCategory.EXEMPT
INV, CN = InvoiceTypeCode.INVOICE, InvoiceTypeCode.CREDIT_NOTE


def _out(
    extractor: ExtractorId,
    type_code: InvoiceTypeCode | None,
    categories: list[VatCategory | None],
    *,
    valid: bool = True,
) -> ExtractorOutput:
    return ExtractorOutput(
        extractor=extractor,
        model="m",
        fields=tuple(RawField(name=n, value=None) for n in FieldName),
        line_items=tuple(
            RawLineItem(
                description="x",
                quantity="1",
                unit_price="1",
                net_amount="1",
                vat_rate=None,
                vat_category=c,
            )
            for c in categories
        ),
        logprobs={},
        valid_json=valid,
        invoice_type_code=type_code,
    )


def test_schema_constrains_classified_values_to_their_enums() -> None:
    schema = extraction_json_schema()
    props = schema["properties"]
    assert isinstance(props, dict)
    assert props[INVOICE_TYPE_CODE_KEY] == {"enum": ["380", "381", None]}
    line_items = props["line_items"]
    assert isinstance(line_items, dict)
    items = line_items["items"]
    assert isinstance(items, dict)
    line_props = items["properties"]
    assert isinstance(line_props, dict)
    assert line_props["vat_category"] == {"enum": ["S", "Z", "E", "AE", "O", None]}
    required = schema["required"]
    assert isinstance(required, list)
    assert required[-2:] == [INVOICE_TYPE_CODE_KEY, "line_items"]


def test_printed_line_keys_exclude_the_classified_category() -> None:
    assert (*PRINTED_LINE_ITEM_KEYS, "vat_category") == LINE_ITEM_KEYS


def test_agreement_keeps_values_and_any_disagreement_or_null_drops_them() -> None:
    agreed = resolve_classification(
        _out(ExtractorId.A, INV, [S, Z]), _out(ExtractorId.B, INV, [S, Z])
    )
    assert agreed.invoice_type_code is INV and agreed.line_vat_categories == (S, Z)
    split = resolve_classification(
        _out(ExtractorId.A, INV, [S, None]), _out(ExtractorId.B, CN, [E, None])
    )
    assert split.invoice_type_code is None and split.line_vat_categories == (None, None)


def test_unequal_line_counts_keep_only_a_single_agreed_category() -> None:
    one = resolve_classification(_out(ExtractorId.A, INV, [E, E, E]), _out(ExtractorId.B, INV, [E]))
    assert one.line_vat_categories == (E, E, E)
    mixed = resolve_classification(
        _out(ExtractorId.A, INV, [S, Z, S]), _out(ExtractorId.B, INV, [S, Z])
    )
    assert mixed.line_vat_categories == (None, None, None)


def test_unequal_line_counts_are_symmetric_when_b_has_more_lines() -> None:
    more_b = resolve_classification(
        _out(ExtractorId.A, INV, [Z, Z]), _out(ExtractorId.B, INV, [Z, Z, Z])
    )
    assert more_b.line_vat_categories == (Z, Z)
    a_mixed = resolve_classification(
        _out(ExtractorId.A, INV, [S, Z]), _out(ExtractorId.B, INV, [S, S, S])
    )
    swapped = resolve_classification(
        _out(ExtractorId.A, INV, [S, S, S]), _out(ExtractorId.B, INV, [S, Z])
    )
    assert a_mixed.line_vat_categories == (None, None)
    assert swapped.line_vat_categories == (None, None, None)


def test_invalid_json_on_either_side_gives_no_classification() -> None:
    a_bad = resolve_classification(
        _out(ExtractorId.A, INV, [S], valid=False), _out(ExtractorId.B, INV, [S, S])
    )
    assert a_bad.invoice_type_code is None and a_bad.line_vat_categories == (None, None)
    b_bad = resolve_classification(
        _out(ExtractorId.A, INV, [S]), _out(ExtractorId.B, INV, [S], valid=False)
    )
    assert b_bad.invoice_type_code is None and b_bad.line_vat_categories == (None,)


def test_both_prompts_carry_the_classification_rules() -> None:
    for prompt in (SYSTEM_PROMPT, B_SYSTEM_PROMPT):
        assert CLASSIFICATION_GUIDE in prompt
        for code in ('"380"', '"381"', '"S"', '"Z"', '"E"', '"AE"', '"O"'):
            assert code in prompt
