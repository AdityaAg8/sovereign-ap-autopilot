"""Tests for the Extractor B label mappings (ADR-0009 §1.3-1.4): UAE, DocILE and FATURA targets
in schema order, every source label mapped or explicitly ignored, the multi-annotation merge
rule and the FATURA eligibility rule."""

from __future__ import annotations

import json

import pytest

from apx.extract.interfaces import LINE_ITEM_KEYS, FieldName, extraction_json_schema
from ml.extractor_b.interfaces import TrainingError
from ml.extractor_b.mappings import (
    DOCILE_KILE,
    DOCILE_KILE_IGNORED,
    DOCILE_LIR,
    DOCILE_LIR_IGNORED,
    FATURA_FIELDS,
    FATURA_IGNORED,
    HEADER_KEYS,
    docile_target,
    empty_target,
    fatura_eligible,
    fatura_target,
    merge_annotations,
    strip_key,
    uae_classification,
    uae_target,
)

# Every top-level key seen across the 10,000 FATURA Original_Format annotations.
_OBSERVED_KEYS = (
    "TABLE INVOICE_INFO OTHER LOGO DATE NUMBER SELLER_ADDRESS TOTAL TITLE SELLER_NAME "
    "SUB_TOTAL BUYER DUE_DATE NOTE SELLER_EMAIL TOTAL_WORDS TAX PAYMENT_DETAILS "
    "CONDITIONS DISCOUNT SEND_TO BILL_TO GSTIN PO_NUMBER SELLER_SITE GSTIN_SELLER "
    "AMOUNT_DUE GSTIN_BUYER GST(18%) GST(9%) GST(1%) GST(20%) GST(12%) GST(5%) GST(7%) "
)
FATURA_OBSERVED = _OBSERVED_KEYS.split()


def _schema_ok(target: dict[str, object]) -> None:
    import jsonschema

    jsonschema.validate(target, extraction_json_schema())
    assert list(target) == [*HEADER_KEYS, "invoice_type_code", "line_items"]
    for item in target["line_items"]:  # type: ignore[attr-defined]
        assert list(item) == list(LINE_ITEM_KEYS)


def test_empty_target_matches_schema() -> None:
    _schema_ok(empty_target())


def test_uae_target() -> None:
    labels = {
        "labels": [
            {"field": "supplier_name", "value": "شركة القمة", "line_index": None},
            {"field": "supplier_name", "value": "Peak IT FZ-LLC", "line_index": None},
            {"field": "buyer_name", "value": "مؤسسة", "line_index": None},
            {"field": "total", "value": "1,050.00", "line_index": None},
            {"field": "description", "value": "Laptop", "line_index": 1},
            {"field": "description", "value": "stand", "line_index": 1},
            {"field": "quantity", "value": "2", "line_index": 1},
            {"field": "description", "value": "Cable", "line_index": 0},
        ]
    }
    target = uae_target(labels, "reverse_charge")
    _schema_ok(target)
    assert target["invoice_type_code"] == "380"
    assert {li["vat_category"] for li in target["line_items"]} == {"AE"}
    assert target["supplier_name"] == "Peak IT FZ-LLC"  # Latin form of a bilingual name
    assert target["buyer_name"] == "مؤسسة"  # Arabic-only stays Arabic
    assert [li["description"] for li in target["line_items"]] == ["Cable", "Laptop stand"]
    assert (
        target["line_items"][1]["quantity"] == "2" and target["line_items"][0]["quantity"] is None
    )
    with pytest.raises(TrainingError):
        uae_target({"labels": [{"field": "swift", "value": "x", "line_index": None}]}, "standard")
    with pytest.raises(TrainingError):
        uae_target({"labels": [{"field": "sku", "value": "x", "line_index": 0}]}, "standard")
    with pytest.raises(TrainingError, match="tax_case"):
        uae_target({"labels": []}, "luxury")
    with pytest.raises(TrainingError):  # the classified key is never a factory label
        uae_target(
            {"labels": [{"field": "vat_category", "value": "S", "line_index": 0}]}, "standard"
        )


@pytest.mark.parametrize(
    ("tax_case", "category"),
    [("standard", "S"), ("zero_rated", "Z"), ("exempt", "E"), ("reverse_charge", "AE"),
     ("designated_zone", "O")],
)  # fmt: skip
def test_uae_classification_matches_the_gold_xml_mapping(tax_case: str, category: str) -> None:
    """Training labels and the M2.5 gold XML agree on every factory tax case (one truth)."""
    from factory.gold_xml import TAX_CASE_VAT_CATEGORY as GOLD

    type_code, got = uae_classification(tax_case)
    assert type_code.value == "380" and got.value == category == GOLD[tax_case]


def test_uae_labels_cover_the_schema() -> None:
    assert set(HEADER_KEYS) == {f.value for f in FieldName}
    assert set(DOCILE_LIR.values()) <= set(LINE_ITEM_KEYS)


def test_docile_mapping_is_complete_and_disjoint() -> None:
    assert set(DOCILE_KILE.values()) == set(HEADER_KEYS)
    assert not set(DOCILE_KILE) & DOCILE_KILE_IGNORED
    assert not set(DOCILE_LIR) & DOCILE_LIR_IGNORED


def test_docile_target_merges_in_reading_order() -> None:
    annotation = {
        "field_extractions": [
            {
                "fieldtype": "vendor_address",
                "text": "Berlin",
                "page": 0,
                "bbox": [0.1, 0.3, 0.2, 0.32],
            },
            {
                "fieldtype": "vendor_address",
                "text": "Main St 1",
                "page": 0,
                "bbox": [0.1, 0.2, 0.2, 0.22],
            },
            {
                "fieldtype": "amount_total_gross",
                "text": "1.050,00",
                "page": 1,
                "bbox": [0, 0.9, 0, 0],
            },
            {"fieldtype": "amount_total_gross", "text": "99.00", "page": 0, "bbox": [0, 0.9, 0, 0]},
            {"fieldtype": "document_id", "text": "A-1", "page": 0, "bbox": [0, 0.1, 0, 0]},
            {"fieldtype": "document_id", "text": "a-1", "page": 0, "bbox": [0, 0.5, 0, 0]},
            {"fieldtype": "payment_terms", "text": "30 days", "page": 0, "bbox": [0, 0, 0, 0]},
        ],
        "line_item_extractions": [
            {
                "fieldtype": "line_item_description",
                "text": "Widget",
                "line_item_id": 2,
                "page": 0,
                "bbox": [0, 0.5, 0, 0],
            },
            {
                "fieldtype": "line_item_quantity",
                "text": "3",
                "line_item_id": 2,
                "page": 0,
                "bbox": [0.5, 0.5, 0, 0],
            },
            {
                "fieldtype": "line_item_code",
                "text": "W1",
                "line_item_id": 2,
                "page": 0,
                "bbox": [0, 0.5, 0, 0],
            },
        ],
    }
    target = docile_target(annotation)
    _schema_ok(target)
    assert target["supplier_address"] == "Main St 1 Berlin"  # TEXT: joined in reading order
    assert target["total"] == "99.00"  # non-TEXT, different values: first in reading order
    assert target["invoice_number"] == "A-1"
    assert target["line_items"] == [
        {
            "description": "Widget",
            "quantity": "3",
            "unit_price": None,
            "net_amount": None,
            "vat_rate": None,
            "vat_category": None,  # DocILE is not UAE VAT: unknown, never guessed
        }
    ]
    assert target["invoice_type_code"] is None
    with pytest.raises(TrainingError):
        docile_target({"field_extractions": [{"fieldtype": "brand_new", "text": "x"}]})
    with pytest.raises(TrainingError):
        docile_target(
            {
                "line_item_extractions": [
                    {"fieldtype": "line_item_new", "text": "x", "line_item_id": 0}
                ]
            }
        )


def test_merge_equal_normalised_values_keeps_one() -> None:
    assert merge_annotations("total", ["1,050.00", "1050.00"]) == "1,050.00"
    assert merge_annotations("supplier_name", ["ACME", "Acme"]) == "ACME"


def test_every_observed_fatura_key_is_mapped_or_ignored() -> None:
    for key in FATURA_OBSERVED:
        annotation = {key: {"text": "Total : 1.00 USD"}}
        fatura_target(annotation)  # raises on an unmapped key
    assert not set(FATURA_FIELDS) & FATURA_IGNORED
    with pytest.raises(TrainingError):
        fatura_target({"NEW_BOX": {"text": "x"}})


def test_fatura_target_strips_key_text() -> None:
    annotation = {
        "NUMBER": {"text": "INVOICE # INV/30-14/832"},
        "DATE": {"text": "Date: 03-Jan-1994"},
        "DUE_DATE": {"text": "Due Date : 06-Feb-2007"},
        "BUYER": {"text": "Bill to:Michael Sparks\n4894 Melanie Pass"},
        "SELLER_ADDRESS": {"text": "Address:4894 Melanie Pass Apt. 742\nTammyland, SD 42587 US"},
        "SUB_TOTAL": {"text": "SUB_TOTAL : 400.00  $"},
        "GST(5%)": {"text": "GST(5%) : 20.00"},
        "TOTAL": {"text": "TOTAL : 420.00 EUR"},
        "TABLE": [[{"bbox": [[0, 0], [1, 1]]}]],
        "LOGO": {"bbox": [[0, 0], [1, 1]]},
    }
    target = fatura_target(annotation)
    _schema_ok(target)
    assert target["invoice_number"] == "INV/30-14/832"
    assert target["issue_date"] == "03-Jan-1994" and target["due_date"] == "06-Feb-2007"
    assert target["buyer_name"] == "Michael Sparks"
    assert target["supplier_address"] == "4894 Melanie Pass Apt. 742 Tammyland, SD 42587 US"
    assert (target["subtotal"], target["vat_total"], target["total"]) == (
        "400.00",
        "20.00",
        "420.00",
    )
    assert target["currency"] == "EUR"  # "$" is not an ISO code, "EUR" is
    assert target["line_items"] == []
    assert fatura_eligible(annotation)


def test_fatura_eligibility() -> None:
    assert not fatura_eligible({"DISCOUNT": {"text": "DISCOUNT(2%): (-) 9.39"}})
    inconsistent = {
        "SUB_TOTAL": {"text": "SUB_TOTAL : 383.26"},
        "TAX": {"text": "TAX: 23.95"},
        "TOTAL": {"text": "TOTAL : 392.61"},
    }
    assert not fatura_eligible(inconsistent)
    assert fatura_eligible({"TOTAL": {"text": "TOTAL : 392.61"}})


def test_strip_key() -> None:
    assert strip_key("Invoice Date: 26-Jan-2000") == "26-Jan-2000"
    assert strip_key("PO Number :35") == "35"
    assert strip_key("Mclean-Cochran") == "Mclean-Cochran"
    assert json.dumps(strip_key("Buyer :Andrew White")) == '"Andrew White"'
