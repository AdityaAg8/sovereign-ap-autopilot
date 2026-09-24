"""Tests for `apx.extract.schema`: full BT-code coverage of the 51 mandatory header fields, a
gold round-trip through JSON, and the llama.cpp JSON-Schema keyword whitelist (PRD §10 M3.2)."""

from __future__ import annotations

import json
import re
from typing import Any

from apx.extract.normalize import (
    normalize_amount,
    normalize_date,
    normalize_iban,
    normalize_trn,
)
from apx.extract.schema import (
    HEADER_FIELD_BT,
    LINE_ITEM_FIELD_BT,
    LLAMA_SCHEMA_KEYWORDS,
    InvoiceFields,
    LineItemFields,
    invoice_fields_json_schema,
)

_BT_CODE_RE = re.compile(r"^BT-\d+$")


def test_header_field_bt_covers_all_51_mandatory_fields() -> None:
    header_fields = set(InvoiceFields.model_fields) - {"line_items"}
    assert header_fields == set(HEADER_FIELD_BT)
    assert len(HEADER_FIELD_BT) == 51
    assert len(set(HEADER_FIELD_BT.values())) == 51, "BT codes must be unique per header field"
    assert all(_BT_CODE_RE.match(code) for code in HEADER_FIELD_BT.values())


def test_line_item_field_bt_covers_every_line_field() -> None:
    line_fields = set(LineItemFields.model_fields)
    assert line_fields == set(LINE_ITEM_FIELD_BT)
    assert len(set(LINE_ITEM_FIELD_BT.values())) == len(LINE_ITEM_FIELD_BT)
    assert all(_BT_CODE_RE.match(code) for code in LINE_ITEM_FIELD_BT.values())


def _gold_invoice() -> InvoiceFields:
    return InvoiceFields(
        invoice_number="INV-2024-0001",
        issue_date="15 يناير 2024",  # "15 January 2024" in Arabic
        due_date="2024-02-14",
        currency_code="AED",
        specification_identifier="urn:peppol:pint:billing-1@ae-1",
        supplier_name="Al Futtaim Trading LLC",
        # "100456789012345" written with Arabic-Indic digits.
        supplier_trn="١٠٠٤٥٦٧٨٩٠١٢٣٤٥",
        buyer_name="Emirates Global Aluminium",
        buyer_trn="100987654321098",
        payment_account_iban=" ae07 0331 2345 6789 0123 456 ",
        sum_line_net_amount="AED 1,234.56",
        tax_exclusive_amount="1,234.56",
        vat_total_amount="61.73",
        tax_inclusive_amount="1,296.29",
        payable_amount="1,296.29",
        line_items=(
            LineItemFields(
                line_id="1",
                item_name="Widget",
                quantity="10",
                unit_code="EA",
                unit_price="123.456",
                net_amount="1,234.56",
                vat_category_code="S",
                vat_rate="5",
            ),
        ),
    )


def test_gold_round_trip_through_json() -> None:
    gold = _gold_invoice()
    restored = InvoiceFields.model_validate_json(gold.model_dump_json())
    assert restored == gold
    # A second round trip changes nothing further (idempotent serialization).
    assert InvoiceFields.model_validate_json(restored.model_dump_json()) == restored


def test_normalization_of_gold_fields_is_idempotent() -> None:
    gold = _gold_invoice()
    assert gold.issue_date is not None
    assert gold.supplier_trn is not None
    assert gold.payment_account_iban is not None
    assert gold.tax_exclusive_amount is not None

    once_date = normalize_date(gold.issue_date)
    assert once_date.isoformat() == "2024-01-15"
    assert normalize_date(once_date.isoformat()) == once_date

    once_trn = normalize_trn(gold.supplier_trn)
    assert once_trn == "100456789012345"
    assert normalize_trn(once_trn) == once_trn

    once_iban = normalize_iban(gold.payment_account_iban)
    assert once_iban == "AE070331234567890123456"
    assert normalize_iban(once_iban) == once_iban

    once_amount = normalize_amount(gold.tax_exclusive_amount)
    assert normalize_amount(str(once_amount)) == once_amount


_NAME_KEYED_CONTAINERS = {"$defs", "properties"}
"""`$defs` and `properties` are keyed by definition/property *names*, not JSON Schema keywords;
those names must not be checked against the keyword whitelist."""


def _collect_keys(node: Any) -> set[str]:
    """All JSON Schema keyword keys used anywhere in `node`, skipping the arbitrary names nested
    under `$defs`/`properties` (see `_NAME_KEYED_CONTAINERS`)."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            if key in _NAME_KEYED_CONTAINERS and isinstance(value, dict):
                for subschema in value.values():
                    keys |= _collect_keys(subschema)
            else:
                keys |= _collect_keys(value)
    elif isinstance(node, list):
        for item in node:
            keys |= _collect_keys(item)
    return keys


def test_json_schema_export_is_llama_grammar_compatible() -> None:
    schema = invoice_fields_json_schema()

    assert schema["additionalProperties"] is False
    header_properties = schema["properties"]  # type: ignore[index]
    assert set(header_properties) == set(HEADER_FIELD_BT) | {"line_items"}

    defs = schema["$defs"]  # type: ignore[index]
    for definition in defs.values():
        assert definition["additionalProperties"] is False
    line_item_properties = defs["LineItemFields"]["properties"]
    assert set(line_item_properties) == set(LINE_ITEM_FIELD_BT)

    used_keys = _collect_keys(schema)
    assert used_keys <= LLAMA_SCHEMA_KEYWORDS, used_keys - LLAMA_SCHEMA_KEYWORDS

    # Must be plain, serializable JSON (this is what response_format.json_schema receives).
    json.dumps(schema)


def test_json_schema_line_items_reference_has_no_cycle() -> None:
    schema = invoice_fields_json_schema()
    defs = schema["$defs"]  # type: ignore[index]
    assert "LineItemFields" in defs
    line_items_schema = schema["properties"]["line_items"]  # type: ignore[index]
    assert line_items_schema["items"]["$ref"] == "#/$defs/LineItemFields"
