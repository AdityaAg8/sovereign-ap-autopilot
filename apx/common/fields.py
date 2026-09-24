"""The UAE PINT AE invoice field schema: `InvoiceFields` covers the guideline's 51 mandatory
header business terms plus repeating line items, and exports a JSON Schema restricted to the
keywords llama.cpp's grammar converter supports, for use as llama-server's structured-output
`response_format` (PRD §10 M3.2, §4)."""

from __future__ import annotations

from typing import Any, Final, cast

from pydantic import JsonValue

from apx.common.types import Frozen

# Keywords `json-schema.cpp`'s `build_node` understands (confirmed against llama.cpp source,
# 2026-09-22). Anything else (`title`, `description`, `default`, ...) is stripped from the
# exported schema so it stays inside what the grammar converter can turn into GBNF.
LLAMA_SCHEMA_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "$ref",
        "$defs",
        "type",
        "oneOf",
        "anyOf",
        "allOf",
        "const",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "pattern",
        "minLength",
        "maxLength",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }
)


class LineItemFields(Frozen):
    """One PINT AE invoice line, raw (pre-normalization) text as an extractor emits it. Business
    terms follow the Peppol BIS Billing 3.0 / EN16931 vocabulary that PINT AE specializes."""

    line_id: str | None = None  # BT-126 Invoice line identifier
    item_name: str | None = None  # BT-153 Item name
    item_description: str | None = None  # BT-154 Item description
    item_classification_id: str | None = None  # BT-156 Item classification identifier
    item_standard_id: str | None = None  # BT-157 Item standard identifier (e.g. GTIN)
    quantity: str | None = None  # BT-129 Invoiced quantity
    unit_code: str | None = None  # BT-130 Invoiced quantity unit of measure code
    unit_price: str | None = None  # BT-146 Item net price
    price_base_quantity: str | None = None  # BT-149 Item price base quantity
    price_base_quantity_unit_code: str | None = None  # BT-150 Item price base quantity unit code
    net_amount: str | None = None  # BT-131 Invoice line net amount
    vat_category_code: str | None = None  # BT-151 Invoiced item VAT category code
    vat_rate: str | None = None  # BT-152 Invoiced item VAT rate
    buyer_accounting_reference: str | None = None  # BT-133 Invoice line buyer accounting reference


LINE_ITEM_FIELD_BT: Final[dict[str, str]] = {
    "line_id": "BT-126",
    "item_name": "BT-153",
    "item_description": "BT-154",
    "item_classification_id": "BT-156",
    "item_standard_id": "BT-157",
    "quantity": "BT-129",
    "unit_code": "BT-130",
    "unit_price": "BT-146",
    "price_base_quantity": "BT-149",
    "price_base_quantity_unit_code": "BT-150",
    "net_amount": "BT-131",
    "vat_category_code": "BT-151",
    "vat_rate": "BT-152",
    "buyer_accounting_reference": "BT-133",
}
"""Field name -> PINT AE business term, one entry per `LineItemFields` attribute."""


class InvoiceFields(Frozen):
    """Raw (pre-normalization) values for the UAE guideline's 51 mandatory header business terms,
    plus line items. Every attribute is optional text: extraction may miss a mandatory field, and
    `apx.validate.ViolationCode.MISSING_MANDATORY` is what flags that, not this type. Values are
    normalized by `apx.common.normalize` after grounding."""

    # Document (BT-1 .. BT-25)
    invoice_number: str | None = None  # BT-1 Invoice number
    issue_date: str | None = None  # BT-2 Invoice issue date
    invoice_type_code: str | None = None  # BT-3 Invoice type code
    currency_code: str | None = None  # BT-5 Invoice currency code
    vat_accounting_currency_code: str | None = None  # BT-6 VAT accounting currency code
    vat_point_date: str | None = None  # BT-7 Value added tax point date
    vat_point_date_code: str | None = None  # BT-8 Value added tax point date code
    due_date: str | None = None  # BT-9 Payment due date
    buyer_reference: str | None = None  # BT-10 Buyer reference
    project_reference: str | None = None  # BT-11 Project reference
    contract_reference: str | None = None  # BT-12 Contract reference
    purchase_order_reference: str | None = None  # BT-13 Purchase order reference
    buyer_accounting_reference: str | None = None  # BT-19 Buyer accounting reference
    payment_terms: str | None = None  # BT-20 Payment terms
    invoice_note: str | None = None  # BT-22 Invoice note
    business_process_type: str | None = None  # BT-23 Business process type
    specification_identifier: str | None = None  # BT-24 Specification identifier
    preceding_invoice_reference: str | None = None  # BT-25 Preceding invoice reference

    # Seller / supplier (BT-27 .. BT-41)
    supplier_name: str | None = None  # BT-27 Seller name
    supplier_identifier: str | None = None  # BT-29 Seller identifier
    supplier_legal_registration_id: str | None = None  # BT-30 Seller legal registration id
    supplier_trn: str | None = None  # BT-31 Seller VAT identifier (TRN)
    supplier_legal_info: str | None = None  # BT-33 Seller additional legal information
    supplier_electronic_address: str | None = None  # BT-34 Seller electronic address
    supplier_address_line1: str | None = None  # BT-35 Seller address line 1
    supplier_city: str | None = None  # BT-37 Seller city
    supplier_post_code: str | None = None  # BT-38 Seller post code
    supplier_country_code: str | None = None  # BT-40 Seller country code
    supplier_country_subdivision: str | None = None  # BT-39 Seller country subdivision

    # Buyer (BT-44 .. BT-55)
    buyer_name: str | None = None  # BT-44 Buyer name
    buyer_identifier: str | None = None  # BT-46 Buyer identifier
    buyer_legal_registration_id: str | None = None  # BT-47 Buyer legal registration id
    buyer_trn: str | None = None  # BT-48 Buyer VAT identifier (TRN)
    buyer_electronic_address: str | None = None  # BT-49 Buyer electronic address
    buyer_address_line1: str | None = None  # BT-50 Buyer address line 1
    buyer_city: str | None = None  # BT-52 Buyer city
    buyer_post_code: str | None = None  # BT-53 Buyer post code
    buyer_country_code: str | None = None  # BT-55 Buyer country code

    # Payee and tax representative (BT-59 .. BT-63)
    payee_name: str | None = None  # BT-59 Payee name
    payee_identifier: str | None = None  # BT-60 Payee identifier
    payee_legal_registration_id: str | None = None  # BT-61 Payee legal registration id
    tax_representative_name: str | None = None  # BT-62 Tax representative name
    tax_representative_trn: str | None = None  # BT-63 Tax representative VAT identifier

    # Delivery and payment (BT-72, BT-81, BT-84)
    actual_delivery_date: str | None = None  # BT-72 Actual delivery date
    payment_means_code: str | None = None  # BT-81 Payment means type code
    payment_account_iban: str | None = None  # BT-84 Payment account identifier (IBAN)

    # Document totals (BT-106 .. BT-115)
    sum_line_net_amount: str | None = None  # BT-106 Sum of invoice line net amounts
    tax_exclusive_amount: str | None = None  # BT-109 Invoice total amount without VAT
    vat_total_amount: str | None = None  # BT-110 Invoice total VAT amount
    tax_inclusive_amount: str | None = None  # BT-112 Invoice total amount with VAT
    payable_amount: str | None = None  # BT-115 Amount due for payment

    line_items: tuple[LineItemFields, ...] = ()


HEADER_FIELD_BT: Final[dict[str, str]] = {
    "invoice_number": "BT-1",
    "issue_date": "BT-2",
    "invoice_type_code": "BT-3",
    "currency_code": "BT-5",
    "vat_accounting_currency_code": "BT-6",
    "vat_point_date": "BT-7",
    "vat_point_date_code": "BT-8",
    "due_date": "BT-9",
    "buyer_reference": "BT-10",
    "project_reference": "BT-11",
    "contract_reference": "BT-12",
    "purchase_order_reference": "BT-13",
    "buyer_accounting_reference": "BT-19",
    "payment_terms": "BT-20",
    "invoice_note": "BT-22",
    "business_process_type": "BT-23",
    "specification_identifier": "BT-24",
    "preceding_invoice_reference": "BT-25",
    "supplier_name": "BT-27",
    "supplier_identifier": "BT-29",
    "supplier_legal_registration_id": "BT-30",
    "supplier_trn": "BT-31",
    "supplier_legal_info": "BT-33",
    "supplier_electronic_address": "BT-34",
    "supplier_address_line1": "BT-35",
    "supplier_city": "BT-37",
    "supplier_post_code": "BT-38",
    "supplier_country_code": "BT-40",
    "supplier_country_subdivision": "BT-39",
    "buyer_name": "BT-44",
    "buyer_identifier": "BT-46",
    "buyer_legal_registration_id": "BT-47",
    "buyer_trn": "BT-48",
    "buyer_electronic_address": "BT-49",
    "buyer_address_line1": "BT-50",
    "buyer_city": "BT-52",
    "buyer_post_code": "BT-53",
    "buyer_country_code": "BT-55",
    "payee_name": "BT-59",
    "payee_identifier": "BT-60",
    "payee_legal_registration_id": "BT-61",
    "tax_representative_name": "BT-62",
    "tax_representative_trn": "BT-63",
    "actual_delivery_date": "BT-72",
    "payment_means_code": "BT-81",
    "payment_account_iban": "BT-84",
    "sum_line_net_amount": "BT-106",
    "tax_exclusive_amount": "BT-109",
    "vat_total_amount": "BT-110",
    "tax_inclusive_amount": "BT-112",
    "payable_amount": "BT-115",
}
"""Field name -> PINT AE business term, one entry per `InvoiceFields` header attribute (the UAE
guideline's 51 mandatory fields; `line_items` is the separate repeating structure)."""


_NAME_KEYED_CONTAINERS: Final[frozenset[str]] = frozenset({"$defs", "properties"})
"""Keywords whose object VALUE is keyed by arbitrary names (definition or property names), not
by further JSON Schema keywords, so those inner keys must survive the whitelist filter as-is."""


def _restrict_keywords(node: Any) -> JsonValue:
    """Recursively drop every JSON Schema keyword not in `LLAMA_SCHEMA_KEYWORDS`, without
    touching the definition/property *names* nested under `$defs` and `properties`."""
    if isinstance(node, dict):
        result: dict[str, JsonValue] = {}
        for key, value in node.items():
            if key not in LLAMA_SCHEMA_KEYWORDS:
                continue
            if key in _NAME_KEYED_CONTAINERS and isinstance(value, dict):
                result[key] = {
                    str(name): _restrict_keywords(subschema) for name, subschema in value.items()
                }
            else:
                result[key] = _restrict_keywords(value)
        return result
    if isinstance(node, list):
        return [_restrict_keywords(item) for item in node]
    return cast(JsonValue, node)


def _ref_graph(defs: dict[str, JsonValue]) -> dict[str, set[str]]:
    """Adjacency of `$defs` name -> the `$defs` names it `$ref`s, for cycle detection."""
    graph: dict[str, set[str]] = {name: set() for name in defs}

    def walk(node: JsonValue, source: str) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                graph[source].add(ref.removeprefix("#/$defs/"))
            for value in node.values():
                walk(value, source)
        elif isinstance(node, list):
            for item in node:
                walk(item, source)

    for name, subschema in defs.items():
        walk(subschema, name)
    return graph


def _has_cycle(graph: dict[str, set[str]]) -> bool:
    """DFS cycle check over a `$defs` reference graph."""
    unvisited, visiting, done = 0, 1, 2
    state = dict.fromkeys(graph, unvisited)

    def visit(node: str) -> bool:
        state[node] = visiting
        for neighbor in graph.get(node, set()):
            if state.get(neighbor, unvisited) == visiting:
                return True
            if state.get(neighbor, unvisited) == unvisited and visit(neighbor):
                return True
        state[node] = done
        return False

    return any(state[name] == unvisited and visit(name) for name in graph)


def invoice_fields_json_schema() -> dict[str, JsonValue]:
    """`InvoiceFields` as a JSON Schema usable for llama-server's `response_format.json_schema`:
    `additionalProperties: false` throughout (inherited from `Frozen`), no `$ref` cycles, and only
    keywords `LLAMA_SCHEMA_KEYWORDS` lists. Raises `ValueError` if a cycle is ever introduced."""
    raw_schema = InvoiceFields.model_json_schema()
    schema = _restrict_keywords(raw_schema)
    if not isinstance(schema, dict):  # pragma: no cover - model_json_schema always returns a dict
        raise ValueError("InvoiceFields.model_json_schema() did not return an object")
    defs = schema.get("$defs")
    if isinstance(defs, dict) and _has_cycle(_ref_graph(defs)):
        raise ValueError(
            "InvoiceFields JSON schema has a $ref cycle; llama.cpp's grammar converter cannot "
            "expand a cyclic schema"
        )
    return schema
