"""Interfaces for extraction: the invoice field schema, tool-free Extractor A/B and arbiter
clients, and grounding of values onto parse words. Model output carries no boxes; boxes are
attached only by the Grounder (PRD §3, §4, §6.2 I1/I2, ADR-0003). Classified values (BT-3 type
code, per-line VAT category) are enum-constrained and resolved by A/B agreement (ADR-0012)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import JsonValue

from apx.common.invoice import VatCategory
from apx.common.types import FieldValue, Frozen, Sha256Hex
from apx.extract.normalize import NormalizationError, normalize_field, normalize_text
from apx.llm.interfaces import StructuredRequest
from apx.parse.interfaces import ParsedDocument


class FieldName(StrEnum):
    """Header fields of the extraction schema. Line items are extracted separately."""

    INVOICE_NUMBER = "invoice_number"
    ISSUE_DATE = "issue_date"
    DUE_DATE = "due_date"
    SUPPLIER_NAME = "supplier_name"
    SUPPLIER_TRN = "supplier_trn"
    SUPPLIER_ADDRESS = "supplier_address"
    BUYER_NAME = "buyer_name"
    BUYER_TRN = "buyer_trn"
    CURRENCY = "currency"
    PO_NUMBER = "po_number"
    SUBTOTAL = "subtotal"
    VAT_TOTAL = "vat_total"
    TOTAL = "total"
    IBAN = "iban"


class InvoiceTypeCode(StrEnum):
    """BT-3 as the document states it (ADR-0012 §1): the title says invoice or credit note. The
    out-of-scope codes 480/81 are never extracted; `apx.einvoice` derives them from the line VAT
    categories (PINT AE ibr-151-ae)."""

    INVOICE = "380"
    CREDIT_NOTE = "381"


class ExtractorId(StrEnum):
    A = "a"
    B = "b"
    ARBITER = "arbiter"


class RawField(Frozen):
    """A field as the model emitted it. Deliberately has no box or word-id fields (I1)."""

    name: FieldName
    value: str | None


class RawLineItem(Frozen):
    """One line as the model emitted it. The first five values are copied as printed;
    `vat_category` (BT-151) is classified from the stated VAT treatment, None when unclear."""

    description: str | None
    quantity: str | None
    unit_price: str | None
    net_amount: str | None
    vat_rate: str | None
    vat_category: VatCategory | None = None


class FieldLogprobs(Frozen):
    """Token logprob summary over the value tokens of one field."""

    mean: float
    min: float


class ExtractorOutput(Frozen):
    extractor: ExtractorId
    model: str
    fields: tuple[RawField, ...]
    line_items: tuple[RawLineItem, ...]
    logprobs: Mapping[FieldName, FieldLogprobs]
    valid_json: bool
    invoice_type_code: InvoiceTypeCode | None = None
    """BT-3 as classified from the document title; None = the model could not tell."""
    finish_reason: str | None = None
    """Why decoding stopped as the serving layer reported it (`stop`, `length`, ...); None when
    the output did not come from a live call or the server sent none. Audited with the extract
    step so an invalid output's cause (for example a budget cut, `length`) is visible."""


class ExtractionResult(Frozen):
    """A, B and optional arbiter outputs plus grounded fields, input to the trust gate."""

    document_sha256: Sha256Hex
    a: ExtractorOutput
    b: ExtractorOutput
    arbiter: ExtractorOutput | None
    disputed: frozenset[FieldName]
    grounded: Mapping[FieldName, FieldValue]


class PromptBuilder(Protocol):
    """Builds extraction prompts. Returns StructuredRequest, which cannot carry tools (I2)."""

    def build(self, document: ParsedDocument, model: str) -> StructuredRequest: ...


class Extractor(Protocol):
    async def extract(self, document: ParsedDocument) -> ExtractorOutput:
        """Run one tool-free extraction. Raises ModelServingError; invalid JSON is reported via
        `valid_json=False`, never silently repaired."""
        ...


class Arbiter(Protocol):
    async def arbitrate(
        self,
        document: ParsedDocument,
        a: ExtractorOutput,
        b: ExtractorOutput,
        disputed: Sequence[FieldName],
    ) -> ExtractorOutput:
        """Re-extract only the disputed fields, tool-free."""
        ...


class Grounder(Protocol):
    def ground(
        self, document: ParsedDocument, fields: Sequence[RawField]
    ) -> dict[FieldName, FieldValue]:
        """Match each value to parse words (normalised, label-proximity aware) and take boxes from
        those words only (I1). Unmatched values get empty word_ids and boxes."""
        ...


LINE_ITEM_KEYS: tuple[str, ...] = tuple(RawLineItem.model_fields)
CLASSIFIED_LINE_ITEM_KEYS: tuple[str, ...] = ("vat_category",)
PRINTED_LINE_ITEM_KEYS: tuple[str, ...] = tuple(
    key for key in LINE_ITEM_KEYS if key not in CLASSIFIED_LINE_ITEM_KEYS
)
"""Line values copied as printed: the factory tags exactly these with `data-field`."""
INVOICE_TYPE_CODE_KEY = "invoice_type_code"
"""Top-level JSON key of the classified BT-3, between the header fields and `line_items`."""
MAX_LINE_ITEMS = 80
"""Upper bound in the decoding grammar, so a looping model cannot run to `max_tokens`."""


def extraction_json_schema() -> dict[str, JsonValue]:
    """JSON schema used for constrained decoding of ExtractorOutput fields and line items: every
    header field is a required string-or-null (null = not on the document), then the classified
    `invoice_type_code` (enum or null), then `line_items`, each ending in an enum-or-null
    `vat_category`. `line_items` stays last, so a masked FATURA suffix is one contiguous span."""
    nullable_string: dict[str, JsonValue] = {"type": ["string", "null"]}
    line_properties: dict[str, JsonValue] = {key: dict(nullable_string) for key in LINE_ITEM_KEYS}
    line_properties["vat_category"] = {"enum": [*(c.value for c in VatCategory), None]}
    line_item: dict[str, JsonValue] = {
        "type": "object",
        "properties": line_properties,
        "required": list(LINE_ITEM_KEYS),
        "additionalProperties": False,
    }
    properties: dict[str, JsonValue] = {name.value: dict(nullable_string) for name in FieldName}
    properties[INVOICE_TYPE_CODE_KEY] = {"enum": [*(c.value for c in InvoiceTypeCode), None]}
    properties["line_items"] = {"type": "array", "items": line_item, "maxItems": MAX_LINE_ITEMS}
    return {
        "type": "object",
        "properties": properties,
        "required": [*(name.value for name in FieldName), INVOICE_TYPE_CODE_KEY, "line_items"],
        "additionalProperties": False,
    }


def _comparable(name: FieldName, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        return normalize_field(name.value, value)
    except NormalizationError:
        return "raw:" + normalize_text(value)


def disputed_fields(a: ExtractorOutput, b: ExtractorOutput) -> frozenset[FieldName]:
    """Fields whose normalised A and B values differ; these go to the arbiter. A value that one
    side cannot normalise is compared as text, so it can only agree with the same text. A field
    missing from one output counts as null."""
    a_values = {field.name: field.value for field in a.fields}
    b_values = {field.name: field.value for field in b.fields}
    return frozenset(
        name
        for name in FieldName
        if _comparable(name, a_values.get(name)) != _comparable(name, b_values.get(name))
    )


class ResolvedClassification(Frozen):
    """Classified values the invoice builder may use: agreed by A and B, else None (review)."""

    invoice_type_code: InvoiceTypeCode | None
    line_vat_categories: tuple[VatCategory | None, ...]
    """One entry per line of the line source (A, or B when A's JSON is invalid)."""


def resolve_classification(a: ExtractorOutput, b: ExtractorOutput) -> ResolvedClassification:
    """ADR-0012 §1.4: a classified value is kept only when both extractors agree on it; a
    disagreement, a null on either side, or invalid JSON on either side gives None, which the
    validator turns into a review-routed violation (never a default). Categories follow the line
    source (A, else B). With unequal line counts lines keep a category only if every line on both
    sides carries that same single value, so swapping A and B never changes what is kept."""
    source = a if a.valid_json else b
    if not (a.valid_json and b.valid_json):
        return ResolvedClassification(
            invoice_type_code=None, line_vat_categories=(None,) * len(source.line_items)
        )
    type_code = a.invoice_type_code if a.invoice_type_code == b.invoice_type_code else None
    a_cats = [item.vat_category for item in a.line_items]
    b_cats = [item.vat_category for item in b.line_items]
    if len(a_cats) == len(b_cats):
        cats = tuple(x if x == y else None for x, y in zip(a_cats, b_cats, strict=True))
    else:
        both = set(a_cats) | set(b_cats)  # symmetric: neither side's line count decides
        single = next(iter(both)) if len(both) == 1 else None
        cats = (single,) * len(a_cats)
    return ResolvedClassification(invoice_type_code=type_code, line_vat_categories=cats)
