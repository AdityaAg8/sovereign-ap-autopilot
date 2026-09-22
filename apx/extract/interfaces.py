"""Interfaces for extraction: the invoice field schema, tool-free Extractor A/B and arbiter
clients, and grounding of values onto parse words. Model output carries no boxes; boxes are
attached only by the Grounder (PRD §3, §4, §6.2 I1/I2, ADR-0003)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import JsonValue

from apx.common.types import FieldValue, Frozen, Sha256Hex
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


class ExtractorId(StrEnum):
    A = "a"
    B = "b"
    ARBITER = "arbiter"


class RawField(Frozen):
    """A field as the model emitted it. Deliberately has no box or word-id fields (I1)."""

    name: FieldName
    value: str | None


class RawLineItem(Frozen):
    description: str | None
    quantity: str | None
    unit_price: str | None
    net_amount: str | None
    vat_rate: str | None


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


def extraction_json_schema() -> dict[str, JsonValue]:
    """JSON schema used for constrained decoding of ExtractorOutput fields and line items."""
    raise NotImplementedError


def disputed_fields(a: ExtractorOutput, b: ExtractorOutput) -> frozenset[FieldName]:
    """Fields whose normalised A and B values differ; these go to the arbiter."""
    raise NotImplementedError
