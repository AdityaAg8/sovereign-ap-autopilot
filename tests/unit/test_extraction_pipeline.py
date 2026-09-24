"""Tests for `apx.extract.pipeline` (PRD §10 M5.6): A and B both run, disputes go to the arbiter
only when there is one and something to settle, the arbiter rules only on disputed fields, and an
invalid A falls back to B's values."""

from __future__ import annotations

from collections.abc import Sequence

from apx.common.types import FieldValue, Language, Sha256Hex
from apx.extract.interfaces import ExtractorId, ExtractorOutput, FieldName, RawField
from apx.extract.pipeline import DualExtractionPipeline, final_fields
from apx.parse.interfaces import ParsedDocument

DOC = ParsedDocument(
    document_sha256=Sha256Hex("a" * 64),
    pages=(),
    reading_order=(),
    languages=frozenset({Language.EN}),
    parser_version="test-1",
)


def _output(
    which: ExtractorId, values: dict[FieldName, str | None], *, valid: bool = True
) -> ExtractorOutput:
    return ExtractorOutput(
        extractor=which,
        model=f"model-{which.value}",
        fields=tuple(RawField(name=n, value=v) for n, v in values.items()),
        line_items=(),
        logprobs={},
        valid_json=valid,
    )


class FixedExtractor:
    def __init__(self, output: ExtractorOutput) -> None:
        self.output = output
        self.calls = 0

    async def extract(self, document: ParsedDocument) -> ExtractorOutput:
        self.calls += 1
        return self.output


class RecordingArbiter:
    def __init__(self, output: ExtractorOutput) -> None:
        self.output = output
        self.disputed: list[Sequence[FieldName]] = []

    async def arbitrate(
        self,
        document: ParsedDocument,
        a: ExtractorOutput,
        b: ExtractorOutput,
        disputed: Sequence[FieldName],
    ) -> ExtractorOutput:
        self.disputed.append(disputed)
        return self.output


class EchoGrounder:
    def ground(
        self, document: ParsedDocument, fields: Sequence[RawField]
    ) -> dict[FieldName, FieldValue]:
        return {
            f.name: FieldValue(name=f.name.value, raw=f.value, normalized=f.value) for f in fields
        }


A = _output(ExtractorId.A, {FieldName.INVOICE_NUMBER: "INV-1", FieldName.TOTAL: "105.00"})
B_AGREES = _output(ExtractorId.B, {FieldName.INVOICE_NUMBER: "INV-1", FieldName.TOTAL: "105.00"})
B_DISAGREES = _output(ExtractorId.B, {FieldName.INVOICE_NUMBER: "INV-1", FieldName.TOTAL: "150.00"})


async def test_agreement_skips_the_arbiter_and_grounds_a() -> None:
    arbiter = RecordingArbiter(_output(ExtractorId.ARBITER, {}))
    pipeline = DualExtractionPipeline(
        FixedExtractor(A), FixedExtractor(B_AGREES), EchoGrounder(), arbiter
    )
    result = await pipeline.run(DOC)
    assert result.disputed == frozenset()
    assert result.arbiter is None
    assert arbiter.disputed == []
    assert result.grounded[FieldName.TOTAL].raw == "105.00"
    assert result.document_sha256 == DOC.document_sha256


async def test_arbiter_rules_only_on_disputed_fields() -> None:
    arbiter = RecordingArbiter(
        _output(
            ExtractorId.ARBITER,
            {FieldName.TOTAL: "150.00", FieldName.INVOICE_NUMBER: "HIJACKED"},
        )
    )
    pipeline = DualExtractionPipeline(
        FixedExtractor(A), FixedExtractor(B_DISAGREES), EchoGrounder(), arbiter
    )
    result = await pipeline.run(DOC)
    assert result.disputed == frozenset({FieldName.TOTAL})
    assert list(arbiter.disputed[0]) == [FieldName.TOTAL]
    assert result.grounded[FieldName.TOTAL].raw == "150.00"
    assert result.grounded[FieldName.INVOICE_NUMBER].raw == "INV-1"


async def test_without_an_arbiter_a_disputed_field_keeps_a_value() -> None:
    pipeline = DualExtractionPipeline(
        FixedExtractor(A), FixedExtractor(B_DISAGREES), EchoGrounder()
    )
    result = await pipeline.run(DOC)
    assert result.disputed == frozenset({FieldName.TOTAL})
    assert result.arbiter is None
    assert result.grounded[FieldName.TOTAL].raw == "105.00"


def test_invalid_a_falls_back_to_b_and_invalid_arbiter_is_ignored() -> None:
    bad_a = _output(ExtractorId.A, {}, valid=False)
    bad_arbiter = _output(ExtractorId.ARBITER, {FieldName.TOTAL: "999.00"}, valid=False)
    chosen = final_fields(bad_a, B_DISAGREES, bad_arbiter, frozenset({FieldName.TOTAL}))
    assert {f.name: f.value for f in chosen} == {
        FieldName.INVOICE_NUMBER: "INV-1",
        FieldName.TOTAL: "150.00",
    }
