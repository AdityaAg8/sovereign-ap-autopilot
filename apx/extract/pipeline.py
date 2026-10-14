"""Dual extraction: Extractors A and B read the same document concurrently, the
fields they disagree on go to the arbiter when one is configured, and the final values are
grounded onto parse words. Satisfies `apx.workflows.ports.ExtractionPipeline`."""

from __future__ import annotations

import asyncio

from apx.extract.interfaces import (
    Arbiter,
    ExtractionResult,
    Extractor,
    ExtractorOutput,
    FieldName,
    Grounder,
    RawField,
    disputed_fields,
)
from apx.parse.interfaces import ParsedDocument


def final_fields(
    a: ExtractorOutput,
    b: ExtractorOutput,
    arbiter: ExtractorOutput | None,
    disputed: frozenset[FieldName],
) -> tuple[RawField, ...]:
    """A's value for every field, except disputed fields a valid arbiter output rules on. With no
    arbiter a disputed field keeps A's value; the trust gate's agreement feature still sees the
    dispute, so that field is never auto-accepted on A's word alone. When A's JSON was invalid,
    B's fields stand in; hard-reject rule H1 then keeps every field out of auto-accept."""
    base = a if a.valid_json or not b.valid_json else b
    ruled = (
        {f.name: f for f in arbiter.fields if f.name in disputed}
        if arbiter is not None and arbiter.valid_json
        else {}
    )
    return tuple(ruled.get(field.name, field) for field in base.fields)


class DualExtractionPipeline:
    """A and B run concurrently (they share the `extract` llama-swap group, so both stay
    resident); the arbiter only runs when there is a dispute to settle."""

    def __init__(
        self,
        extractor_a: Extractor,
        extractor_b: Extractor,
        grounder: Grounder,
        arbiter: Arbiter | None = None,
    ) -> None:
        self._a = extractor_a
        self._b = extractor_b
        self._grounder = grounder
        self._arbiter = arbiter

    async def run(self, document: ParsedDocument) -> ExtractionResult:
        a, b = await asyncio.gather(self._a.extract(document), self._b.extract(document))
        disputed = disputed_fields(a, b)
        arbiter = None
        if self._arbiter is not None and disputed:
            arbiter = await self._arbiter.arbitrate(document, a, b, sorted(disputed))
        grounded = self._grounder.ground(document, final_fields(a, b, arbiter, disputed))
        return ExtractionResult(
            document_sha256=document.document_sha256,
            a=a,
            b=b,
            arbiter=arbiter,
            disputed=disputed,
            grounded=grounded,
        )
