"""Interfaces for parsing: pages, words, lines, tables and reading order with boxes, plus the
small `ParsedDocumentRef` handle that stands in for a stored parse across Temporal activity
boundaries. Parse words are the sole source of field boxes (PRD §3, §6.2 I1)."""

from __future__ import annotations

from typing import Annotated, Protocol

from pydantic import Field

from apx.common.types import BBox, Frozen, Language, Sha256Hex


class Word(Frozen):
    id: str
    """Stable id `p<page>w<index>`; referenced by FieldValue.word_ids."""
    text: str
    bbox: BBox
    line_id: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class Line(Frozen):
    id: str
    word_ids: tuple[str, ...]
    bbox: BBox


class TableCell(Frozen):
    row: int
    col: int
    word_ids: tuple[str, ...]
    bbox: BBox


class Table(Frozen):
    id: str
    page: int
    cells: tuple[TableCell, ...]


class Page(Frozen):
    index: int
    width_px: int
    height_px: int
    words: tuple[Word, ...]
    lines: tuple[Line, ...]
    tables: tuple[Table, ...]
    image_png: bytes
    """Rendered page image used by VLM extractors and page similarity."""
    truncated: bool = False
    """The OCR service stopped before transcribing the whole page (decode limit or a repetition
    loop, ADR-0013 amendment 1), so `words` may miss text that is on the page. Always False for a
    born-digital page; False by default so parses stored before the flag existed still load."""


class ParsedDocument(Frozen):
    document_sha256: Sha256Hex
    pages: tuple[Page, ...]
    reading_order: tuple[str, ...]
    """Line ids in reading order (RTL-aware for Arabic)."""
    languages: frozenset[Language]
    parser_version: str

    @property
    def truncated_pages(self) -> tuple[int, ...]:
        """Indexes of the pages whose OCR transcription was cut short (`Page.truncated`)."""
        return tuple(page.index for page in self.pages if page.truncated)


class ParsedDocumentRef(Frozen):
    """JSON-safe, constant-size handle to a `ParsedDocument` held in a `ParsedDocumentStore`.
    Workflow activities exchange this instead of the document itself: page PNGs are raw bytes
    and a multi-page parse exceeds Temporal's ~2 MB payload limit."""

    document_sha256: Sha256Hex
    content_sha256: Sha256Hex
    """Digest of the stored parse (words, boxes and every page PNG's sha256); the store key."""
    parser_version: str
    page_count: Annotated[int, Field(ge=0)]


class ParsedDocumentStore(Protocol):
    """Content-addressed local store of parses, reachable by every worker process."""

    def put(self, document: ParsedDocument) -> ParsedDocumentRef:
        """Store `document` (idempotent: the same parse always yields the same ref)."""
        ...

    def load(self, ref: ParsedDocumentRef) -> ParsedDocument:
        """Raises ParsedArtifactMissingError when any part is absent, and
        ParsedArtifactIntegrityError when what is stored does not match `ref`."""
        ...


class DocumentParser(Protocol):
    async def parse(self, document_sha256: Sha256Hex, data: bytes) -> ParsedDocument:
        """Parse a PDF or image via the PaddleOCR doc-parser service (pdfplumber text layer
        when present). Raises ParseError."""
        ...
