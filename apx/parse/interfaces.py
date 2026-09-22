"""Interfaces for parsing: pages, words, lines, tables and reading order with boxes. Parse words
are the sole source of field boxes (PRD §3, §6.2 I1)."""

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


class ParsedDocument(Frozen):
    document_sha256: Sha256Hex
    pages: tuple[Page, ...]
    reading_order: tuple[str, ...]
    """Line ids in reading order (RTL-aware for Arabic)."""
    languages: frozenset[Language]
    parser_version: str


class DocumentParser(Protocol):
    async def parse(self, document_sha256: Sha256Hex, data: bytes) -> ParsedDocument:
        """Parse a PDF or image via the PaddleOCR doc-parser service (pdfplumber text layer
        when present). Raises ParseError."""
        ...
