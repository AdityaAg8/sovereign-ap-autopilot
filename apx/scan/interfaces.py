"""Interfaces for the hidden-content scan: invisible/hidden text, zero-width and bidi controls,
PDF metadata/attachments/JavaScript, and the ModernBERT injection classifier (PRD §3, §6.3 P4)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Protocol

from pydantic import Field

from apx.common.types import BBox, Frozen, Sha256Hex
from apx.common.untrusted import UntrustedText

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class ScanFlagKind(StrEnum):
    HIDDEN_TEXT = "hidden_text"
    INVISIBLE_TEXT = "invisible_text"
    ZERO_WIDTH = "zero_width"
    BIDI_CONTROL = "bidi_control"
    PDF_METADATA = "pdf_metadata"
    PDF_ATTACHMENT = "pdf_attachment"
    PDF_JAVASCRIPT = "pdf_javascript"
    INJECTION = "injection"


class ScanFlag(Frozen):
    kind: ScanFlagKind
    page: int | None
    score: Score
    evidence: UntrustedText
    bbox: BBox | None = None
    """Location of the offending content, page-normalised (M6.2). None for document-level
    findings that have no single position (metadata, attachments, JavaScript/OpenAction/AA)."""


class ScanReport(Frozen):
    """Scan outcome. `injection_score` feeds P4 review/block thresholds in apx.policy."""

    document_sha256: Sha256Hex
    flags: tuple[ScanFlag, ...]
    injection_score: Score


class HiddenContentDetector(Protocol):
    def detect(self, pdf_bytes: bytes) -> list[ScanFlag]:
        """Find hidden/invisible text, control characters and active PDF content.
        Raises ScanError; never returns an empty list on failure."""
        ...


class InjectionClassifier(Protocol):
    def score(self, texts: Sequence[str]) -> list[float]:
        """Probability of prompt injection per text, in [0, 1]."""
        ...


class Scanner(Protocol):
    def scan(self, document_sha256: Sha256Hex, pdf_bytes: bytes) -> ScanReport:
        """Run all detectors and the classifier. Raises ScanError."""
        ...
