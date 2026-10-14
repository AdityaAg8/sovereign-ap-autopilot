"""Collects the text spans the injection classifier scores (P4): every
text line of the PDF's text layer, visible or hidden, plus the text the hidden-content detector
found in metadata, annotations and attachments. Text is cut down to the guard's validated domain:
NFKC, controls removed (flagged separately), and Arabic-script runs removed (ADR-0017 §5)."""

from __future__ import annotations

import io
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pdfplumber

from apx.common.types import BBox
from apx.scan.controls import SUSPICIOUS_CODEPOINTS
from apx.scan.interfaces import ScanFlag, ScanFlagKind

DOCUMENT_TEXT_KINDS: Final[frozenset[ScanFlagKind]] = frozenset(
    {ScanFlagKind.PDF_METADATA, ScanFlagKind.PDF_ATTACHMENT}
)
"""Detector flags whose evidence is document text a page's text layer does not contain (Info,
XMP, annotation contents, attachment names). Hidden page text is already in the page lines."""

_C0_CONTROLS = frozenset(range(0x00, 0x20)) - {0x09, 0x0A, 0x0D}
_STRIP = {cp: None for cp in SUSPICIOUS_CODEPOINTS | _C0_CONTROLS | {0x7F}}
_ESCAPE = re.compile(r"\\u[0-9a-f]{4}")
_SPACE = re.compile(r"\s+")
_ARABIC_BLOCKS: Final[tuple[tuple[int, int], ...]] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
)
_ARABIC_RUN = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _ARABIC_BLOCKS) + "]+")
"""Arabic script after NFKC (which folds the presentation forms into these blocks)."""


@dataclass(frozen=True, slots=True)
class TextSpan:
    """One unit of classifier input. `page` is 0-based; None for document-level text."""

    page: int | None
    text: str
    bbox: BBox | None = None


def clean_span_text(text: str) -> str:
    """NFKC, drop zero-width/bidi/C0 controls, collapse whitespace."""
    stripped = unicodedata.normalize("NFKC", text).translate(_STRIP)
    return _SPACE.sub(" ", stripped).strip()


def without_arabic(text: str) -> str:
    """`text` with its Arabic-script runs removed. The guard never saw the text layer's Arabic
    (visual-order glyphs, mixed with table numbers) in training or calibration, and on clean
    UAE invoices it scores those lines as injections (ADR-0017 §5), so only the Latin part of
    a line is scored. Arabic hidden text is still caught by the rule-based detector."""
    return _SPACE.sub(" ", _ARABIC_RUN.sub(" ", text)).strip()


def _unescape_evidence(text: str) -> str:
    """Scan evidence shows zero-width, bidi and C0 controls as literal `\\uXXXX` escapes
    (apx.scan.text); none is readable text, so drop them all, as the renderer does."""
    return _ESCAPE.sub("", text)


def _scorable(text: str) -> bool:
    """Spans without a single letter (amounts, dates, IBAN-only digits) carry no instruction."""
    return any(ch.isalpha() for ch in text)


def _line_bbox(line: dict[str, Any], page: int, width: float, height: float) -> BBox | None:
    if width <= 0 or height <= 0:
        return None

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    x0, x1 = clamp(float(line["x0"]) / width), clamp(float(line["x1"]) / width)
    y0, y1 = clamp(float(line["top"]) / height), clamp(float(line["bottom"]) / height)
    return BBox(page=page, x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))


def page_spans(pdf_bytes: bytes) -> list[TextSpan]:
    """Each text line of every page, one span per line: the unit the guard was trained on
    (`ml.guard.mine` mines single text-layer lines). Joining adjacent lines was measured to
    flag clean address/date concatenations (ADR-0017 §5). Raises whatever pdfplumber raises on
    an unreadable PDF; the caller wraps it."""
    spans: list[TextSpan] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
        for page_index, page in enumerate(doc.pages):
            width, height = float(page.width), float(page.height)
            for raw in page.extract_text_lines(strip=True, return_chars=False):
                text = without_arabic(clean_span_text(str(raw["text"])))
                if text:
                    spans.append(
                        TextSpan(page_index, text, _line_bbox(raw, page_index, width, height))
                    )
    return spans


def evidence_spans(flags: Iterable[ScanFlag]) -> list[TextSpan]:
    """Document text the detector surfaced as evidence (`DOCUMENT_TEXT_KINDS`). Evidence is an
    excerpt capped at `apx.scan.text.EXCERPT_MAX_CHARS`; a longer payload is scored by its head."""
    spans: list[TextSpan] = []
    for flag in flags:
        if flag.kind in DOCUMENT_TEXT_KINDS:
            text = without_arabic(clean_span_text(_unescape_evidence(flag.evidence.text)))
            if text:
                spans.append(TextSpan(flag.page, text, flag.bbox))
    return spans


def scorable_spans(spans: Sequence[TextSpan]) -> list[TextSpan]:
    """Spans with at least one letter, first occurrence of each (page, text) only."""
    seen: set[tuple[int | None, str]] = set()
    out: list[TextSpan] = []
    for span in spans:
        key = (span.page, span.text)
        if key in seen or not _scorable(span.text):
            continue
        seen.add(key)
        out.append(span)
    return out
