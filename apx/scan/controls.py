"""Detects zero-width and bidi control characters hidden in PDF text (PRD §3, §6.3 A05/A06,
M6.2). Arabic-aware: ZWJ/ZWNJ inside an Arabic word and RLM next to a digit in Arabic-script
context are normal shaping and never flag; bidi overrides (RLO/LRO) always flag."""

from __future__ import annotations

from typing import Any

import pdfplumber.page

from apx.common.types import BBox
from apx.scan.interfaces import ScanFlag, ScanFlagKind
from apx.scan.text import build_evidence

PdfChar = dict[str, Any]
"""pdfplumber's per-character dict (`page.chars` items); pdfplumber ships no stub for its
runtime shape, so field access below is intentionally untyped at that boundary."""

ZERO_WIDTH_CODEPOINTS: frozenset[int] = frozenset(
    {
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER (ZWNJ)
        0x200D,  # ZERO WIDTH JOINER (ZWJ)
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)
"""Genuinely zero-advance-width characters that carry no visible glyph (A05)."""

BIDI_OVERRIDE_CODEPOINTS: frozenset[int] = frozenset(
    {
        0x202D,  # LEFT-TO-RIGHT OVERRIDE (LRO)
        0x202E,  # RIGHT-TO-LEFT OVERRIDE (RLO)
    }
)
"""Directional *overrides*: they force a display order unrelated to logical order and have no
legitimate use in an invoice. Always flagged, regardless of script context (A06)."""

BIDI_FORMAT_CODEPOINTS: frozenset[int] = frozenset(
    {
        0x200E,  # LEFT-TO-RIGHT MARK (LRM)
        0x200F,  # RIGHT-TO-LEFT MARK (RLM)
        0x202A,  # LEFT-TO-RIGHT EMBEDDING (LRE)
        0x202B,  # RIGHT-TO-LEFT EMBEDDING (RLE)
        0x202C,  # POP DIRECTIONAL FORMATTING (PDF)
        0x2066,  # LEFT-TO-RIGHT ISOLATE (LRI)
        0x2067,  # RIGHT-TO-LEFT ISOLATE (RLI)
        0x2068,  # FIRST STRONG ISOLATE (FSI)
        0x2069,  # POP DIRECTIONAL ISOLATE (PDI)
    }
    | BIDI_OVERRIDE_CODEPOINTS
)
"""All bidi formatting characters (embeddings, isolates, marks and overrides), U+200E-F,
U+202A-E, U+2066-9 (A06)."""

SUSPICIOUS_CODEPOINTS: frozenset[int] = ZERO_WIDTH_CODEPOINTS | BIDI_FORMAT_CODEPOINTS
"""Full set scanned in text and in PDF metadata (`apx.scan.active_content`)."""

_ARABIC_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

_ARABIC_CONTEXT_WINDOW = 8
"""How far either side of a bidi mark to look for Arabic script when deciding whether an
RLM/LRM next to a digit is normal Arabic-context number display."""


def is_arabic_char(ch: str) -> bool:
    """True if `ch` is a single Arabic-script code point (letters, incl. presentation forms)."""
    if not ch:
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_SCRIPT_RANGES)


def scan_page_controls(page: pdfplumber.page.Page, page_index: int) -> list[ScanFlag]:
    """Flag zero-width and bidi control characters in `page`'s text, grouped into reading-order
    lines from pdfplumber's char stream so shaping exemptions can see each character's
    neighbours. Never raises for a page with no suspicious characters (returns [])."""
    flags: list[ScanFlag] = []
    width, height = float(page.width), float(page.height)
    for line in _reading_order_lines(page):
        text = "".join(str(ch["text"]) for ch in line)
        for i, ch in enumerate(text):
            cp = ord(ch)
            if cp not in SUSPICIOUS_CODEPOINTS:
                continue
            if _is_exempt(cp, text, i):
                continue
            kind = (
                ScanFlagKind.ZERO_WIDTH
                if cp in ZERO_WIDTH_CODEPOINTS
                else ScanFlagKind.BIDI_CONTROL
            )
            char = line[i]
            bbox = _char_bbox(char, page_index, width, height)
            evidence = build_evidence(kind, _context_window(text, i))
            flags.append(
                ScanFlag(kind=kind, page=page_index, score=1.0, evidence=evidence, bbox=bbox)
            )
    return flags


def _is_exempt(cp: int, text: str, index: int) -> bool:
    if cp in BIDI_OVERRIDE_CODEPOINTS:
        return False  # RLO/LRO always flag (A06)
    if cp in (0x200C, 0x200D):  # ZWNJ, ZWJ: normal inside an Arabic word
        before = text[index - 1] if index > 0 else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        return is_arabic_char(before) or is_arabic_char(after)
    if cp in (0x200E, 0x200F):  # LRM, RLM: normal next to a digit in Arabic-script context
        before = text[index - 1] if index > 0 else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        next_to_digit = before.isdigit() or after.isdigit()
        if not next_to_digit:
            return False
        lo = max(0, index - _ARABIC_CONTEXT_WINDOW)
        hi = min(len(text), index + _ARABIC_CONTEXT_WINDOW + 1)
        return any(is_arabic_char(c) for c in text[lo:hi])
    return False


def _context_window(text: str, index: int, radius: int = 20) -> str:
    lo = max(0, index - radius)
    hi = min(len(text), index + radius + 1)
    return text[lo:hi]


def _reading_order_lines(page: pdfplumber.page.Page) -> list[list[PdfChar]]:
    """Group `page.chars` into lines (by rounded `top`) and sort each line by `x0`, recovering
    the order characters were drawn in -- the same order the test fixtures write strings in,
    regardless of script direction (reportlab does not reorder glyphs for bidi text)."""
    lines: dict[int, list[PdfChar]] = {}
    for ch in page.chars:
        key = round(float(ch["top"]))
        lines.setdefault(key, []).append(ch)
    return [sorted(chars, key=lambda c: float(c["x0"])) for _, chars in sorted(lines.items())]


def _char_bbox(char: PdfChar, page_index: int, width: float, height: float) -> BBox | None:
    if width <= 0 or height <= 0:
        return None
    x0 = max(0.0, min(1.0, float(char["x0"]) / width))
    x1 = max(0.0, min(1.0, float(char["x1"]) / width))
    y0 = max(0.0, min(1.0, float(char["top"]) / height))
    y1 = max(0.0, min(1.0, float(char["bottom"]) / height))
    return BBox(page=page_index, x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))
