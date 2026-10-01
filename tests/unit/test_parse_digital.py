"""Tests for `apx.parse.digital`: the born-digital pdfplumber fast path that the ≥98% IoU gate
(PRD §10 M3.1) runs against -- box normalization, line clustering, RTL word ordering and table
detection. Fast, no GPU, no network; PDFs are generated in-memory with reportlab."""

from __future__ import annotations

import io
from pathlib import Path

import pdfplumber
import pytest
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from apx.common.errors import ParseError
from apx.parse.digital import (
    arabic_fraction,
    has_extractable_text,
    is_arabic_char,
    is_rtl_text,
    parse_digital_page,
    parse_digital_pdf,
)

_PAGE_W, _PAGE_H = 400.0, 300.0


def _make_pdf(draw: object) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(_PAGE_W, _PAGE_H))
    draw(c)  # type: ignore[operator]
    c.save()
    return buf.getvalue()


def _one_word_pdf(text: str, x: float, y: float) -> bytes:
    def draw(c: canvas.Canvas) -> None:
        c.setFont("Helvetica", 12)
        c.drawString(x, y, text)

    return _make_pdf(draw)


def _blank_png(width: int = 100, height: int = 100) -> bytes:
    """An all-white raster with no ink, so `_refine_line_vertical_extent` finds nothing and
    falls back to pdfplumber's own top/bottom -- used by `_FakePlumberPage` tests below, which
    exercise word ordering, not ink-based box refinement."""
    buf = io.BytesIO()
    Image.new("L", (width, height), color=255).save(buf, format="PNG")
    return buf.getvalue()


_ALEF = chr(0x0627)  # ARABIC LETTER ALEF
_BEH = chr(0x0628)  # ARABIC LETTER BEH
_TEH_MARBUTA = chr(0x0629)  # ARABIC LETTER TEH MARBUTA
_ARABIC_ABC = _ALEF + _BEH + _TEH_MARBUTA
# Written as `chr()` code points rather than raw glyphs so this source file stays plain ASCII
# (readable in any editor/terminal encoding), matching apx/extract/normalize.py's convention.


def test_is_arabic_char_and_fraction() -> None:
    assert is_arabic_char(_ALEF)
    assert not is_arabic_char("a")
    assert arabic_fraction("hello") == 0.0
    assert arabic_fraction(_ARABIC_ABC) == 1.0
    assert 0.0 < arabic_fraction("abc" + _ALEF) < 1.0


def test_is_rtl_text_threshold() -> None:
    assert is_rtl_text(_ARABIC_ABC + " name")  # majority Arabic letters
    assert not is_rtl_text("Yas Valley Trading LLC")


def test_has_extractable_text_true_for_text_page() -> None:
    pdf_bytes = _one_word_pdf("INV-2026-000001", 50, 250)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        assert has_extractable_text(pdf.pages[0], min_words=1)


def test_has_extractable_text_false_for_blank_page() -> None:
    # A page with graphical content but no text (a stand-in for a scanned page's rasterized
    # image, which pdfplumber's `extract_words` never finds text in either).
    blank = _make_pdf(lambda c: c.rect(10, 10, 50, 50))
    with pdfplumber.open(io.BytesIO(blank)) as pdf:
        assert not has_extractable_text(pdf.pages[0])


def test_parse_digital_pdf_normalizes_box_and_renders_image() -> None:
    # reportlab's y origin is bottom-left; place a word near the top-left of the page.
    text_x, text_top_from_bottom = 40.0, 260.0
    pdf_bytes = _one_word_pdf("Invoice", text_x, text_top_from_bottom)

    pages = parse_digital_pdf("a" * 64, pdf_bytes)

    assert len(pages) == 1
    page = pages[0]
    assert len(page.words) == 1
    word = page.words[0]
    assert word.text == "Invoice"
    assert word.confidence == 1.0
    # Fully inside the page, normalized coordinates in [0, 1].
    assert 0.0 <= word.bbox.x0 < word.bbox.x1 <= 1.0
    assert 0.0 <= word.bbox.y0 < word.bbox.y1 <= 1.0
    # Near the left edge and near the top (reportlab y=260 of 300 is close to the top).
    assert word.bbox.x0 < 0.3
    assert word.bbox.y0 < 0.2

    assert len(page.lines) == 1
    assert page.lines[0].word_ids == (word.id,)
    assert page.image_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_parse_digital_pdf_raises_on_garbage_bytes() -> None:
    with pytest.raises(ParseError):
        parse_digital_pdf("a" * 64, b"not a pdf at all")


def test_parse_digital_page_matches_single_page_of_multi_page_pdf() -> None:
    def draw(c: canvas.Canvas) -> None:
        c.setFont("Helvetica", 12)
        c.drawString(40, 260, "PageOne")
        c.showPage()
        c.setFont("Helvetica", 12)
        c.drawString(40, 260, "PageTwo")

    pdf_bytes = _make_pdf(draw)
    all_pages = parse_digital_pdf("a" * 64, pdf_bytes)
    assert len(all_pages) == 2

    single = parse_digital_page("a" * 64, pdf_bytes, 1)
    assert single.words[0].text == "PageTwo"
    assert single.index == 1


def test_words_on_same_line_share_ink_refined_vertical_extent() -> None:
    # Two words on the same baseline: after ink refinement (apx.parse.digital's fix for
    # pdfminer font-metric top/bottom not matching some embedded fonts' actual glyph ink,
    # reports/parse.md) they must share one vertical band, not each keep its own tight box.
    def draw(c: canvas.Canvas) -> None:
        c.setFont("Helvetica", 14)
        c.drawString(40, 260, "100")  # digits only: naturally the tightest possible glyph box
        c.drawString(150, 260, "Tall")  # has an ascender, taller natural glyph box

    pdf_bytes = _make_pdf(draw)
    page = parse_digital_pdf("a" * 64, pdf_bytes)[0]

    assert len(page.lines) == 1
    words_by_text = {w.text: w for w in page.words}
    digits, tall = words_by_text["100"], words_by_text["Tall"]
    assert (digits.bbox.y0, digits.bbox.y1) == (tall.bbox.y0, tall.bbox.y1)
    # Sanity: the shared band still lands where this text actually is on the page, not
    # somewhere else entirely.
    assert 0.0 < digits.bbox.y0 < digits.bbox.y1 < 0.3


def test_line_clustering_groups_same_row_and_splits_different_rows() -> None:
    def draw(c: canvas.Canvas) -> None:
        c.setFont("Helvetica", 12)
        c.drawString(40, 260, "Left")
        c.drawString(200, 260, "Right")  # same baseline -> one line
        c.drawString(40, 200, "BelowLine")  # different baseline -> separate line

    pdf_bytes = _make_pdf(draw)
    page = parse_digital_pdf("a" * 64, pdf_bytes)[0]

    assert len(page.lines) == 2
    texts_by_line = [[w.text for w in page.words if w.id in line.word_ids] for line in page.lines]
    assert ["Left", "Right"] in texts_by_line
    assert ["BelowLine"] in texts_by_line


class _FakePlumberPage:
    """Duck-typed stand-in for `pdfplumber.page.Page`: exercises `_parse_digital_page`'s RTL
    word-reordering without needing a real Arabic-shaped PDF (reportlab's default fonts have no
    Arabic glyphs to lay out). Only the attributes/methods `apx.parse.digital` reads are here."""

    page_number = 1
    width = _PAGE_W
    height = _PAGE_H

    def __init__(self, words: list[dict[str, object]]) -> None:
        self._words = words

    def extract_words(self, **_kwargs: object) -> list[dict[str, object]]:
        return self._words

    def find_tables(self) -> list[object]:
        return []


def test_rtl_line_reorders_words_right_to_left() -> None:
    # Bare Arabic letters, no shaping needed for this test (only box positions matter).
    alef_word, beh_word = _ALEF * 3, _BEH * 3
    # Two "words" on the same visual line; rightmost word (higher x0) is logically first in RTL.
    words = [
        {"text": alef_word, "x0": 10.0, "x1": 30.0, "top": 50.0, "bottom": 62.0},
        {"text": beh_word, "x0": 100.0, "x1": 130.0, "top": 50.0, "bottom": 62.0},
    ]
    from apx.parse.digital import _parse_digital_page

    result = _parse_digital_page("a" * 64, 0, _FakePlumberPage(words), _blank_png())
    assert len(result.lines) == 1
    ordered_texts = [w.text for w in result.words]
    assert ordered_texts == [beh_word, alef_word]  # rightmost (x0=100) comes first


def test_ltr_line_keeps_left_to_right_order() -> None:
    words = [
        {"text": "second", "x0": 100.0, "x1": 140.0, "top": 50.0, "bottom": 62.0},
        {"text": "first", "x0": 10.0, "x1": 40.0, "top": 50.0, "bottom": 62.0},
    ]
    from apx.parse.digital import _parse_digital_page

    result = _parse_digital_page("a" * 64, 0, _FakePlumberPage(words), _blank_png())
    assert [w.text for w in result.words] == ["first", "second"]


_NASKH_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "fonts" / "NotoNaskhArabic-Regular.ttf"
)
_NASKH = "ParseTestNaskh"


def _visual_arabic_pdf(words_right_to_left: list[str], font_size: float = 12.0) -> bytes:
    """A born-digital RTL line as a shaping renderer (Chromium, the factory's) writes it: each
    word's glyphs drawn left to right in visual order as Arabic presentation forms, words laid
    out from the right margin leftwards. reportlab draws the given glyph string verbatim, so this
    reproduces the text layer pdfplumber reads from real AR-first invoices."""
    if _NASKH not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_NASKH, str(_NASKH_PATH)))

    def draw(c: canvas.Canvas) -> None:
        c.setFont(_NASKH, font_size)
        right = _PAGE_W - 20.0
        for word in words_right_to_left:
            width = pdfmetrics.stringWidth(word, _NASKH, font_size)
            c.drawString(right - width, 150, word)
            right -= width + 0.4 * font_size  # a word space above pdfplumber's x_tolerance

    return _make_pdf(draw)


def test_arabic_line_parses_to_logical_text_with_word_boxes() -> None:
    # Regression (e2e doc 41f5771c): the text layer holds the Arabic IBAN sentence in visual
    # order with presentation forms, so the LLM saw reversed words and returned no IBAN.
    ila, raqm = "ﯽﻟإ", "ﻢقر"
    al_iban = "نﺎبيآﻟا"
    # Shadda drawn after its left neighbour, so its zero-width origin lands exactly on the next
    # glyph's x0 -- it must re-attach to that glyph (noon), as in the factory PDFs.
    musanni = "ﻊّنﺼﻣُ"
    iban = "AE500335205997283792984"
    pdf_bytes = _visual_arabic_pdf([ila, raqm, al_iban, "." + iban, musanni])

    page = parse_digital_pdf("a" * 64, pdf_bytes)[0]

    assert len(page.lines) == 1
    texts = [w.text for w in page.words]
    assert texts == [
        "إلى",
        "رقم",
        "الآيبان",
        iban + ".",
        "مُصنّع",
    ]
    # Boxes still belong to their own words: logical order runs right to left on the page.
    x0s = [w.bbox.x0 for w in page.words]
    assert x0s == sorted(x0s, reverse=True)
    assert page.lines[0].word_ids == tuple(w.id for w in page.words)
