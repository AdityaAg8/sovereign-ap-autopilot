"""Tests for `apx.parse.bidi`: recovering logical reading order and base Arabic letters from a PDF
line whose glyphs come back in visual order as presentation forms. The visual strings below are
copied from a factory AR-first invoice's pdfplumber output (doc 41f5771c, whose IBAN the extractor
missed), written as escapes so this file stays plain ASCII."""

from __future__ import annotations

from apx.parse.bidi import VisualWord, fold_presentation_forms, logical_line

# Logical forms, for the expected values.
ILA = "إلى"  # "to"
RAQM = "رقم"  # "number"
AL_IBAN = "الآيبان"  # "the IBAN"
TAMMA = "تم"  # "was"
SHARIKA = "شركة"  # "company"
MUSANNI = "مُصنّع"  # "generator", with damma and shadda

# Visual (left-to-right glyph) forms exactly as the PDF text layer yields them.
V_ILA = "ﯽﻟإ"
V_RAQM = "ﻢقر"
V_AL_IBAN = "نﺎبيآﻟا"
V_TAMMA = "ﻢت"
V_SHARIKA = "ﺔﻛﺮﺷ"
IBAN = "AE500335205997283792984"


def _laid_out(texts: list[str], *, gap: float = 3.0, width: float = 20.0) -> list[VisualWord]:
    """Words placed left to right with ordinary word spacing (one segment)."""
    return [
        VisualWord(text, i * (width + gap), i * (width + gap) + width)
        for i, text in enumerate(texts)
    ]


def _read(words: list[VisualWord], *, rtl: bool) -> str:
    return " ".join(text for _, text in logical_line(words, rtl=rtl, line_height=10.0))


def test_iban_sentence_reads_in_logical_order() -> None:
    # Visual left-to-right: "was", ".IBAN", "the-IBAN", "number", "to" (RTL paragraph).
    visual = _laid_out([V_TAMMA, "." + IBAN, V_AL_IBAN, V_RAQM, V_ILA])
    assert _read(visual, rtl=True) == f"{ILA} {RAQM} {AL_IBAN} {IBAN}. {TAMMA}"


def test_each_word_keeps_its_own_index_for_box_grounding() -> None:
    visual = _laid_out([V_TAMMA, "." + IBAN, V_AL_IBAN])
    result = logical_line(visual, rtl=True, line_height=10.0)
    assert sorted(i for i, _ in result) == [0, 1, 2]
    assert dict(result)[1] == IBAN + "."


def test_numbers_and_dates_stay_intact_in_rtl_text() -> None:
    visual = _laid_out(["2026-07-30", V_RAQM, "1,027.04", "5%", V_ILA])
    assert _read(visual, rtl=True) == f"{ILA} 5% 1,027.04 {RAQM} 2026-07-30"


def test_number_only_ltr_line_keeps_left_to_right_order() -> None:
    assert _read(_laid_out(["23", "1,027.04", "5%", "23,621.92"]), rtl=False) == (
        "23 1,027.04 5% 23,621.92"
    )


def test_latin_phrase_inside_rtl_line_keeps_left_to_right_order() -> None:
    visual = _laid_out(["Arab", "Emirates,", "P.O.", "Box", "59903", V_RAQM])
    assert _read(visual, rtl=True) == f"{RAQM} Arab Emirates, P.O. Box 59903"


def test_arabic_word_inside_ltr_line_is_unreversed_in_place() -> None:
    visual = _laid_out(["Supplier:", V_SHARIKA])
    assert _read(visual, rtl=False) == f"Supplier: {SHARIKA}"


def test_wide_gaps_split_table_cells_read_right_to_left() -> None:
    # One RTL table row: description cell (Arabic + a model code) at the right, then quantity,
    # then the line total at the left, each separated by a column gap.
    words = [
        VisualWord("7,739.22", 0.0, 30.0),
        VisualWord("6", 100.0, 105.0),
        VisualWord("EN397", 200.0, 230.0),
        VisualWord(V_SHARIKA, 233.0, 260.0),
    ]
    assert _read(words, rtl=True) == f"{SHARIKA} EN397 6 7,739.22"


def test_combining_marks_stay_on_their_base_letter() -> None:
    # Visual order with each mark directly after the base glyph it is drawn on.
    visual_musanni = "ﻊنّﺼﻣُ"
    assert _read(_laid_out([visual_musanni]), rtl=True) == MUSANNI


def test_fold_presentation_forms() -> None:
    lam_alef = "ﻻ"  # LAM WITH ALEF ISOLATED FORM
    assert fold_presentation_forms(lam_alef) == "لا"
    assert fold_presentation_forms("ﯽ") == "ى"  # dotless final yeh -> alef maksura
    assert fold_presentation_forms("ﻢ") == "م"
    # Only Arabic presentation blocks fold; Latin compatibility characters are left alone.
    assert fold_presentation_forms("m² ﬁ") == "m² ﬁ"
