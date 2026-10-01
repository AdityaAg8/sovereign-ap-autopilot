"""Visual-to-logical reordering of one PDF text line for the born-digital parse path (PRD §3,
§10 M3.1). A PDF text layer draws shaped Arabic glyphs left to right in visual order, so
pdfplumber returns each Arabic word's letters reversed, as presentation forms; this module undoes
the Unicode Bidirectional Algorithm's reordering (UAX #9 rules W4-W7, N1-N2, I1-I2, L2) per word,
so `apx.parse.digital` can emit logical text while every word keeps its own box (I1, PRD §6.2)."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

_PRESENTATION_BLOCKS: Final[tuple[tuple[int, int], ...]] = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
_FINAL_YEH_AS_ALEF_MAKSURA: Final[dict[str, str]] = {"ﯼ": "ى", "ﯽ": "ى"}
"""Farsi yeh's isolated/final forms are dotless and share one glyph with alef maksura (U+0649)
in Arabic fonts, so a font's ToUnicode map may name either; in UAE Arabic text it is alef
maksura (e.g. "إلى"), so these fold there instead of NFKC's U+06CC."""

_SEGMENT_GAP_EM: Final[float] = 0.8
"""A horizontal gap wider than this many line-heights between two words splits the line into
independent segments (table cells, label/value columns): each is reordered on its own, since
the page placed them by position, not as one bidi paragraph."""

_Kind = Literal["L", "R", "NUM", "SEP", "TERM", "N"]


@dataclass(frozen=True, slots=True)
class VisualWord:
    """One pdfplumber word as drawn: `text` in visual (left-to-right glyph) order, with the
    word's horizontal extent in PDF points."""

    text: str
    x0: float
    x1: float


@dataclass(slots=True)
class _Unit:
    """A base character plus any combining marks drawn on it, tagged with its source word
    (-1 for the synthetic space between two words) and its resolved embedding level."""

    text: str
    word: int
    kind: _Kind
    level: int = 0


def fold_presentation_forms(text: str) -> str:
    """Arabic presentation forms (U+FB50-FDFF, U+FE70-FEFF) to base letters via NFKC; other
    characters are left untouched, so Latin values (e.g. "m²") never change. Lam-alef ligatures
    expand to lam + alef in logical order, so call this on text already in logical order."""
    out: list[str] = []
    for ch in text:
        folded = _FINAL_YEH_AS_ALEF_MAKSURA.get(ch)
        if folded is None and any(lo <= ord(ch) <= hi for lo, hi in _PRESENTATION_BLOCKS):
            nfkc = unicodedata.normalize("NFKC", ch)
            folded = ch if " " in nfkc else nfkc  # isolated-harakat forms NFKC to " " + mark
        out.append(ch if folded is None else folded)
    return "".join(out)


def logical_line(
    words: Sequence[VisualWord], *, rtl: bool, line_height: float
) -> list[tuple[int, str]]:
    """Reorder one visual line into logical reading order. Returns `(index into words, logical
    text)` for every word, in reading order. `rtl` is the line's paragraph direction; words
    separated by more than `_SEGMENT_GAP_EM * line_height` are reordered as separate segments,
    taken right to left on an RTL line and left to right otherwise."""
    visual = sorted(range(len(words)), key=lambda i: words[i].x0)
    segments: list[list[int]] = []
    for i in visual:
        gap_limit = _SEGMENT_GAP_EM * line_height
        if segments and words[i].x0 - words[segments[-1][-1]].x1 <= gap_limit:
            segments[-1].append(i)
        else:
            segments.append([i])
    if rtl:
        segments.reverse()

    result: list[tuple[int, str]] = []
    for segment in segments:
        result.extend(_logical_segment(words, segment, rtl=rtl))
    return result


def _logical_segment(
    words: Sequence[VisualWord], segment: list[int], *, rtl: bool
) -> list[tuple[int, str]]:
    units: list[_Unit] = []
    for n, word_index in enumerate(segment):
        if n:
            units.append(_Unit(" ", -1, "N"))
        units.extend(_units(words[word_index].text, word_index))

    base = 1 if rtl else 0
    _resolve_levels(units, base)
    for level in range(1, max((u.level for u in units), default=0) + 1):
        _reverse_runs(units, level)  # L2 applies high->low; its inverse applies low->high

    texts: dict[int, list[str]] = {i: [] for i in segment}
    first_seen: dict[int, int] = {}
    for position, unit in enumerate(units):
        if unit.word >= 0:
            texts[unit.word].append(unit.text)
            first_seen.setdefault(unit.word, position)
    ordered = sorted(segment, key=lambda i: first_seen.get(i, 0))
    return [(i, fold_presentation_forms("".join(texts[i]))) for i in ordered]


def _units(text: str, word: int) -> list[_Unit]:
    """Split visual text into base+mark clusters. A mark follows its base glyph in visual order
    (it is drawn over it), so it stays glued to that base when an RTL run is reversed."""
    units: list[_Unit] = []
    for ch in text:
        if unicodedata.bidirectional(ch) == "NSM" and units:
            units[-1].text += ch
        else:
            units.append(_Unit(ch, word, _kind(ch)))
    return units


def _kind(ch: str) -> _Kind:
    bidi = unicodedata.bidirectional(ch)
    if bidi in ("R", "AL"):
        return "R"
    if bidi == "L":
        return "L"
    if bidi in ("EN", "AN"):
        return "NUM"
    if bidi in ("CS", "ES"):
        return "SEP"
    if bidi == "ET":
        return "TERM"
    return "N"


def _resolve_levels(units: list[_Unit], base: int) -> None:
    kinds: list[_Kind] = [u.kind for u in units]
    # W4: a single separator between two digits is part of the number ("1,027.04", "2026-07-30").
    for i in range(1, len(kinds) - 1):
        if kinds[i] == "SEP" and kinds[i - 1] == "NUM" and kinds[i + 1] == "NUM":
            kinds[i] = "NUM"
    # W5: terminators ("%", currency signs) adjacent to a number join it.
    changed = True
    while changed:
        changed = False
        for i, kind in enumerate(kinds):
            if kind == "TERM" and (
                (i > 0 and kinds[i - 1] == "NUM") or (i + 1 < len(kinds) and kinds[i + 1] == "NUM")
            ):
                kinds[i] = "NUM"
                changed = True

    strong: list[str | None] = ["L" if k == "L" else "R" if k == "R" else None for k in kinds]
    # W7: a number next to Latin text belongs to it ("Box 59903", "INV-2026"); otherwise it
    # counts as R for neutral resolution (N1).
    # With no strong letter on a side, the paragraph edge (sos/eos) stands in for it.
    edge = "R" if base else "L"
    direction: list[str | None] = list(strong)
    for i, kind in enumerate(kinds):
        if kind == "NUM":
            left = _nearest(strong, i, -1) or edge
            right = _nearest(strong, i, 1) or edge
            direction[i] = "L" if "L" in (left, right) else "R"
    before_neutrals = list(direction)
    for i in range(len(kinds)):
        if before_neutrals[i] is None:  # N1/N2: a neutral takes its neighbours' shared direction
            left = _nearest(before_neutrals, i, -1) or edge
            right = _nearest(before_neutrals, i, 1) or edge
            direction[i] = left if left == right else edge

    for unit, kind, resolved in zip(units, kinds, direction, strict=True):
        if kind == "NUM":  # I1/I2: numbers are always LTR inside their run
            unit.level = 2 if (base or resolved == "R") else 0
        elif resolved == "R":
            unit.level = 1
        else:
            unit.level = 2 if base else 0


def _nearest(values: list[str | None], start: int, step: int) -> str | None:
    i = start + step
    while 0 <= i < len(values):
        if values[i] is not None:
            return values[i]
        i += step
    return None


def _reverse_runs(units: list[_Unit], level: int) -> None:
    i = 0
    while i < len(units):
        if units[i].level < level:
            i += 1
            continue
        j = i
        while j < len(units) and units[j].level >= level:
            j += 1
        units[i:j] = units[i:j][::-1]
        i = j
