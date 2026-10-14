"""factory/selfcheck.py: the M2.4 label self-check CLI. For every digital PDF the M2.3 factory
produced, extracts the pdfplumber text within each DOM-derived labeled box (unioning wrapped
lines back into logical reading order, reversing right-to-left/Arabic runs and normalizing digit
scripts on both sides) and compares it exactly to the label's ground-truth value, reporting an
exact-match rate overall and broken down by field, layout and EN/AR to
`reports/factory_selfcheck.md`."""

from __future__ import annotations

import argparse
import os
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.pdf import PDF

from factory.labels import BBox, Label, LabelSet

# --- text normalization, shared by both sides of the comparison --------------------------------

_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_TRANSLATION = str.maketrans(
    {
        **{digit: str(i) for i, digit in enumerate(_ARABIC_INDIC_DIGITS)},
        **{digit: str(i) for i, digit in enumerate(_EXTENDED_ARABIC_INDIC_DIGITS)},
    }
)

# Bidi control characters a browser's own text layout can imply (and this factory's `ltr` filter
# wraps values in a `<bdi>` element for) but which never belong to a label's *logical* value, plus
# the NUL filler glyph some shaped-glyph clusters leave behind in a PDF's ToUnicode CMap.
_STRIP_CHARS = "‎‏‪‫‬‭‮⁦⁧⁨⁩\x00"
_STRIP_TABLE = str.maketrans("", "", _STRIP_CHARS)


def normalize_digits(text: str) -> str:
    """Maps Arabic-Indic (U+0660-0669) and Extended Arabic-Indic/Persian (U+06F0-06F9) digits to
    ASCII 0-9, so a value written with one digit script compares equal to the same value rendered
    or extracted with the other ("digits normalized consistently on both sides")."""
    return text.translate(_DIGIT_TRANSLATION)


def normalize_for_compare(text: str) -> str:
    """The canonical form used on both sides of the self-check comparison: NFKC folds Arabic
    presentation-form glyphs (the isolated/initial/medial/final shapes pdfplumber reads back out
    of a PDF's ToUnicode CMap, plus ligatures such as lam-alef) to their plain base letters, digit
    scripts are normalized, bidi-control and NUL filler characters are dropped, and whitespace
    runs collapse to a single space
    (a value that wraps across `Label.line_boxes` is rejoined with a literal space that has no
    rendered glyph of its own)."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_digits(text)
    text = text.translate(_STRIP_TABLE)
    return " ".join(text.split())


_RTL_BIDI_CLASSES = frozenset({"AL", "R"})


def value_is_rtl(value: str) -> bool:
    """True if `value`'s strongly-directional characters are majority right-to-left (Arabic or
    other RTL script). Decides which visual-to-logical reordering the glyphs pdfplumber reads out
    of one line box need: RTL text is laid out on the page starting at the highest x-coordinate,
    so pdfplumber's ascending-x order is the reverse of the logical string."""
    rtl = sum(1 for c in value if unicodedata.bidirectional(c) in _RTL_BIDI_CLASSES)
    ltr = sum(1 for c in value if unicodedata.bidirectional(c) == "L")
    return rtl > ltr


# --- pdfplumber extraction ----------------------------------------------------------------------

# Small outward padding so a glyph's ink bounding box (what pdfplumber measures) isn't clipped by
# rounding drift against the CSS layout box (what `factory.render`'s `getClientRects()` measured);
# kept small since these invoices pack columns tightly and a larger pad risks pulling in a
# neighboring field's text instead.
_CROP_PAD_PT = 0.75


def _bbox_to_points(
    bbox: BBox, *, page_width: float, page_height: float, pad: float = _CROP_PAD_PT
) -> tuple[float, float, float, float]:
    """Converts one `[0, 1]`-normalized `BBox` fraction into absolute PDF-point coordinates on a
    page of size `page_width` x `page_height` points, padded outward by `pad` and clamped to the
    page."""
    x0f, y0f, wf, hf = bbox
    x0 = x0f * page_width - pad
    top = y0f * page_height - pad
    x1 = (x0f + wf) * page_width + pad
    bottom = (y0f + hf) * page_height + pad
    return (max(0.0, x0), max(0.0, top), min(page_width, x1), min(page_height, bottom))


# A CSS line box (what `getClientRects()` measures, and so what a `Label`'s box is built from) is
# taller than the font's own glyph ink -- `line-height` reserves leading above the ascent and
# below the descent -- so a crop built from that box, even un-padded, can reach a few points into
# a tightly-spaced neighboring row. Below this tolerance two chars are the same visual row; at or
# above it they're different rows, comfortably under the smallest row-to-row gap these templates
# use.
_ROW_CLUSTER_TOL_PT = 2.0


_CharDict = dict[str, object]


def _char_num(char: _CharDict, key: str) -> float:
    """`float()` one numeric field of a pdfplumber char dict. pdfplumber types every char as
    `dict[str, Any]` (`T_obj` in its own `_typing.py`), but `top`/`bottom`/`x0`/`x1` are always
    `T_num` (`int | float`) in practice; the assert gives mypy --strict a real narrowing instead
    of an unchecked `cast`."""
    value = char[key]
    assert isinstance(value, (int, float)), f"pdfplumber char[{key!r}] was {type(value)!r}"
    return float(value)


def _char_is_rtl(text: str) -> bool | None:
    """One character's own bidi strength: `True` for a strong-RTL letter, `False` for a strong-LTR
    letter or digit, `None` for a weak/neutral character (space, punctuation) that carries no
    direction of its own."""
    bidi_class = unicodedata.bidirectional(text)
    if bidi_class in _RTL_BIDI_CLASSES:
        return True
    if bidi_class in ("L", "EN", "AN"):
        return False
    return None


# pdfplumber's `strict=False` crop *clips* a character's reported bbox to the query region rather
# than keeping its true one -- so a character just outside `bbox_pts` (e.g. the sentence-final "."
# this factory's footer prints immediately, with no space, after an IBAN's own span) would be
# reported with a bbox trimmed to sit deceptively close to `bbox_pts`, defeating the
# distance-from-target check `extract_label_text` runs afterward. Querying a margin well past
# `_CROP_PAD_PT` first, wide enough that no character's *true* bbox gets clipped, keeps that check
# honest; row-clustering plus nearest-box assignment downstream still confines what's kept.
_QUERY_MARGIN_PT = 6.0


def _row_chars(page: Page, bbox_pts: tuple[float, float, float, float]) -> list[_CharDict]:
    """Every character near one crop box (queried with `_QUERY_MARGIN_PT` of slack so pdfplumber
    never has to clip a true bbox, see above), NUL filler glyphs dropped, clustered into visual
    rows (a wide query can catch a neighboring row's text, which would otherwise interleave with
    the intended row once both get sorted by x-position), keeping only the row closest to the
    crop's own vertical center, sorted left-to-right by on-page position."""
    x0, top, x1, bottom = bbox_pts
    if x1 <= x0 or bottom <= top:
        return []
    query = (
        max(0.0, x0 - _QUERY_MARGIN_PT),
        max(0.0, top - _QUERY_MARGIN_PT),
        min(float(page.width), x1 + _QUERY_MARGIN_PT),
        min(float(page.height), bottom + _QUERY_MARGIN_PT),
    )
    crop = page.crop(query, strict=False)
    chars: list[_CharDict] = [c for c in crop.chars if c["text"] != "\x00"]
    if not chars:
        return []

    chars.sort(key=lambda c: _char_num(c, "top"))
    rows: list[list[_CharDict]] = [[chars[0]]]
    for char in chars[1:]:
        if abs(_char_num(char, "top") - _char_num(rows[-1][-1], "top")) <= _ROW_CLUSTER_TOL_PT:
            rows[-1].append(char)
        else:
            rows.append([char])

    target_center = (top + bottom) / 2
    row = min(
        rows,
        key=lambda r: abs(
            (min(_char_num(c, "top") for c in r) + max(_char_num(c, "bottom") for c in r)) / 2
            - target_center
        ),
    )
    row.sort(key=lambda c: _char_num(c, "x0"))
    return row


def _char_center(char: _CharDict) -> float:
    return (_char_num(char, "x0") + _char_num(char, "x1")) / 2


def _cluster_marks(chars: Sequence[_CharDict]) -> list[list[_CharDict]]:
    """Groups on-page-ascending-x characters into base-character-plus-combining-mark clusters, so a
    combining mark (Unicode category `Mn`/`Mc`/`Me`, e.g. an Arabic short-vowel diacritic like
    kasra U+0650) never gets reordered relative to the base letter it renders on top of. A plain
    whole-run reversal (`_chars_to_text` below) would otherwise be free to swap a mark and its base
    whenever they're adjacent, turning e.g. "لِ" (lam + kasra) into "ِل".

    A mark has (near-)zero advance width, so which base it visually sits on cannot be read off
    simply by nearest sequence-order neighbor: it can be measurably closer, in reported x0, to the
    *other* side's neighbor than to its true base (font hinting/rounding on a zero-width glyph),
    as observed on this same value rendered by two different layouts. Comparing the mark's own
    center to both its immediately preceding and following non-mark neighbors' centers and picking
    whichever is nearer is resilient to that rounding either way."""
    is_mark = [unicodedata.category(str(c["text"])).startswith("M") for c in chars]
    base_indices = [i for i, mark in enumerate(is_mark) if not mark]
    clusters: dict[int, list[_CharDict]] = {i: [chars[i]] for i in base_indices}
    for i, char in enumerate(chars):
        if not is_mark[i]:
            continue
        candidates = [
            b
            for b in (
                max((b for b in base_indices if b < i), default=None),
                min((b for b in base_indices if b > i), default=None),
            )
            if b is not None
        ]
        if not candidates:
            clusters[i] = [char]
            continue
        mark_x = _char_center(char)
        nearest = min(candidates, key=lambda b: abs(mark_x - _char_center(chars[b])))
        clusters[nearest].append(char)
    return [cluster for _, cluster in sorted(clusters.items(), key=lambda kv: kv[0])]


def _chars_to_text(chars: Sequence[_CharDict], *, default_rtl: bool) -> str:
    """Turns one run's characters (already in on-page ascending-x order) into logical text. A
    run's own direction is decided from its *own* characters' bidi strength (falling back to
    `default_rtl`, the label's overall direction, only when the run has no strongly directional
    character at all, e.g. a run made purely of digits and punctuation already classified as LTR)
    -- a run majority-classified RTL gets its base+combining-mark clusters reversed (see
    `_cluster_marks`), since a contiguous RTL run's on-page ascending-x order is the reverse of its
    own logical order; otherwise its ascending-x order is already correct."""
    if not chars:
        return ""
    strengths = [_char_is_rtl(str(c["text"])) for c in chars]
    n_rtl = sum(1 for s in strengths if s is True)
    n_ltr = sum(1 for s in strengths if s is False)
    is_rtl = default_rtl if n_rtl == n_ltr else n_rtl > n_ltr
    clusters = _cluster_marks(chars)
    ordered_clusters = list(reversed(clusters)) if is_rtl else clusters
    ordered = [char for cluster in ordered_clusters for char in cluster]
    return "".join(unicodedata.normalize("NFKC", str(c["text"])) for c in ordered)


def _distance_to_box_x(x: float, box: tuple[float, float, float, float]) -> float:
    """0 if `x` falls inside `box`'s horizontal span, else the distance to the nearer edge."""
    if box[0] <= x <= box[2]:
        return 0.0
    return min(abs(x - box[0]), abs(x - box[2]))


def extract_label_text(page: Page, label: Label, *, page_width: float, page_height: float) -> str:
    """Reconstructs the text pdfplumber reads for one `Label`. `label.line_boxes` holds one DOM
    rect per rendered text-line *fragment*, which for a value containing an embedded
    left-to-right run (Western digits inside Arabic prose, e.g. a line-item description like
    "...تكييف 25 مم...") the browser bidi-splits into *several* same-line rects rather than one --
    each is an authoritative, browser-measured bidi run. Boxes sharing a `top` are one true visual
    line (only a genuinely different `top` means a real wrap onto a new line); that line's boxes
    are cropped *once*, as their union, and each of the resulting characters is then assigned to
    whichever raw (unpadded) box its own center sits closest to -- run boxes commonly abut with
    zero gap (a bidi run boundary inside one text node draws no border), so cropping and padding
    each box's own region independently can have a glyph sitting right at that shared edge overlap
    both crops and get read into both runs' text; a single crop plus nearest-box assignment gives
    every character exactly one home. Each box's assigned characters become one run, ordered
    right-to-left by leftmost x for an overall-RTL line (left-to-right otherwise) and concatenated
    with *no* synthesized separator: a real space between two same-line runs is itself a character
    with its own bbox, assigned by the very same nearest-box rule to whichever run it truly
    borders, so it already appears at that run's own edge -- adding another would only double it up
    (harmless, `normalize_for_compare` collapses whitespace), but a run boundary can also be one
    with no space at all, e.g. a neutral character such as an Arabic comma sitting, with zero gap,
    directly against an adjacent bidi run (`"...42.5N،"` glued to an embedded LTR run, the comma
    resolving into the RTL run that follows it) -- synthesizing a separator there would corrupt it.

    Joining the *lines* themselves back together needs the opposite default. A CSS soft line-wrap
    swallows the space it breaks at entirely -- neither line renders it, so no run captures it --
    except when the wrap point is a hyphen with no space in the value at all (e.g. "PO-2026-" /
    "000001"), where the hyphen itself is the visible glue and no space ever belonged. So a space is
    inserted between consecutive lines unless the previous line's captured text already ends with
    one (the wrap's space did survive, rare but harmless to not double) or with a hyphen."""
    rtl = value_is_rtl(label.value)
    raw_boxes = [
        _bbox_to_points(box, page_width=page_width, page_height=page_height, pad=0.0)
        for box in label.line_boxes
    ]
    order = sorted(range(len(raw_boxes)), key=lambda i: raw_boxes[i][1])
    line_groups: list[list[int]] = []
    for i in order:
        prev_top = raw_boxes[line_groups[-1][-1]][1] if line_groups else None
        if prev_top is not None and abs(raw_boxes[i][1] - prev_top) <= _ROW_CLUSTER_TOL_PT:
            line_groups[-1].append(i)
        else:
            line_groups.append([i])

    lines: list[str] = []
    for group in line_groups:
        group_boxes = [raw_boxes[i] for i in group]
        union = (
            max(0.0, min(b[0] for b in group_boxes) - _CROP_PAD_PT),
            max(0.0, min(b[1] for b in group_boxes) - _CROP_PAD_PT),
            min(page_width, max(b[2] for b in group_boxes) + _CROP_PAD_PT),
            min(page_height, max(b[3] for b in group_boxes) + _CROP_PAD_PT),
        )
        row = _row_chars(page, union)
        assigned: list[list[_CharDict]] = [[] for _ in group_boxes]
        for char in row:
            center = _char_center(char)
            nearest = min(
                range(len(group_boxes)), key=lambda bi: _distance_to_box_x(center, group_boxes[bi])
            )
            # A character whose *center* sits well outside every one of this label's own boxes is
            # something else entirely -- e.g. the sentence-final "." this factory's footer prints
            # immediately after an IBAN's `<bdi>`-isolated span, with no space in between, close
            # enough to fall inside the padded crop -- not a piece of this label merely clipped at
            # a shared run edge (that always leaves the character's *center*, not just a sliver of
            # its bbox, inside its own box). Drop it rather than force-assigning it to the nearest
            # box regardless of distance.
            if _distance_to_box_x(center, group_boxes[nearest]) > _CROP_PAD_PT:
                continue
            assigned[nearest].append(char)

        runs = [
            (_chars_to_text(chars, default_rtl=rtl), group_boxes[bi][0])
            for bi, chars in enumerate(assigned)
            if chars
        ]
        runs.sort(key=lambda r: r[1], reverse=rtl)
        lines.append("".join(text for text, _x0 in runs))

    text = ""
    for line in lines:
        if not line:
            continue
        if text and not text.endswith((" ", "-")):
            text += " "
        text += line
    return text


def label_matches(label: Label, extracted: str) -> bool:
    """True if `extracted` (pdfplumber's reading of `label`'s box) exactly matches `label.value`
    once both sides go through the same `normalize_for_compare` canonicalization."""
    return normalize_for_compare(label.value) == normalize_for_compare(extracted)


def check_label_set(pdf: PDF, label_set: LabelSet) -> list[bool]:
    """Checks every label in `label_set` against `pdf`'s text layer; returns one bool per label,
    in `label_set.labels` order. The primitive the unit tests and `check_document` both use."""
    results: list[bool] = []
    for label in label_set.labels:
        page = pdf.pages[label.page]
        extracted = extract_label_text(
            page, label, page_width=float(page.width), page_height=float(page.height)
        )
        results.append(label_matches(label, extracted))
    return results


# --- dataset-scale driver ------------------------------------------------------------------------

_MAX_FAILURES_PER_DOC = 3
_MAX_FAILURES_IN_REPORT = 20


@dataclass(frozen=True)
class LabelFailure:
    """One mismatch, kept for the report's small "sample failures" section."""

    sha256: str
    split: str
    layout_id: str
    field_name: str
    line_index: int | None
    page: int
    expected: str
    extracted: str


@dataclass(frozen=True)
class DocSummary:
    """One document's self-check outcome: aggregate counts only (never every per-label record),
    so this stays cheap to ship back from a worker process across thousands of documents."""

    sha256: str
    split: str
    layout_id: str
    total: int
    matched: int
    field_totals: dict[str, int] = field(default_factory=dict)
    field_matched: dict[str, int] = field(default_factory=dict)
    arabic_total: int = 0
    arabic_matched: int = 0
    latin_total: int = 0
    latin_matched: int = 0
    failures: tuple[LabelFailure, ...] = ()


def check_document(pdf_path: Path, labels_path: Path, *, split: str) -> DocSummary:
    """Runs the self-check for one rendered invoice: every label in `labels_path` against the text
    layer of `pdf_path`."""
    label_set = LabelSet.model_validate_json(labels_path.read_text(encoding="utf-8"))
    sha256 = pdf_path.stem
    field_totals: Counter[str] = Counter()
    field_matched: Counter[str] = Counter()
    arabic_total = arabic_matched = latin_total = latin_matched = 0
    matched_count = 0
    failures: list[LabelFailure] = []

    with pdfplumber.open(pdf_path) as pdf:
        for label in label_set.labels:
            page = pdf.pages[label.page]
            extracted = extract_label_text(
                page, label, page_width=float(page.width), page_height=float(page.height)
            )
            is_match = label_matches(label, extracted)
            is_arabic = value_is_rtl(label.value)

            field_totals[label.field] += 1
            if is_arabic:
                arabic_total += 1
            else:
                latin_total += 1

            if is_match:
                matched_count += 1
                field_matched[label.field] += 1
                if is_arabic:
                    arabic_matched += 1
                else:
                    latin_matched += 1
            elif len(failures) < _MAX_FAILURES_PER_DOC:
                failures.append(
                    LabelFailure(
                        sha256=sha256,
                        split=split,
                        layout_id=label_set.layout_id,
                        field_name=label.field,
                        line_index=label.line_index,
                        page=label.page,
                        expected=label.value,
                        extracted=extracted,
                    )
                )

    return DocSummary(
        sha256=sha256,
        split=split,
        layout_id=label_set.layout_id,
        total=len(label_set.labels),
        matched=matched_count,
        field_totals=dict(field_totals),
        field_matched=dict(field_matched),
        arabic_total=arabic_total,
        arabic_matched=arabic_matched,
        latin_total=latin_total,
        latin_matched=latin_matched,
        failures=tuple(failures),
    )


def discover_documents(data_dir: Path) -> list[tuple[Path, Path, str]]:
    """Every `(pdf_path, labels_path, split)` under `data_dir/<split>/*.pdf`."""
    documents: list[tuple[Path, Path, str]] = []
    for split_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for pdf_path in sorted(split_dir.glob("*.pdf")):
            labels_path = pdf_path.parent / f"{pdf_path.stem}.labels.json"
            if not labels_path.exists():
                raise FileNotFoundError(f"no labels.json next to {pdf_path}")
            documents.append((pdf_path, labels_path, split_dir.name))
    return documents


def run_self_check(
    *, data_dir: Path, workers: int = 1, limit: int | None = None
) -> list[DocSummary]:
    """Runs `check_document` over every document `discover_documents` finds under `data_dir`,
    optionally capped to the first `limit` documents (debugging), in parallel across `workers`
    processes."""
    documents = discover_documents(data_dir)
    if limit is not None:
        documents = documents[:limit]

    if workers <= 1:
        return [check_document(pdf, labels, split=split) for pdf, labels, split in documents]

    summaries: list[DocSummary] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(check_document, pdf, labels, split=split)
            for pdf, labels, split in documents
        ]
        for future in as_completed(futures):
            summaries.append(future.result())
    return summaries


@dataclass(frozen=True)
class SelfCheckSummary:
    """The aggregated self-check outcome across every document `run_self_check` checked."""

    overall_total: int
    overall_matched: int
    by_field: dict[str, tuple[int, int]]
    by_layout: dict[str, tuple[int, int]]
    by_language: dict[str, tuple[int, int]]
    by_split: dict[str, tuple[int, int]]
    failures: tuple[LabelFailure, ...]

    @property
    def overall_rate(self) -> float:
        return self.overall_matched / self.overall_total if self.overall_total else 1.0


def aggregate(
    summaries: Sequence[DocSummary], *, max_failures: int = _MAX_FAILURES_IN_REPORT
) -> SelfCheckSummary:
    """Merges every document's `DocSummary` into one dataset-wide `SelfCheckSummary`
    ("exact-match rate overall and per field / per layout / EN vs AR")."""
    field_total: Counter[str] = Counter()
    field_matched: Counter[str] = Counter()
    layout_total: Counter[str] = Counter()
    layout_matched: Counter[str] = Counter()
    split_total: Counter[str] = Counter()
    split_matched: Counter[str] = Counter()
    arabic_total = arabic_matched = latin_total = latin_matched = 0
    overall_total = overall_matched = 0
    failures: list[LabelFailure] = []

    for doc in summaries:
        overall_total += doc.total
        overall_matched += doc.matched
        layout_total[doc.layout_id] += doc.total
        layout_matched[doc.layout_id] += doc.matched
        split_total[doc.split] += doc.total
        split_matched[doc.split] += doc.matched
        arabic_total += doc.arabic_total
        arabic_matched += doc.arabic_matched
        latin_total += doc.latin_total
        latin_matched += doc.latin_matched
        for name, count in doc.field_totals.items():
            field_total[name] += count
        for name, count in doc.field_matched.items():
            field_matched[name] += count
        if len(failures) < max_failures:
            failures.extend(doc.failures[: max_failures - len(failures)])

    return SelfCheckSummary(
        overall_total=overall_total,
        overall_matched=overall_matched,
        by_field={name: (field_matched.get(name, 0), total) for name, total in field_total.items()},
        by_layout={
            name: (layout_matched.get(name, 0), total) for name, total in layout_total.items()
        },
        by_language={
            "latin": (latin_matched, latin_total),
            "arabic": (arabic_matched, arabic_total),
        },
        by_split={name: (split_matched.get(name, 0), total) for name, total in split_total.items()},
        failures=tuple(failures),
    )


def _rate_cell(matched: int, total: int) -> str:
    return f"{matched}/{total} ({matched / total:.2%})" if total else "n/a"


def format_report(summary: SelfCheckSummary, *, threshold: float) -> str:
    """Renders `summary` as a small Markdown report."""
    status = "PASS" if summary.overall_rate >= threshold else "FAIL"
    lines = [
        "# Factory label self-check (M2.4)",
        "",
        "pdfplumber text at each DOM-labeled box vs. the label's ground-truth value, on every "
        "digital PDF in the M2.3 UAE invoice dataset.",
        "",
        f"**Overall exact match: {_rate_cell(summary.overall_matched, summary.overall_total)} "
        f"-- {status}** (threshold {threshold:.1%})",
        "",
        "## By language",
        "",
        "| script | exact match |",
        "|---|---|",
    ]
    for name in ("latin", "arabic"):
        matched, total = summary.by_language[name]
        lines.append(f"| {name} | {_rate_cell(matched, total)} |")

    lines += ["", "## By split", "", "| split | exact match |", "|---|---|"]
    for name in sorted(summary.by_split):
        matched, total = summary.by_split[name]
        lines.append(f"| {name} | {_rate_cell(matched, total)} |")

    lines += ["", "## By layout", "", "| layout | exact match |", "|---|---|"]
    for name in sorted(summary.by_layout):
        matched, total = summary.by_layout[name]
        lines.append(f"| {name} | {_rate_cell(matched, total)} |")

    lines += ["", "## By field", "", "| field | exact match |", "|---|---|"]
    for name in sorted(summary.by_field):
        matched, total = summary.by_field[name]
        lines.append(f"| {name} | {_rate_cell(matched, total)} |")

    if summary.failures:
        lines += [
            "",
            "## Sample failures",
            "",
            "| sha256 | split | layout | field | line | page | expected | extracted |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in summary.failures:
            lines.append(
                f"| {f.sha256[:12]} | {f.split} | {f.layout_id} | {f.field_name} | "
                f"{f.line_index if f.line_index is not None else ''} | {f.page} | "
                f"`{f.expected[:40]}` | `{f.extracted[:40]}` |"
            )

    return "\n".join(lines) + "\n"


def write_report(summary: SelfCheckSummary, path: Path, *, threshold: float) -> None:
    """Writes `format_report(summary)` to `path` as UTF-8 with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(format_report(summary, threshold=threshold))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m factory.selfcheck",
        description="M2.4 pdfplumber label self-check for the UAE invoice factory dataset.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae"))
    parser.add_argument("--report", type=Path, default=Path("reports/factory_selfcheck.md"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument(
        "--limit", type=int, default=None, help="only check the first N documents (debugging)"
    )
    parser.add_argument("--threshold", type=float, default=0.995)
    args = parser.parse_args(argv)

    started = time.monotonic()
    summaries = run_self_check(data_dir=args.data_dir, workers=args.workers, limit=args.limit)
    summary = aggregate(summaries)
    elapsed = time.monotonic() - started

    write_report(summary, args.report, threshold=args.threshold)

    print(f"checked {len(summaries)} documents / {summary.overall_total} labels in {elapsed:.1f}s")
    print(f"overall exact-match rate: {summary.overall_rate:.3%}")
    print(f"report written to {args.report}")

    return 0 if summary.overall_rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
