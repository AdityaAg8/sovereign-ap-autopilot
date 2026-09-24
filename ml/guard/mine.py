"""Mines benign invoice text (EN, AR and bilingual) from the generated UAE invoice sets for the
M6.4 guard's hard negatives (PRD §10 M6.4). Reads only a dataset's `train` pool, never cal or
test, so the guard data cannot leak into the M3/M7 extraction evals.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

MINED_SPLIT = "train"  # the only dataset pool this module will read

_CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("tax_case_note_en", "en"),
    ("tax_case_note_ar", "ar"),
    ("tax_case_label_en", "en"),
    ("tax_case_label_ar", "ar"),
    ("supplier_name_en", "en"),
    ("supplier_name_ar", "ar"),
    ("supplier_address_en", "en"),
    ("buyer_name_en", "en"),
    ("buyer_name_ar", "ar"),
    ("buyer_address_en", "en"),
)
_ARABIC_RANGES = ((0x0600, 0x06FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))  # incl. presentation forms
_ARABIC = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _ARABIC_RANGES) + "]")
_WORD = re.compile(r"[^\W\d_]{2,}")


# Month names vary like numbers do (dates), so they are not part of a line's template.
_MONTHS = frozenset(
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "يناير",
        "فبراير",
        "مارس",
        "أبريل",
        "مايو",
        "يونيو",
        "يوليو",
        "أغسطس",
        "سبتمبر",
        "أكتوبر",
        "نوفمبر",
        "ديسمبر",
    ]
)


# Entity fields: their text varies per supplier/buyer, so all of a field's values form one
# family ("field:<name>") instead of being keyed on their (entity-specific) words.
_ENTITY_FIELDS = frozenset(
    {
        "supplier_name_en",
        "supplier_name_ar",
        "supplier_address_en",
        "buyer_name_en",
        "buyer_name_ar",
        "buyer_address_en",
    }
)


class MiningSplitError(ValueError):
    """A document outside the dataset's train pool reached the miner."""


class MalformedTruthError(ValueError):
    """A ground-truth JSON lacks the structure the miner reads."""


@dataclass(frozen=True)
class MinedText:
    text: str
    lang: str  # "en", "ar" or "bi"
    template: str  # family key: "field:<name>", "item:<en>|<ar>" or a leading-words key
    invoice: str  # source document sha256


def template_key(text: str, n_words: int = 4) -> str:
    """The first `n_words` alphabetic words (no numbers, months or punctuation),
    casefolded. Lines differing only in amounts, IBANs, dates, row numbers or PDF line wraps
    share a key and therefore a split; deliberately coarse so near-duplicates never leak."""
    words = [w for w in _WORD.findall(text.casefold()) if w not in _MONTHS]
    return " ".join(words[:n_words])


def _clean_pdf_line(line: str) -> bool:
    # The PDF text layer scrambles Arabic shaping (presentation forms, NULs), so only clean
    # Latin lines are taken from it; Arabic comes from the ground-truth context instead.
    if "\x00" in line or _ARABIC.search(line):
        return False
    return len(line) >= 15 and len(line.split()) >= 3


def _context(truth: dict[str, object]) -> dict[str, object]:
    context = truth.get("context")
    if not isinstance(context, dict):
        raise MalformedTruthError("truth JSON has no 'context' object")
    return context


def _item_key(en: str, ar: str) -> str:
    return f"item:{template_key(en)}|{template_key(ar)}"


def texts_from_truth(truth: dict[str, object]) -> Iterator[tuple[str, str, str]]:
    """(text, lang, template) triples from one document's ground-truth rendering context.
    Entity fields get a per-field template; a line item's EN, AR and combined "EN / AR" texts
    share one template built from both languages, so they always land in the same split."""
    context = _context(truth)
    for field, lang in _CONTEXT_FIELDS:
        value = context.get(field)
        if isinstance(value, str) and value.strip():
            key = f"field:{field}" if field in _ENTITY_FIELDS else template_key(value)
            if key:
                yield value, lang, key
    lines = context.get("lines", [])
    if not isinstance(lines, list):
        raise MalformedTruthError("truth context 'lines' is not a list")
    for line in lines:
        en, ar = str(line["description_en"]), str(line["description_ar"])
        key = _item_key(en, ar)
        yield en, "en", key
        yield ar, "ar", key
        yield f"{en} / {ar}", "bi", key


def _starts_inside(line: str, value: str, n_words: int = 3) -> bool:
    """True when the line's first 3 (or 2) alphabetic words occur, contiguously, in
    `value`: the line is a wrapped piece of that entity with other columns appended."""
    words = [w for w in _WORD.findall(line.casefold()) if w not in _MONTHS]
    haystack = f" {' '.join(_WORD.findall(value.casefold()))} "
    return any(
        f" {' '.join(words[:n])} " in haystack for n in range(min(n_words, len(words)), 1, -1)
    )


def pdf_line_template(line: str, context: dict[str, object]) -> str | None:
    """Template for one clean PDF text-layer line, using the document's truth to recognise
    entity text (a fragment of, or a line containing, a name/address) and line-item rows.
    Returns None for a line with no letters left to key on."""
    folded = line.casefold()
    stripped = line
    for field in sorted(_ENTITY_FIELDS):
        value = context.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        # A fragment of the entity, or a wrapped line that starts like it (the PDF splits
        # addresses mid-way and appends other columns), belongs to the entity's family.
        if folded in value.casefold() or _starts_inside(line, value):
            return f"field:{field}"
        if value.casefold() in folded:
            stripped = re.sub(re.escape(value), f" {field} ", stripped, flags=re.IGNORECASE)
    key = template_key(stripped)
    lines = context.get("lines", [])
    for item in lines if isinstance(lines, list) else []:
        en, ar = str(item["description_en"]), str(item["description_ar"])
        if key and key == template_key(en):
            return _item_key(en, ar)
    return key or None


def texts_from_pdf(pdf_path: Path) -> Iterator[tuple[str, str]]:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                if _clean_pdf_line(line.strip()):
                    yield line.strip(), "en"


def mine_dataset(
    dataset_dir: Path, *, pdfs_per_layout: int = 60, split: str = MINED_SPLIT
) -> list[MinedText]:
    """Every benign text of the clean (no fraud/attack hook) documents in `dataset_dir/train`,
    plus the text layer of the first `pdfs_per_layout` PDFs of each layout (sorted by sha)."""
    if split != MINED_SPLIT:
        raise MiningSplitError(f"refusing to mine the {split!r} pool; only {MINED_SPLIT!r}")
    out: list[MinedText] = []
    pdfs_taken: dict[str, int] = {}
    pool = (dataset_dir / split).resolve()
    for truth_path in sorted(pool.glob("*.truth.json")):
        truth_path = truth_path.resolve()
        if not truth_path.is_relative_to(pool):
            raise MiningSplitError(f"{truth_path} resolves outside {pool}")
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if truth.get("split") != split:
            raise MiningSplitError(f"{truth_path} is labelled {truth.get('split')!r}")
        if truth.get("fraud_hooks") or truth.get("attack_hooks"):
            continue
        sha = str(truth["sha256"])
        out += [MinedText(t, lang, key, sha) for t, lang, key in texts_from_truth(truth)]
        layout = str(truth["layout_id"])
        pdf = truth_path.with_name(f"{sha}.pdf").resolve()
        if not pdf.is_relative_to(pool):
            raise MiningSplitError(f"{pdf} resolves outside {pool}")
        if pdf.exists() and pdfs_taken.get(layout, 0) < pdfs_per_layout:
            pdfs_taken[layout] = pdfs_taken.get(layout, 0) + 1
            context = _context(truth)
            for text, lang in texts_from_pdf(pdf):
                key = pdf_line_template(text, context)
                if key is not None:
                    out.append(MinedText(text, lang, key, sha))
    return out


def drop_attack_overlap(mined: Iterable[MinedText], payloads: Iterable[str]) -> list[MinedText]:
    """Removes mined text equal to, containing, or contained in any attack payload."""
    keys = [" ".join(_WORD.findall(p.casefold())) for p in payloads]
    return [
        m
        for m in mined
        if not any(
            k and (k in full or full in k)
            for full in [" ".join(_WORD.findall(m.text.casefold()))]
            for k in keys
        )
    ]
