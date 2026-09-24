"""Unit tests for `attacks.render`: renders one committed case per category and verifies its
payload actually lands in the PDF in the category's intended hidden form (PRD §10 M6.3:
"generated PDFs actually contain the payload in the intended hidden form"), that rendering is
byte-identical across repeated calls with the same seed, and that `evals.splits.register_documents`
accepts the resulting sha256s into the `attacks` eval set without leakage (PRD §10 M6.3;
`evals/splits.py`)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from io import BytesIO

import pdfplumber
import pytest

from attacks.manifest import CATEGORIES, AttackCase, load_all_cases, load_cases
from attacks.render import build_suite_erp, render_case
from attacks.textsource import homoglyph_spoof
from evals.splits import empty_splits, register_documents, validate_splits
from factory.erp import Erp
from factory.hooks import ATTACK_HOOKS
from factory.layouts import LAYOUTS, TaxCase, build_invoice_context
from factory.render import RenderWorker

_LAYOUTS_BY_ID = {layout.id: layout for layout in LAYOUTS}


@pytest.fixture(scope="module")
def erp() -> Erp:
    return build_suite_erp()


@pytest.fixture(scope="module")
def worker() -> Iterator[RenderWorker]:
    with RenderWorker(scale=1.0) as w:
        yield w


@pytest.fixture(scope="module")
def first_case_per_category() -> dict[str, AttackCase]:
    return {spec.name: load_cases(spec.name)[0] for spec in CATEGORIES}


def _numeric_or_latin_anchor(payload_text: str) -> str:
    """A substring of `payload_text` that survives Chromium's Arabic text shaping/reordering
    unchanged (an IBAN, a 6-digit reference, or a run of Latin letters/digits) -- used to check
    a rendered sentence landed in the PDF even for the Arabic-language cases, where full-text
    substring matching is unreliable across bidi/ligature reshaping (verified empirically:
    Arabic glyphs come back from `pdfplumber` in presentation-form/visual order, but embedded
    Latin/digit runs do not)."""
    for pattern in (r"AE\d{21}", r"\b\d{6}\b", r"[A-Za-z][A-Za-z0-9 @.]{5,}[A-Za-z0-9]"):
        match = re.search(pattern, payload_text)
        if match:
            return match.group(0)
    raise AssertionError(f"no stable anchor found in payload: {payload_text!r}")


@pytest.mark.parametrize(
    "category",
    [
        "hidden_white_text",
        "tiny_font",
        "off_page",
        "behind_image",
        "iban_swap_instruction",
        "fake_approval_stamp",
        "line_item_injection",
    ],
)
def test_rendered_glyph_text_categories_contain_their_payload_anchor(
    category: str,
    erp: Erp,
    worker: RenderWorker,
    first_case_per_category: dict[str, AttackCase],
) -> None:
    case = first_case_per_category[category]
    result = render_case(case, erp=erp, worker=worker)
    anchor = _numeric_or_latin_anchor(case.payload_text)
    with pdfplumber.open(BytesIO(result.pdf_bytes)) as pdf:
        joined = "".join(c["text"] for page in pdf.pages for c in page.chars)
    assert anchor in joined


def test_pdf_metadata_injection_lands_in_info_dict_and_xmp(
    erp: Erp, worker: RenderWorker, first_case_per_category: dict[str, AttackCase]
) -> None:
    case = first_case_per_category["pdf_metadata_injection"]
    result = render_case(case, erp=erp, worker=worker)
    with pdfplumber.open(BytesIO(result.pdf_bytes)) as pdf:
        metadata = pdf.metadata
    assert metadata.get("Keywords") == case.payload_text
    assert metadata.get("Subject") == case.payload_text
    assert case.payload_text.encode("utf-8") in result.pdf_bytes


def test_zero_width_lands_in_a_custom_info_field_with_zero_width_chars(
    erp: Erp, worker: RenderWorker, first_case_per_category: dict[str, AttackCase]
) -> None:
    case = first_case_per_category["zero_width"]
    result = render_case(case, erp=erp, worker=worker)
    with pdfplumber.open(BytesIO(result.pdf_bytes)) as pdf:
        value = pdf.metadata.get("APZeroWidthNote", "")
    assert value != ""
    assert any(chr(cp) in value for cp in (0x200B, 0x200C, 0x200D))
    # the visible characters (with zero-width chars removed) reproduce the original payload
    stripped = "".join(ch for ch in value if ch not in (chr(0x200B), chr(0x200C), chr(0x200D)))
    assert stripped == case.payload_text


def test_bidi_override_lands_in_a_custom_info_field_with_control_chars(
    erp: Erp, worker: RenderWorker, first_case_per_category: dict[str, AttackCase]
) -> None:
    case = first_case_per_category["bidi_override"]
    result = render_case(case, erp=erp, worker=worker)
    with pdfplumber.open(BytesIO(result.pdf_bytes)) as pdf:
        value = pdf.metadata.get("APBidiNote", "")
    assert value.startswith(chr(0x202E))
    assert value.endswith(chr(0x202C))
    assert case.payload_text in value


def test_homoglyph_vendor_spoofs_the_actual_rendered_vendor_name(
    erp: Erp, first_case_per_category: dict[str, AttackCase]
) -> None:
    case = first_case_per_category["homoglyph_vendor"]
    layout = _LAYOUTS_BY_ID[case.layout_id]
    context = build_invoice_context(
        layout,
        erp,
        vendor_index=case.vendor_index,
        tax_case=TaxCase(case.tax_case),
        n_lines=case.n_lines,
        seed=case.seed,
    )
    original_name = context["supplier_name_en"]
    mutated = ATTACK_HOOKS.apply_context(context, names=("homoglyph_vendor",), seed=case.seed)
    assert mutated["supplier_name_en"] == homoglyph_spoof(original_name, seed=case.seed)
    assert mutated["supplier_name_en"] != original_name


def test_filename_email_injection_writes_a_sidecar_email_body(
    erp: Erp, worker: RenderWorker, first_case_per_category: dict[str, AttackCase]
) -> None:
    case = first_case_per_category["filename_email_injection"]
    result = render_case(case, erp=erp, worker=worker)
    assert len(result.extra_files) == 1
    (email_name, email_bytes) = next(iter(result.extra_files.items()))
    assert email_name.endswith(".email.txt")
    assert case.payload_text in email_bytes.decode("utf-8")
    assert result.pdf_filename.endswith(".pdf")
    assert not any(ch in result.pdf_filename for ch in '<>:"/\\|?*')


def test_render_case_is_byte_identical_across_repeated_calls(
    erp: Erp, worker: RenderWorker, first_case_per_category: dict[str, AttackCase]
) -> None:
    for category in ("hidden_white_text", "off_page", "pdf_metadata_injection", "zero_width"):
        case = first_case_per_category[category]
        a = render_case(case, erp=erp, worker=worker)
        b = render_case(case, erp=erp, worker=worker)
        assert a.sha256 == b.sha256
        assert a.pdf_bytes == b.pdf_bytes


# --- leakage / splits registration --------------------------------------------------------------


def test_attack_case_sha256s_register_into_the_attacks_eval_set_without_leakage(
    erp: Erp, worker: RenderWorker, first_case_per_category: dict[str, AttackCase]
) -> None:
    splits = empty_splits()
    assignments: dict[str, str] = {}
    for case in first_case_per_category.values():
        result = render_case(case, erp=erp, worker=worker)
        assignments[result.sha256] = "eval"
    splits = register_documents(splits, "attacks", assignments)
    assert set(splits.sets["attacks"]["eval"]) == set(assignments)
    assert validate_splits(splits) == []


def test_registering_the_same_sha256_twice_does_not_duplicate_it() -> None:
    splits = empty_splits()
    splits = register_documents(splits, "attacks", {"a" * 64: "eval"})
    splits = register_documents(splits, "attacks", {"a" * 64: "eval"})
    assert splits.sets["attacks"]["eval"] == ["a" * 64]


def test_all_committed_cases_have_valid_sha256_shaped_ids_are_loadable() -> None:
    # Sanity check the full committed suite loads and every id is well-formed, without
    # rendering all ~300+ PDFs in this fast unit test (rendering a representative subset is
    # covered above; PDFs themselves are generated at test/run time, never committed).
    cases = load_all_cases()
    assert len(cases) >= 300
    for case in cases:
        assert re.fullmatch(r"A\d{2}-\d{3}", case.id)
