"""Unit tests for `factory.render`: small-N (<=5 invoices) Playwright-driven rendering covering
every canonical field getting a label, boxes staying inside their page, multi-page pagination
via the inserted-spacer simulation, the clean per-page screenshot path, and PDF/label determinism
(PRD §10 M2.3)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pdfplumber
import pytest
from PIL import Image

from factory.erp import generate_erp
from factory.labels import label_bounds_problems
from factory.layouts import (
    HEADER_FIELD_NAMES,
    LAYOUTS,
    LINE_FIELD_NAMES,
    TaxCase,
    build_invoice_context,
)
from factory.render import (
    PAGE_HEIGHT_PX,
    PAGE_WIDTH_PX,
    RenderedInvoice,
    RenderWorker,
    _normalize_pdf_bytes,
    _png_chunk_crcs_are_valid,
    css_page_geometry_matches,
)
from factory.render_html import render_template

_ERP = generate_erp(seed=90210, n_vendors=5, n_users=4)
_LAYOUTS_BY_ID = {layout.id: layout for layout in LAYOUTS}


@pytest.fixture(scope="module")
def worker() -> Iterator[RenderWorker]:
    with RenderWorker(scale=1.0) as w:
        yield w


def _render(
    worker: RenderWorker, layout_id: str, *, n_lines: int, capture_pages: bool = False
) -> RenderedInvoice:
    layout = _LAYOUTS_BY_ID[layout_id]
    context = build_invoice_context(
        layout, _ERP, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=n_lines
    )
    html = render_template(layout.template, context)
    return worker.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
        capture_pages=capture_pages,
    )


# Exactly the small-N (<=5) set of renders this whole test module shares, so the suite stays
# fast: one plain layout, one Arabic-first layout, one fully-duplicated bilingual layout, one
# inline-bilingual layout, and one deliberately long invoice to exercise multi-page pagination.
@pytest.fixture(scope="module")
def rendered(worker: RenderWorker) -> dict[str, tuple[RenderedInvoice, int]]:
    cases: dict[str, tuple[str, int, bool]] = {
        "en": ("en_ledger_classic", 4, False),
        "ar_first": ("ar_ledger_classic", 4, False),
        "bilingual_duplicated": ("bilingual_modern_minimal", 4, True),
        "bilingual_inline": ("bilingual_ledger_classic", 3, False),
        "multipage": ("en_modern_minimal", 40, True),
    }
    return {
        name: (_render(worker, layout_id, n_lines=n_lines, capture_pages=capture), n_lines)
        for name, (layout_id, n_lines, capture) in cases.items()
    }


def test_normalize_pdf_bytes_masks_differing_timestamps_to_the_same_value() -> None:
    a = b"prefix /CreationDate (D:20260101000000+00'00') /ModDate (D:20260101000000+00'00') tail"
    b = b"prefix /CreationDate (D:20261231235959+05'30') /ModDate (D:20261231235959+05'30') tail"
    normalized_a = _normalize_pdf_bytes(a)
    normalized_b = _normalize_pdf_bytes(b)
    assert normalized_a == normalized_b
    assert len(normalized_a) == len(a)
    assert len(normalized_b) == len(b)


def test_normalize_pdf_bytes_is_a_noop_without_a_timestamp() -> None:
    assert _normalize_pdf_bytes(b"no timestamp here") == b"no timestamp here"


def _real_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_png_chunk_crcs_are_valid_for_a_real_png() -> None:
    assert _png_chunk_crcs_are_valid(_real_png_bytes())


def test_png_chunk_crcs_are_invalid_for_flipped_idat_byte() -> None:
    data = bytearray(_real_png_bytes())
    idat_at = data.index(b"IDAT")
    # Flip a byte inside the IDAT chunk's data (well past its type tag), which must not change
    # the chunk's *stored* CRC, so this simulates exactly the "IDAT: CRC error" corruption seen
    # from concurrent Chromium screenshots.
    data[idat_at + 8] ^= 0xFF
    assert not _png_chunk_crcs_are_valid(bytes(data))


def test_png_chunk_crcs_are_invalid_for_truncated_data() -> None:
    data = _real_png_bytes()
    assert not _png_chunk_crcs_are_valid(data[: len(data) // 2])


def test_png_chunk_crcs_are_invalid_for_non_png_bytes() -> None:
    assert not _png_chunk_crcs_are_valid(b"not a png at all")


def test_css_page_geometry_matches_shared_template() -> None:
    css_path = Path(__file__).resolve().parents[2] / "factory" / "templates" / "_shared.html.j2"
    assert css_page_geometry_matches(css_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case", ["en", "ar_first", "bilingual_duplicated", "bilingual_inline", "multipage"]
)
def test_pdf_bytes_look_like_a_pdf(
    rendered: dict[str, tuple[RenderedInvoice, int]], case: str
) -> None:
    result, _ = rendered[case]
    assert result.pdf_bytes.startswith(b"%PDF-")
    assert len(result.pdf_bytes) > 1000


@pytest.mark.parametrize(
    "case", ["en", "ar_first", "bilingual_duplicated", "bilingual_inline", "multipage"]
)
def test_labels_stay_within_page_bounds(
    rendered: dict[str, tuple[RenderedInvoice, int]], case: str
) -> None:
    result, _ = rendered[case]
    assert label_bounds_problems(result.labels) == []


def test_every_canonical_field_is_labeled(rendered: dict[str, tuple[RenderedInvoice, int]]) -> None:
    result, _ = rendered["en"]
    found = {label.field for label in result.labels.labels}
    assert found == HEADER_FIELD_NAMES | LINE_FIELD_NAMES


def test_line_item_indices_form_a_dense_zero_based_range(
    rendered: dict[str, tuple[RenderedInvoice, int]],
) -> None:
    result, n_lines = rendered["en"]
    quantity_lines = {
        label.line_index for label in result.labels.labels if label.field == "quantity"
    }
    assert quantity_lines == set(range(n_lines))


def test_bilingual_modern_minimal_labels_language_variant_fields_twice(
    rendered: dict[str, tuple[RenderedInvoice, int]],
) -> None:
    """`bilingual_modern_minimal` renders a whole second (Arabic) copy of the invoice, so a field
    whose text genuinely differs by language (`supplier_name`) is tagged in both blocks -- but a
    field with no language variant at all (`quantity`, the same digits either way) is tagged only
    once, in the English block; the Arabic block's own copy is a visual echo, not a second
    physical occurrence of the field. Tagging it a second time gave the M2.3 extraction schema two
    ground-truth boxes for one field, which is what M3.1's IoU gate caught downstream (13
    `currency` labels scored a low IoU because a consumer picked whichever of the two boxes did
    not match the one `apx.parse` found) -- PRD §10 M2.4/M3.1."""
    result, n_lines = rendered["bilingual_duplicated"]
    supplier_name_labels = [
        label for label in result.labels.labels if label.field == "supplier_name"
    ]
    assert len(supplier_name_labels) == 2
    quantity_labels = [label for label in result.labels.labels if label.field == "quantity"]
    assert len(quantity_labels) == n_lines
    line_indices: list[int] = []
    for label in quantity_labels:
        assert label.line_index is not None
        line_indices.append(label.line_index)
    assert sorted(line_indices) == list(range(n_lines))


def test_multipage_invoice_spans_more_than_one_page(
    rendered: dict[str, tuple[RenderedInvoice, int]],
) -> None:
    result, _n_lines = rendered["multipage"]
    assert result.labels.n_pages > 1
    pages_seen = {label.page for label in result.labels.labels}
    assert pages_seen == set(range(result.labels.n_pages))
    assert len(result.page_pngs) == result.labels.n_pages


def test_captured_page_pngs_are_page_sized(
    rendered: dict[str, tuple[RenderedInvoice, int]],
) -> None:
    result, _ = rendered["multipage"]
    for page_png in result.page_pngs:
        with Image.open(BytesIO(page_png)) as image:
            width, height = image.size
        assert abs(width - round(PAGE_WIDTH_PX)) <= 2
        assert abs(height - round(PAGE_HEIGHT_PX)) <= 2


def test_render_is_deterministic_for_identical_html(worker: RenderWorker) -> None:
    layout = _LAYOUTS_BY_ID["en_compact_utility"]
    context = build_invoice_context(
        layout, _ERP, vendor_index=1, tax_case=TaxCase.ZERO_RATED, n_lines=6
    )
    html = render_template(layout.template, context)

    first = worker.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
    )
    second = worker.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
    )
    # Geometry (and therefore every label) is a pure function of the HTML/CSS layout. The PDF
    # bytes are too: `RenderWorker.render` normalizes Chromium's wall-clock `/CreationDate` and
    # `/ModDate` stamps (see `factory.render._normalize_pdf_bytes`), which would otherwise be the
    # only source of nondeterminism if the two calls straddled a clock second.
    assert first.labels == second.labels
    assert first.pdf_bytes == second.pdf_bytes


def test_pdf_timestamps_are_normalized_even_a_second_apart(worker: RenderWorker) -> None:
    layout = _LAYOUTS_BY_ID["en_compact_utility"]
    context = build_invoice_context(
        layout, _ERP, vendor_index=2, tax_case=TaxCase.EXEMPT, n_lines=3
    )
    html = render_template(layout.template, context)

    first = worker.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
    )
    time.sleep(1.1)  # long enough to roll over Chromium's second-resolution PDF timestamp
    second = worker.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
    )
    assert first.pdf_bytes == second.pdf_bytes


def test_rendered_html_declares_valid_font_face_rules() -> None:
    """Regression: autoescape once turned the font-family quotes into `&#34;`, which made every
    `@font-face` rule invalid CSS so every invoice silently rendered in system fallback fonts
    (on Linux, Arabic with no usable ToUnicode map)."""
    layout = _LAYOUTS_BY_ID["ar_ledger_classic"]
    context = build_invoice_context(
        layout, _ERP, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=1
    )
    html = render_template(layout.template, context)
    assert "&#34;" not in html
    assert html.count('@font-face{font-family:"Noto ') == 6


def test_every_bundled_font_face_loads_in_the_browser(worker: RenderWorker) -> None:
    result = _render(worker, "ar_ledger_classic", n_lines=2)
    with pdfplumber.open(BytesIO(result.pdf_bytes)) as pdf:
        fonts = {char["fontname"].split("+")[-1] for char in pdf.pages[0].chars}
    assert fonts, "no text extracted"
    assert all(name.startswith("Noto") for name in fonts), fonts
