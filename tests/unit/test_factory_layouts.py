"""Unit tests for `factory.layouts` / `factory.render_html`: every layout renders for every UAE
tax case, Arabic-first layouts declare RTL, every canonical extraction field is tagged with
`data-field`, and the VAT totals baked into the HTML equal independently recomputed ERP math
(PRD §10 M2.2)."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from apx.validate.ids import is_valid_trn
from factory.erp import generate_erp
from factory.layouts import (
    BUYER_TRN,
    HEADER_FIELD_NAMES,
    LAYOUTS,
    LINE_FIELD_NAMES,
    LayoutMode,
    TaxCase,
    build_invoice_context,
    compute_totals,
    render_invoice,
)

FONTS_DIR = Path(__file__).resolve().parents[2] / "factory" / "fonts"
DATA_FIELD_RE = re.compile(r'data-field="([^"]*)"')
MAX_FONT_BYTES = 1_000_000

_ERP = generate_erp(seed=101)


def _field_names(html: str) -> set[str]:
    return set(DATA_FIELD_RE.findall(html))


def test_at_least_twelve_layouts_registered() -> None:
    assert len(LAYOUTS) >= 12
    assert len({layout.id for layout in LAYOUTS}) == len(LAYOUTS), "layout ids must be unique"
    assert len({layout.template for layout in LAYOUTS}) == len(LAYOUTS), "templates must be unique"


def test_all_language_modes_and_both_directions_are_covered() -> None:
    modes = {layout.mode for layout in LAYOUTS}
    directions = {layout.direction for layout in LAYOUTS}
    assert modes == {LayoutMode.EN, LayoutMode.AR_FIRST, LayoutMode.BILINGUAL}
    assert directions == {"ltr", "rtl"}


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: layout.id)
@pytest.mark.parametrize("tax_case", list(TaxCase), ids=lambda tc: tc.value)
def test_every_layout_renders_every_tax_case(layout, tax_case) -> None:  # type: ignore[no-untyped-def]
    html = render_invoice(layout.id, _ERP, vendor_index=0, tax_case=tax_case, n_lines=5)
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "<html" in html


@pytest.mark.parametrize("n_lines", [1, 13, 40])
@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: layout.id)
def test_line_item_count_range_renders(layout, n_lines) -> None:  # type: ignore[no-untyped-def]
    html = render_invoice(
        layout.id, _ERP, vendor_index=1, tax_case=TaxCase.STANDARD, n_lines=n_lines
    )
    # Bilingual layouts tag each line's description twice (once per language); EN/AR-first
    # layouts tag it once.
    expected_occurrences = n_lines * (2 if layout.mode is LayoutMode.BILINGUAL else 1)
    assert html.count('data-field="description"') == expected_occurrences


def test_rtl_present_in_arabic_first_layouts() -> None:
    for layout in LAYOUTS:
        if layout.mode is not LayoutMode.AR_FIRST:
            continue
        html = render_invoice(layout.id, _ERP, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=4)
        assert 'dir="rtl"' in html, f"{layout.id} must declare dir=rtl somewhere"
        assert re.search(r"<html[^>]*dir=\"rtl\"", html), f"{layout.id} <html> must be dir=rtl"


def test_bilingual_layouts_carry_both_scripts() -> None:
    arabic_char_re = re.compile(r"[؀-ۿ]")
    for layout in LAYOUTS:
        if layout.mode is not LayoutMode.BILINGUAL:
            continue
        html = render_invoice(layout.id, _ERP, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=3)
        assert arabic_char_re.search(html), f"{layout.id} must contain Arabic script"
        assert "TRN" in html or "Invoice" in html or "invoice" in html.lower()


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: layout.id)
@pytest.mark.parametrize("tax_case", list(TaxCase), ids=lambda tc: tc.value)
def test_all_canonical_data_field_keys_present(layout, tax_case) -> None:  # type: ignore[no-untyped-def]
    html = render_invoice(layout.id, _ERP, vendor_index=2, tax_case=tax_case, n_lines=6)
    found = _field_names(html)
    missing = (HEADER_FIELD_NAMES | LINE_FIELD_NAMES) - found
    assert not missing, f"{layout.id}/{tax_case.value} is missing data-field(s): {missing}"


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda layout: layout.id)
@pytest.mark.parametrize("tax_case", list(TaxCase), ids=lambda tc: tc.value)
def test_vat_totals_in_html_equal_erp_math(layout, tax_case) -> None:  # type: ignore[no-untyped-def]
    context = build_invoice_context(layout, _ERP, vendor_index=3, tax_case=tax_case, n_lines=9)
    html = render_invoice(layout.id, _ERP, vendor_index=3, tax_case=tax_case, n_lines=9)

    lines = context["lines"]
    expected_subtotal = sum((line.net_amount for line in lines), start=Decimal("0"))
    expected_vat_total = sum(
        (line.net_amount * line.vat_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        for line in lines
    )
    expected_total = expected_subtotal + expected_vat_total

    assert context["subtotal"] == expected_subtotal
    assert context["vat_total"] == expected_vat_total
    assert context["total"] == expected_total
    assert compute_totals(lines) == (expected_subtotal, expected_vat_total, expected_total)

    assert f"{expected_subtotal:,.2f}" in html
    assert f"{expected_vat_total:,.2f}" in html
    assert f"{expected_total:,.2f}" in html


def test_render_is_deterministic_per_seed() -> None:
    layout = LAYOUTS[0]
    first = render_invoice(
        layout.id, _ERP, vendor_index=5, tax_case=TaxCase.REVERSE_CHARGE, n_lines=8
    )
    second = render_invoice(
        layout.id, _ERP, vendor_index=5, tax_case=TaxCase.REVERSE_CHARGE, n_lines=8
    )
    assert first == second

    fresh_erp = generate_erp(seed=_ERP.seed)
    third = render_invoice(
        layout.id, fresh_erp, vendor_index=5, tax_case=TaxCase.REVERSE_CHARGE, n_lines=8
    )
    assert first == third


def test_different_seed_changes_output() -> None:
    layout = LAYOUTS[0]
    a = render_invoice(layout.id, _ERP, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=5)
    other_erp = generate_erp(seed=_ERP.seed + 1)
    b = render_invoice(layout.id, other_erp, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=5)
    assert a != b


@pytest.mark.parametrize("bad_n_lines", [0, -1, 41, 100])
def test_line_count_out_of_range_raises(bad_n_lines: int) -> None:
    with pytest.raises(ValueError):
        render_invoice(
            LAYOUTS[0].id, _ERP, vendor_index=0, tax_case=TaxCase.STANDARD, n_lines=bad_n_lines
        )


def test_buyer_trn_is_valid() -> None:
    assert is_valid_trn(BUYER_TRN)


def test_fonts_bundled_under_size_limit_with_license() -> None:
    families = ("NotoSans", "NotoNaskhArabic", "NotoKufiArabic")
    found_any_woff = False
    for family in families:
        family_dir = FONTS_DIR / family
        assert family_dir.is_dir(), f"missing font family directory: {family_dir}"
        ofl = family_dir / "OFL.txt"
        assert ofl.is_file(), f"missing OFL.txt for {family}"
        assert "Open Font License" in ofl.read_text(encoding="utf-8")
        woff_files = list(family_dir.glob("*.woff"))
        assert woff_files, f"no .woff files bundled for {family}"
        for font_file in woff_files:
            size = font_file.stat().st_size
            assert size < MAX_FONT_BYTES, f"{font_file} is {size} bytes, must be < 1MB"
            found_any_woff = True
    assert found_any_woff
