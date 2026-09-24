"""Unit tests for `factory.labels`: pure geometry (no Playwright) covering per-page assignment,
multi-line union boxes, unknown/empty field rejection, bounds checking and JSON round-tripping
(PRD §10 M2.3)."""

from __future__ import annotations

import json

import pytest

from factory.labels import (
    Label,
    LabelSet,
    RawField,
    RawPagination,
    RawRect,
    build_label_set,
    dump_label_set_json,
    label_bounds_problems,
    write_label_set_json,
)

PAGE_W = 100.0
PAGE_H = 200.0
HEADER_NAMES = frozenset({"invoice_number", "supplier_name"})
LINE_NAMES = frozenset({"description", "quantity"})


def _rect(left: float, top: float, width: float, height: float) -> RawRect:
    return {"left": left, "top": top, "width": width, "height": height}


def _pagination(fields: list[RawField], total_pages: int = 1) -> RawPagination:
    return {"fields": fields, "totalPages": total_pages}


def test_single_rect_field_produces_one_label_on_page_zero() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "invoice_number",
                "text": "INV-2026-000001",
                "lineIndex": None,
                "rects": [_rect(10, 20, 30, 10)],
            }
        ]
    )
    result = build_label_set(
        raw,
        layout_id="test",
        page_width_px=PAGE_W,
        page_height_px=PAGE_H,
        header_field_names=HEADER_NAMES,
        line_field_names=LINE_NAMES,
    )
    assert len(result.labels) == 1
    label = result.labels[0]
    assert label.field == "invoice_number"
    assert label.value == "INV-2026-000001"
    assert label.page == 0
    assert label.line_index is None
    assert label.bbox == pytest.approx((0.10, 0.10, 0.30, 0.05))
    assert len(label.line_boxes) == 1
    assert label.line_boxes[0] == pytest.approx(label.bbox)


def test_wrapped_value_produces_union_box_and_per_line_boxes() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "supplier_name",
                "text": "Al Falah Trading LLC",
                "lineIndex": None,
                "rects": [_rect(10, 20, 50, 10), _rect(10, 30, 25, 10)],
            }
        ]
    )
    result = build_label_set(
        raw,
        layout_id="test",
        page_width_px=PAGE_W,
        page_height_px=PAGE_H,
        header_field_names=HEADER_NAMES,
        line_field_names=LINE_NAMES,
    )
    label = result.labels[0]
    assert len(label.line_boxes) == 2
    # Union spans from the top of the first line to the bottom of the second.
    assert label.bbox == pytest.approx((0.10, 0.10, 0.50, 0.10))


def test_field_split_across_two_pages_yields_two_labels() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "description",
                "text": "wraps across the page break",
                "lineIndex": 3,
                "rects": [_rect(10, 190, 40, 10), _rect(10, 205, 40, 10)],
            }
        ],
        total_pages=2,
    )
    result = build_label_set(
        raw,
        layout_id="test",
        page_width_px=PAGE_W,
        page_height_px=PAGE_H,
        header_field_names=HEADER_NAMES,
        line_field_names=LINE_NAMES,
    )
    assert len(result.labels) == 2
    pages = {label.page for label in result.labels}
    assert pages == {0, 1}
    for label in result.labels:
        assert label.line_index == 3
        assert len(label.line_boxes) == 1


def test_line_index_is_preserved_for_line_item_fields() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "quantity",
                "text": "12",
                "lineIndex": 2,
                "rects": [_rect(5, 5, 5, 5)],
            }
        ]
    )
    result = build_label_set(
        raw,
        layout_id="test",
        page_width_px=PAGE_W,
        page_height_px=PAGE_H,
        header_field_names=HEADER_NAMES,
        line_field_names=LINE_NAMES,
    )
    assert result.labels[0].line_index == 2


def test_unknown_field_name_raises() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "not_a_real_field",
                "text": "x",
                "lineIndex": None,
                "rects": [_rect(0, 0, 1, 1)],
            }
        ]
    )
    with pytest.raises(ValueError, match="not part of the extraction schema"):
        build_label_set(
            raw,
            layout_id="test",
            page_width_px=PAGE_W,
            page_height_px=PAGE_H,
            header_field_names=HEADER_NAMES,
            line_field_names=LINE_NAMES,
        )


def test_field_with_no_visible_rects_raises() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "invoice_number",
                "text": "INV-2026-000001",
                "lineIndex": None,
                "rects": [],
            }
        ]
    )
    with pytest.raises(ValueError, match="no visible box"):
        build_label_set(
            raw,
            layout_id="test",
            page_width_px=PAGE_W,
            page_height_px=PAGE_H,
            header_field_names=HEADER_NAMES,
            line_field_names=LINE_NAMES,
        )


def test_zero_size_rects_are_skipped_but_field_still_needs_one_visible_rect() -> None:
    raw = _pagination(
        [
            {
                "elementIndex": 0,
                "field": "invoice_number",
                "text": "x",
                "lineIndex": None,
                "rects": [_rect(0, 0, 0, 0), _rect(1, 1, 5, 5)],
            }
        ]
    )
    result = build_label_set(
        raw,
        layout_id="test",
        page_width_px=PAGE_W,
        page_height_px=PAGE_H,
        header_field_names=HEADER_NAMES,
        line_field_names=LINE_NAMES,
    )
    assert len(result.labels) == 1
    assert len(result.labels[0].line_boxes) == 1


def test_label_bounds_problems_empty_for_well_formed_labels() -> None:
    label_set = LabelSet(
        layout_id="test",
        n_pages=1,
        labels=(
            Label(
                field="invoice_number",
                value="x",
                page=0,
                bbox=(0.1, 0.1, 0.2, 0.05),
                line_boxes=((0.1, 0.1, 0.2, 0.05),),
            ),
        ),
    )
    assert label_bounds_problems(label_set) == []


@pytest.mark.parametrize(
    "bbox",
    [
        (-0.01, 0.1, 0.2, 0.05),
        (0.1, -0.01, 0.2, 0.05),
        (0.9, 0.1, 0.2, 0.05),
        (0.1, 0.9, 0.2, 0.2),
        (0.1, 0.1, 0.0, 0.05),
        (0.1, 0.1, 0.2, 0.0),
    ],
)
def test_label_bounds_problems_flags_out_of_page_boxes(
    bbox: tuple[float, float, float, float],
) -> None:
    label_set = LabelSet(
        layout_id="test",
        n_pages=1,
        labels=(Label(field="invoice_number", value="x", page=0, bbox=bbox, line_boxes=(bbox,)),),
    )
    problems = label_bounds_problems(label_set)
    assert len(problems) >= 1


def test_dump_and_write_label_set_json_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    label_set = LabelSet(
        layout_id="en_ledger_classic",
        n_pages=1,
        labels=(
            Label(
                field="invoice_number",
                value="INV-2026-000042",
                page=0,
                bbox=(0.1, 0.1, 0.2, 0.05),
                line_boxes=((0.1, 0.1, 0.2, 0.05),),
            ),
        ),
    )
    text = dump_label_set_json(label_set)
    assert text.endswith("\n")
    assert "\r\n" not in text
    restored = LabelSet.model_validate(json.loads(text))
    assert restored == label_set

    path = tmp_path / "labels.json"
    write_label_set_json(label_set, path)
    assert LabelSet.model_validate_json(path.read_text(encoding="utf-8")) == label_set
    with path.open("rb") as handle:
        assert b"\r\n" not in handle.read()
