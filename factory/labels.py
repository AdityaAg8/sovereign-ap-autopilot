"""factory/labels.py: the factory label schema (field, value, page, normalized bounding box) plus
the pure geometry that turns raw DOM `getClientRects()` measurements collected by
`factory.render` into per-page `Label` records, unioning the individual line boxes of a value
that wraps across lines while keeping each line's own box. Contains no Playwright
import, so it is fully testable with synthetic input."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

BBox = tuple[float, float, float, float]
"""(x, y, width, height): each a fraction of one page's width/height, in [0, 1]."""


class RawRect(TypedDict):
    """One `DOMRect` as `factory.render`'s in-page JS collects it: CSS pixels relative to the
    document, after page-break spacers have been inserted so that page boundaries fall on exact
    multiples of the page height (see `factory.render`'s pagination script)."""

    left: float
    top: float
    width: float
    height: float


class RawField(TypedDict):
    """One `[data-field]` DOM element as collected: its field name, visible text, its 0-based
    row index within its own line-items table (`None` for header fields), and one `RawRect` per
    rendered text line fragment (more than one only when the value wraps)."""

    elementIndex: int
    field: str
    text: str
    lineIndex: int | None
    rects: list[RawRect]


class RawPagination(TypedDict):
    """The full JS collection result for one rendered invoice document."""

    fields: list[RawField]
    totalPages: int


class Label(BaseModel):
    """One ground-truth box: `field` with value `value`, appearing on 0-based page `page`.
    `bbox` is the union of every rendered line's box on that page; `line_boxes` keeps each line's
    own box (length 1 unless the value wraps). `line_index` is the 0-based row position within
    its own line-items table for line-item fields, `None` for header fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    value: str
    page: int = Field(ge=0)
    line_index: int | None = Field(default=None, ge=0)
    bbox: BBox
    line_boxes: tuple[BBox, ...]


class LabelSet(BaseModel):
    """Every label for one rendered invoice document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout_id: str
    n_pages: int = Field(ge=1)
    labels: tuple[Label, ...]


def _normalize_rect(
    rect: RawRect, *, page_width_px: float, page_height_px: float
) -> tuple[int, BBox]:
    """Splits one spacer-adjusted `RawRect` into the 0-based page it falls on and its bbox
    normalized to that page's own [0, 1] coordinate frame."""
    page_index = int(rect["top"] // page_height_px)
    local_top = rect["top"] - page_index * page_height_px
    bbox = (
        rect["left"] / page_width_px,
        local_top / page_height_px,
        rect["width"] / page_width_px,
        rect["height"] / page_height_px,
    )
    return page_index, bbox


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def build_label_set(
    raw: RawPagination,
    *,
    layout_id: str,
    page_width_px: float,
    page_height_px: float,
    header_field_names: frozenset[str],
    line_field_names: frozenset[str],
) -> LabelSet:
    """Pure transform from one `factory.render` JS collection into a `LabelSet`: deterministic,
    same `raw` input always yields the same labels. Raises `ValueError` if a
    `data-field` name is not part of the extraction schema, or if a field rendered with no
    visible box at all (a template bug, not something to silently drop)."""
    known = header_field_names | line_field_names
    labels: list[Label] = []
    for entry in raw["fields"]:
        name = entry["field"]
        if name not in known:
            raise ValueError(f"data-field {name!r} is not part of the extraction schema")
        by_page: dict[int, list[BBox]] = {}
        for rect in entry["rects"]:
            if rect["width"] <= 0 or rect["height"] <= 0:
                continue
            page_index, bbox = _normalize_rect(
                rect, page_width_px=page_width_px, page_height_px=page_height_px
            )
            by_page.setdefault(page_index, []).append(bbox)
        if not by_page:
            raise ValueError(
                f"data-field {name!r} (value {entry['text']!r}) rendered with no visible box"
            )
        for page_index in sorted(by_page):
            boxes = by_page[page_index]
            labels.append(
                Label(
                    field=name,
                    value=entry["text"],
                    page=page_index,
                    line_index=entry["lineIndex"],
                    bbox=_union_bbox(boxes),
                    line_boxes=tuple(boxes),
                )
            )
    return LabelSet(layout_id=layout_id, n_pages=raw["totalPages"], labels=tuple(labels))


def label_bounds_problems(label_set: LabelSet, *, eps: float = 1e-6) -> list[str]:
    """Every out-of-page-bounds box in `label_set` (coordinates below 0, non-positive size, or
    extending past 1). Empty means every label is well-formed; used by tests and can gate a
    generation run before it is trusted."""
    problems: list[str] = []
    for label in label_set.labels:
        boxes: list[tuple[str, BBox]] = [("bbox", label.bbox)]
        boxes.extend((f"line_boxes[{i}]", b) for i, b in enumerate(label.line_boxes))
        for tag, (x, y, w, h) in boxes:
            if x < -eps or y < -eps or w <= 0 or h <= 0 or x + w > 1 + eps or y + h > 1 + eps:
                problems.append(
                    f"{label.field}@page{label.page} {tag}=({x:.4f},{y:.4f},{w:.4f},{h:.4f}) "
                    "is outside [0, 1] page bounds"
                )
    return problems


def dump_label_set_json(label_set: LabelSet) -> str:
    """Canonical JSON text for a `LabelSet`: pydantic `mode="json"` dump, LF-terminated, matching
    `factory.erp.dump_erp_json`'s convention so both artifacts read the same way."""
    return json.dumps(label_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def write_label_set_json(label_set: LabelSet, path: Path) -> None:
    """Writes `label_set` to `path` as UTF-8 JSON with LF line endings."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dump_label_set_json(label_set))
