"""Unit tests for `factory.scan`: the Augraphy scan pipeline preserves pixel dimensions and is
deterministic per seed within a single process (the specific pitfall Augraphy's own
reproducibility notes call out), and the box-to-raster remap is a trivial page-fraction scale."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from factory.scan import map_bbox_to_pixels, scan_page


def _synthetic_page_png(width: int = 240, height: int = 160) -> bytes:
    """A small white page with a black text-like block, standing in for a real invoice page
    screenshot without needing a browser."""
    array = np.full((height, width, 3), 250, dtype=np.uint8)
    array[20:40, 20:180] = 0
    array[60:70, 20:120] = 0
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_scan_page_preserves_pixel_dimensions() -> None:
    original = _synthetic_page_png(240, 160)
    scanned = scan_page(original, seed=1)
    with Image.open(BytesIO(scanned)) as image:
        assert image.size == (240, 160)


def test_scan_page_is_deterministic_for_the_same_seed_in_one_process() -> None:
    """Augraphy's reproducibility notes warn that a seed only reproduces output when it is reset
    immediately before each use; `factory.scan._build_pipeline` does this per call, so two calls
    with the same seed in the same process (not just across separate script runs) must match."""
    original = _synthetic_page_png()
    first = scan_page(original, seed=99)
    second = scan_page(original, seed=99)
    assert first == second


def test_scan_page_differs_for_a_different_seed() -> None:
    original = _synthetic_page_png()
    a = scan_page(original, seed=1)
    b = scan_page(original, seed=2)
    assert a != b


@pytest.mark.parametrize(
    ("bbox", "width", "height", "expected"),
    [
        ((0.25, 0.5, 0.1, 0.2), 200, 100, (50, 50, 20, 20)),
        ((0.0, 0.0, 1.0, 1.0), 300, 150, (0, 0, 300, 150)),
    ],
)
def test_map_bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    expected: tuple[int, int, int, int],
) -> None:
    assert map_bbox_to_pixels(bbox, image_width=width, image_height=height) == expected
