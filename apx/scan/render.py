"""Rasterises PDF pages with pypdfium2 so `apx.scan.visibility` can compare what pdfplumber
parsed against what a human would actually see on the page (PRD §3, §6.3 A01-A04, M6.2)."""

from __future__ import annotations

from typing import Final

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray

RENDER_DPI: Final[int] = 200
"""Raster DPI for the render-vs-parse comparison. High enough to resolve anti-aliased glyph
strokes, low enough to keep a 3-page invoice well under the 1s/document CPU budget (M6.2)."""

RGBImage = NDArray[np.uint8]


def render_page(pdfium_page: pdfium.PdfPage, dpi: int = RENDER_DPI) -> RGBImage:
    """Rasterise one page to an (height, width, 3) uint8 RGB array at `dpi`.

    The array's pixel grid lines up with pdfplumber's `top`/`x0`-style coordinates (both are
    top-left origin) once scaled by `dpi / 72`.
    """
    scale = dpi / 72.0
    bitmap = pdfium_page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    return np.asarray(image, dtype=np.uint8)
