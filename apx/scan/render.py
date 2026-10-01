"""Rasterises PDF pages with pypdfium2 so `apx.scan.visibility` can compare what pdfplumber
parsed against what a human would actually see on the page (PRD §3, §6.3 A01-A04, M6.2). Refuses
any page over the `apx.common.raster` caps before allocating a bitmap (REVIEW H-03)."""

from __future__ import annotations

from typing import Final

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray

from apx.common.raster import MAX_RENDER_DPI, check_pdf_page

RENDER_DPI: Final[int] = MAX_RENDER_DPI
"""Raster DPI for the render-vs-parse comparison. High enough to resolve anti-aliased glyph
strokes, low enough to keep a 3-page invoice well under the 1s/document CPU budget (M6.2)."""

RGBImage = NDArray[np.uint8]


def render_page(pdfium_page: pdfium.PdfPage, dpi: int = RENDER_DPI) -> RGBImage:
    """Rasterise one page to an (height, width, 3) uint8 RGB array at `dpi`.

    The array's pixel grid lines up with pdfplumber's `top`/`x0`-style coordinates (both are
    top-left origin) once scaled by `dpi / 72`. Raises `RasterTooLargeError` (nothing rendered)
    if the page, its pixel count at `dpi` or an embedded image exceeds the raster caps.
    """
    scale = dpi / 72.0
    check_pdf_page(pdfium_page, scale=scale, where="render_page")
    bitmap = pdfium_page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    return np.asarray(image, dtype=np.uint8)
