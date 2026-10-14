"""What a factory document looks like at runtime (ADR-0012 amendment 3): a
born-digital invoice is its `<sha>.pdf`; a `has_scan` invoice is its Augraphy scan PNGs
(`<sha>.pageN.scan.png`) assembled into an image-only PDF with no text layer, the form a
multi-page scanned upload takes, so `apx.parse.parser` routes every page to the OCR service.
Shared by `evals.extraction_a` and `scripts/e2e_pipeline.py` so both feed the same bytes."""

from __future__ import annotations

import io
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from apx.common.errors import ApxError

InputKind = Literal["digital", "scan"]
INPUT_KINDS: Final[tuple[InputKind, ...]] = ("digital", "scan")

SCAN_DPI: Final[float] = 96.0
"""Resolution of the factory's scan PNGs: `factory.render` screenshots one A4 page in CSS pixels
(96 per inch), so at 96 dpi the assembled page is a physical A4 sheet, as a scanner writes it."""

_SCAN_PAGE = re.compile(r"\.page(\d+)\.scan\.png$")


class ScanInputError(ApxError):
    """A `has_scan` document's scan pages are missing, incomplete or undecodable."""


def input_kind(truth: Mapping[str, Any]) -> InputKind:
    """The runtime form of a factory document, from its truth file's `has_scan` flag."""
    has_scan = truth["has_scan"]
    if not isinstance(has_scan, bool):
        raise ScanInputError(f"truth has_scan must be a bool, got {has_scan!r}")
    return "scan" if has_scan else "digital"


def scan_page_paths(data_dir: Path, sha: str) -> list[Path]:
    """The scan PNGs of one document in page order; they must run 0..N-1 without a gap."""
    found: dict[int, Path] = {}
    for path in data_dir.glob(f"{sha}.page*.scan.png"):
        match = _SCAN_PAGE.search(path.name)
        if match is not None:
            found[int(match.group(1))] = path
    if not found:
        raise ScanInputError(f"{sha[:12]}: has_scan is true but {data_dir} has no scan pages")
    if sorted(found) != list(range(len(found))):
        raise ScanInputError(f"{sha[:12]}: scan pages {sorted(found)} are not 0..{len(found) - 1}")
    return [found[index] for index in range(len(found))]


def _pdf_page_count(data: bytes) -> int:
    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _rgb(png: bytes, page: int) -> Image.Image:
    try:
        return Image.open(io.BytesIO(png)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ScanInputError(f"scan page {page} does not decode: {exc}") from exc


def scan_pdf(pages_png: Sequence[bytes], *, dpi: float = SCAN_DPI) -> bytes:
    """An image-only PDF with one page per PNG, each image filling its page at `dpi`. Lossless
    (Flate-compressed RGB) and byte-deterministic, so a document keeps one sha256 across runs."""
    if not pages_png:
        raise ScanInputError("a scan PDF needs at least one page")
    n = len(pages_png)
    # Object numbers: 1 catalog, 2 page tree, then (page, content, image) per page.
    kids = " ".join(f"{3 + 3 * i} 0 R" for i in range(n))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("ascii"),
    ]
    for index, png in enumerate(pages_png):
        image = _rgb(png, index)
        width_px, height_px = image.size
        width_pt, height_pt = width_px * 72 / dpi, height_px * 72 / dpi
        content_obj, image_obj = 4 + 3 * index, 5 + 3 * index  # the page itself is 3 + 3 * index
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt:.4f} {height_pt:.4f}] "
            f"/Resources << /XObject << /Im0 {image_obj} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>".encode("ascii")
        )
        content = f"q {width_pt:.4f} 0 0 {height_pt:.4f} 0 0 cm /Im0 Do Q".encode("ascii")
        objects.append(_stream(b"", content))
        pixels = zlib.compress(image.tobytes(), 6)
        objects.append(
            _stream(
                f"/Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} "
                "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode ".encode("ascii"),
                pixels,
            )
        )
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    out += b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def _stream(dictionary: bytes, data: bytes) -> bytes:
    return (
        b"<< "
        + dictionary
        + f"/Length {len(data)} >>\nstream\n".encode("ascii")
        + data
        + b"\nendstream"
    )


def scan_document(data_dir: Path, sha: str) -> bytes:
    """The scanned upload of `sha`: its scan PNGs as one image-only PDF. The page count must
    match the born-digital PDF, so a lost scan page can never silently drop fields."""
    paths = scan_page_paths(data_dir, sha)
    digital_pages = _pdf_page_count((data_dir / f"{sha}.pdf").read_bytes())
    if len(paths) != digital_pages:
        raise ScanInputError(
            f"{sha[:12]}: {len(paths)} scan pages for a {digital_pages}-page invoice"
        )
    return scan_pdf([path.read_bytes() for path in paths])


def load_truth(data_dir: Path, sha: str) -> dict[str, Any]:
    truth: dict[str, Any] = json.loads((data_dir / f"{sha}.truth.json").read_text("utf-8"))
    return truth


def document_input(data_dir: Path, sha: str, truth: Mapping[str, Any]) -> tuple[InputKind, bytes]:
    """The bytes the runtime would receive for factory document `sha`, and their kind."""
    kind = input_kind(truth)
    if kind == "scan":
        return kind, scan_document(data_dir, sha)
    return kind, (data_dir / f"{sha}.pdf").read_bytes()
