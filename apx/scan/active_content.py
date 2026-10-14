"""Finds PDF content a rendered page never shows a human but a naive text-extraction step
could still read: Info/XMP metadata, annotations, embedded attachments and active content
(JavaScript, `/OpenAction`, `/AA`) (A10).

Metadata is inherently present on every PDF (Producer, CreationDate, ...), so flagging its mere
existence would fail the 750-clean-invoice gate. Info/XMP text is instead flagged only when it
carries a zero-width/bidi control character (`apx.scan.controls.SUSPICIOUS_CODEPOINTS`) or is
long enough that it is not a normal short metadata value; annotations, attachments and active
content are flagged on presence alone, since none of those belong in a generated invoice. A
baseline Info dict key (Producer, Creator, ...) only skips the length heuristic -- a suspicious
codepoint is never exempt there, since a writer name has no legitimate reason to carry one
(SEC-02). An XMP stream that cannot be decoded fails closed with `ScanError`; the scan never
treats an undecodable stream as if it were empty (SEC-02)."""

from __future__ import annotations

from typing import Any

from pdfminer.pdftypes import PDFStream, resolve1
from pdfplumber.pdf import PDF

from apx.common.errors import ScanError
from apx.scan.controls import SUSPICIOUS_CODEPOINTS
from apx.scan.interfaces import ScanFlag, ScanFlagKind
from apx.scan.text import build_evidence

_METADATA_LENGTH_LIMIT = 80
"""Above this, an Info dict value reads as free text rather than a short label (A10). Applied
only to non-baseline keys -- see `_METADATA_BASELINE_KEYS`."""

_METADATA_BASELINE_KEYS = frozenset({"Producer", "Creator", "CreationDate", "ModDate", "Trapped"})
"""Info dict keys every PDF writer (reportlab, browsers, Office) sets unattended, often with a
value longer than `_METADATA_LENGTH_LIMIT` (e.g. "Skia/PDF m120 Google Docs Renderer"). Exempt
from the length heuristic so the scan does not fire on every clean invoice, but still scanned for
suspicious codepoints (F2): no legitimate writer stitches a zero-width or bidi character into its
own name."""

_XMP_LENGTH_LIMIT = 4000
"""A templated XMP packet from a normal generator is a few hundred bytes; well past this the
stream is carrying more than schema boilerplate."""


def scan_document_active_content(pdf: PDF) -> list[ScanFlag]:
    """Flag suspicious metadata, annotations, attachments and active content anywhere in the
    document. Document-level findings (metadata, attachments, JS/OpenAction/AA) get `page=None`;
    annotations keep their page. Never raises for a clean document (returns [])."""
    catalog = resolve1(pdf.doc.catalog)
    flags: list[ScanFlag] = []
    flags.extend(_scan_info_metadata(pdf.metadata))
    flags.extend(_scan_annotations(pdf))
    flags.extend(_scan_xmp(catalog))
    flags.extend(_scan_attachments(catalog))
    flags.extend(_scan_active_actions(catalog, pdf))
    return flags


def _metadata_flag(excerpt: str, page: int | None = None) -> ScanFlag:
    return ScanFlag(
        kind=ScanFlagKind.PDF_METADATA,
        page=page,
        score=1.0,
        evidence=build_evidence(ScanFlagKind.PDF_METADATA, excerpt),
    )


def _active_flag(excerpt: str, page: int | None = None) -> ScanFlag:
    return ScanFlag(
        kind=ScanFlagKind.PDF_JAVASCRIPT,
        page=page,
        score=1.0,
        evidence=build_evidence(ScanFlagKind.PDF_JAVASCRIPT, excerpt),
    )


def _has_suspicious_codepoint(text: str) -> bool:
    return any(ord(c) in SUSPICIOUS_CODEPOINTS for c in text)


def _scan_info_metadata(metadata: dict[str, Any]) -> list[ScanFlag]:
    flags = []
    for key, value in metadata.items():
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        too_long = key not in _METADATA_BASELINE_KEYS and len(text) > _METADATA_LENGTH_LIMIT
        if _has_suspicious_codepoint(text) or too_long:
            flags.append(_metadata_flag(f"Info/{key}: {text}"))
    return flags


def _scan_annotations(pdf: PDF) -> list[ScanFlag]:
    flags = []
    for annot in pdf.annots:
        contents = annot.get("contents")
        if not contents or not str(contents).strip():
            continue
        page_index = int(annot["page_number"]) - 1
        flags.append(_metadata_flag(f"annotation: {contents}", page=page_index))
    return flags


def _scan_xmp(catalog: dict[str, Any]) -> list[ScanFlag]:
    stream = resolve1(catalog.get("Metadata"))
    if not isinstance(stream, PDFStream):
        return []
    try:
        raw = stream.get_data()
    except Exception as exc:
        raise ScanError(f"XMP metadata stream could not be decoded: {exc}") from exc
    text = raw.decode("utf-8", errors="replace")
    if _has_suspicious_codepoint(text) or len(text) > _XMP_LENGTH_LIMIT:
        return [_metadata_flag(f"XMP metadata ({len(text)} chars): {text}")]
    return []


def _decode_name(value: object) -> str:
    resolved = resolve1(value)
    if isinstance(resolved, bytes):
        return resolved.decode("utf-8", "replace")
    return str(resolved) if resolved is not None else ""


def _scan_attachments(catalog: dict[str, Any]) -> list[ScanFlag]:
    names = resolve1(catalog.get("Names"))
    if not isinstance(names, dict):
        return []
    embedded = resolve1(names.get("EmbeddedFiles"))
    if not isinstance(embedded, dict):
        return []
    raw_names = resolve1(embedded.get("Names")) or []
    filenames = [_decode_name(v) for v in raw_names[::2]]  # a name tree alternates name, value
    joined = ", ".join(n for n in filenames if n)
    excerpt = f"embedded file(s): {joined}" if joined else "embedded file attachment"
    return [
        ScanFlag(
            kind=ScanFlagKind.PDF_ATTACHMENT,
            page=None,
            score=1.0,
            evidence=build_evidence(ScanFlagKind.PDF_ATTACHMENT, excerpt),
        )
    ]


def _scan_active_actions(catalog: dict[str, Any], pdf: PDF) -> list[ScanFlag]:
    flags = []
    if resolve1(catalog.get("OpenAction")) is not None:
        flags.append(_active_flag("document /OpenAction present"))
    if resolve1(catalog.get("AA")) is not None:
        flags.append(_active_flag("document /AA (additional actions) present"))

    names = resolve1(catalog.get("Names"))
    if isinstance(names, dict) and resolve1(names.get("JavaScript")) is not None:
        flags.append(_active_flag("document /Names/JavaScript entries present"))

    for page in pdf.pages:
        attrs = getattr(page.page_obj, "attrs", None) or {}
        if resolve1(attrs.get("AA")) is not None:
            page_index = page.page_number - 1
            flags.append(
                _active_flag(
                    f"page {page.page_number} /AA (additional actions) present", page=page_index
                )
            )
    return flags
