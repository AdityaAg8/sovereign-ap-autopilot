"""attacks/pdf_metadata.py: injects a payload into a rendered invoice's PDF /Info dictionary
(Keywords, Subject) and a newly attached XMP metadata stream on the document catalog (PRD §10
M6.3 "PDF metadata/XMP injection"; docs/security.md A10). No PDF-editing library (pypdf,
pikepdf) is available in this project's environment, so this rewrites the classic cross-
reference table directly: Chromium's `page.pdf()` output is a simple, single-generation,
classic-xref PDF (verified empirically -- no object streams), so the original file's bytes are
left completely untouched and only three new objects (an updated catalog, an updated Info
dictionary, and the new XMP stream) are appended, with a freshly written xref table pointing
existing objects at their original, unmoved offsets and the three touched ones at their new,
appended offsets. Fully deterministic: nothing here reads the clock or any randomness."""

from __future__ import annotations

import re
from dataclasses import dataclass

_OBJ_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj\b")
_XREF_ENTRY_RE = re.compile(rb"(\d{10}) (\d{5}) ([nf])\s*\r?\n?")
_XREF_SUBSECTION_HEADER_RE = re.compile(rb"(\d+) (\d+)\r?\n")


class PdfStructureError(ValueError):
    """Raised when `pdf_bytes` is not the simple, classic-xref, single-generation PDF this
    module assumes (PRD §8: no silent fallbacks on the money/document path)."""


def _parse_xref(pdf_bytes: bytes, *, xref_start: int) -> dict[int, tuple[int, int, bytes]]:
    """Parses one classic xref section starting right after the `xref` keyword into
    `{object_number: (offset, generation, b"n" | b"f")}`, following however many `start
    count` subsections it declares (PRD §10 M6.3: Chromium always writes a single `0 N`
    subsection in practice, but nothing here assumes exactly one)."""
    entries: dict[int, tuple[int, int, bytes]] = {}
    cursor = xref_start
    while True:
        header = _XREF_SUBSECTION_HEADER_RE.match(pdf_bytes, cursor)
        if header is None:
            break
        start, count = int(header.group(1)), int(header.group(2))
        cursor = header.end()
        for i in range(count):
            entry = _XREF_ENTRY_RE.match(pdf_bytes, cursor)
            if entry is None:
                raise PdfStructureError(
                    f"attacks.pdf_metadata: malformed xref entry for object {start + i}"
                )
            entries[start + i] = (int(entry.group(1)), int(entry.group(2)), entry.group(3))
            cursor = entry.end()
    if not entries:
        raise PdfStructureError("attacks.pdf_metadata: xref section had no entries")
    return entries


def _parse_trailer(pdf_bytes: bytes) -> tuple[int, int | None]:
    """`(root_object_number, info_object_number)` from the `trailer` dictionary. `/Info` is
    optional per the PDF spec, though Chromium always writes one."""
    match = re.search(rb"trailer\b(.*?)startxref", pdf_bytes, re.DOTALL)
    if match is None:
        raise PdfStructureError("attacks.pdf_metadata: no trailer dictionary found")
    body = match.group(1)
    root_match = re.search(rb"/Root\s+(\d+)\s+0\s+R", body)
    if root_match is None:
        raise PdfStructureError("attacks.pdf_metadata: trailer has no /Root")
    info_match = re.search(rb"/Info\s+(\d+)\s+0\s+R", body)
    return int(root_match.group(1)), (int(info_match.group(1)) if info_match else None)


def _extract_dict_bytes(pdf_bytes: bytes, *, offset: int, obj_num: int) -> bytes:
    """The full `<< ... >>` dictionary bytes (including nested dictionaries such as
    `/ViewerPreferences`) for the object declared at `offset`, found by depth-counting `<<`/`>>`
    rather than a first-match regex, since Chromium's catalog nests one."""
    header = _OBJ_RE.match(pdf_bytes, offset)
    if header is None or int(header.group(1)) != obj_num:
        raise PdfStructureError(f"attacks.pdf_metadata: object {obj_num} not found at {offset}")
    start = pdf_bytes.index(b"<<", header.end())
    depth = 0
    i = start
    while i < len(pdf_bytes):
        if pdf_bytes[i : i + 2] == b"<<":
            depth += 1
            i += 2
        elif pdf_bytes[i : i + 2] == b">>":
            depth -= 1
            i += 2
            if depth == 0:
                return pdf_bytes[start:i]
        else:
            i += 1
    raise PdfStructureError(f"attacks.pdf_metadata: unbalanced dictionary for object {obj_num}")


def _pdf_text_string(value: str) -> bytes:
    """A PDF text string literal for `value`, as a UTF-16BE hex string with a byte-order mark
    (ISO 32000-1 §7.9.2.2), which is valid for any Unicode text -- including the Arabic payload
    variants -- and needs no escaping."""
    encoded = value.encode("utf-16-be")
    return b"<FEFF" + encoded.hex().encode("ascii").upper() + b">"


_XMP_BOM = chr(0xFEFF)  # required at the start of an XMP packet's `begin` attribute


def _xmp_packet(payload: str) -> bytes:
    """A minimal, valid XMP packet (Adobe XMP Specification Part 3) carrying `payload` in
    `dc:description`, the field a naive LLM-based PDF summarizer is most likely to surface."""
    escaped = payload.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = (
        f'<?xpacket begin="{_XMP_BOM}" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escaped}'
        "</rdf:li></rdf:Alt></dc:description>"
        "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return xml.encode("utf-8")


@dataclass
class _Parsed:
    """The pieces every function in this module needs from `pdf_bytes`, parsed once: where the
    original body ends (`xref_pos`), every original object's `(offset, generation, kind)`, and
    the trailer's `/Root`/`/Info` object numbers. `max_obj_num` is mutated by
    `_new_info_object` when it has to allocate a fresh object number."""

    xref_pos: int
    xref_entries: dict[int, tuple[int, int, bytes]]
    root_num: int
    info_num: int | None
    max_obj_num: int


def _parse(pdf_bytes: bytes) -> _Parsed:
    try:
        xref_pos = pdf_bytes.rindex(b"\nxref\n") + 1  # start of the "xref" keyword itself
    except ValueError as exc:
        raise PdfStructureError("attacks.pdf_metadata: no classic xref table found") from exc
    xref_entries = _parse_xref(pdf_bytes, xref_start=xref_pos + len(b"xref\n"))
    root_num, info_num = _parse_trailer(pdf_bytes)
    if root_num not in xref_entries:
        raise PdfStructureError(f"attacks.pdf_metadata: /Root {root_num} not in xref")
    return _Parsed(
        xref_pos=xref_pos,
        xref_entries=xref_entries,
        root_num=root_num,
        info_num=info_num,
        max_obj_num=max(xref_entries),
    )


def _new_info_object(parsed: _Parsed, pdf_bytes: bytes) -> tuple[int, bytes]:
    """The Info object's number and its current dict bytes (or an empty dict at a freshly
    allocated number if the PDF has no /Info at all)."""
    if parsed.info_num is not None and parsed.info_num in parsed.xref_entries:
        info_dict = _extract_dict_bytes(
            pdf_bytes, offset=parsed.xref_entries[parsed.info_num][0], obj_num=parsed.info_num
        )
        return parsed.info_num, info_dict
    parsed.max_obj_num += 1
    return parsed.max_obj_num, b"<<>>"


def _finalize(
    pdf_bytes: bytes, parsed: _Parsed, *, appended: dict[int, bytes], info_num: int
) -> bytes:
    """Appends each `{object_number: dict_or_stream_content}` in `appended` (a brand new number
    or an existing one being redefined) after the original, untouched body, then writes a fresh
    classic xref table: every object keeps its original offset unless `appended` redefines it,
    and the trailer's `/Info` points at `info_num` (which may itself be new)."""
    body = pdf_bytes[: parsed.xref_pos]
    new_offsets: dict[int, int] = {}
    for obj_num, content in appended.items():
        new_offsets[obj_num] = len(body)
        body += f"{obj_num} 0 obj\n".encode("ascii") + content + b"\nendobj\n"

    size = max(parsed.max_obj_num, max(appended, default=0)) + 1
    lines = [f"xref\n0 {size}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for obj_num in range(1, size):
        if obj_num in new_offsets:
            offset, generation = new_offsets[obj_num], 0
        elif obj_num in parsed.xref_entries:
            offset, generation, _kind = parsed.xref_entries[obj_num]
        else:
            raise PdfStructureError(f"attacks.pdf_metadata: no offset for object {obj_num}")
        lines.append(f"{offset:010d} {generation:05d} n \n".encode("ascii"))
    trailer = (
        f"trailer\n<</Size {size}/Root {parsed.root_num} 0 R/Info {info_num} 0 R>>\n"
        f"startxref\n{len(body)}\n%%EOF\n"
    ).encode("ascii")
    return body + b"".join(lines) + trailer


def inject(pdf_bytes: bytes, *, payload: str) -> bytes:
    """Returns `pdf_bytes` with `payload` injected into the /Info dictionary's /Keywords and
    /Subject and into a new XMP metadata stream attached to the catalog's /Metadata entry
    (PRD §10 M6.3 "PDF metadata/XMP injection"). Deterministic and pure: same
    `(pdf_bytes, payload)` always produces the same output bytes."""
    parsed = _parse(pdf_bytes)
    catalog_dict = _extract_dict_bytes(
        pdf_bytes, offset=parsed.xref_entries[parsed.root_num][0], obj_num=parsed.root_num
    )
    info_num, info_dict = _new_info_object(parsed, pdf_bytes)

    parsed.max_obj_num += 1
    xmp_num = parsed.max_obj_num
    xmp_bytes = _xmp_packet(payload)
    keyword_string = _pdf_text_string(payload)

    appended = {
        xmp_num: (
            b"<</Type /Metadata /Subtype /XML /Length "
            + str(len(xmp_bytes)).encode("ascii")
            + b">>\nstream\n"
            + xmp_bytes
            + b"\nendstream"
        ),
        info_num: (
            info_dict[:-2] + b"/Keywords " + keyword_string + b"/Subject " + keyword_string + b">>"
        ),
        parsed.root_num: catalog_dict[:-2] + f"/Metadata {xmp_num} 0 R>>".encode("ascii"),
    }
    return _finalize(pdf_bytes, parsed, appended=appended, info_num=info_num)


def inject_info_field(pdf_bytes: bytes, *, field_name: str, value: str) -> bytes:
    """Returns `pdf_bytes` with one custom `/{field_name}` key added to the /Info dictionary,
    carrying `value` verbatim (any Unicode codepoint, including zero-width and bidi-override
    control characters -- PDF text strings are plain data, never shaped or rendered).

    Used by `zero_width` and `bidi_override` instead of an HTML/CSS mutation: verified
    empirically (see this module's docstring reasoning in `inject`) that Chromium's print
    pipeline drops any character with no visible glyph -- including U+200B-U+200D and
    U+202C/U+202E -- before it reaches the exported PDF's page content stream, so injecting
    them as ordinary document text never survives to be extracted. The /Info dictionary has no
    such glyph pipeline in the way, so the exact codepoints always round-trip."""
    parsed = _parse(pdf_bytes)
    info_num, info_dict = _new_info_object(parsed, pdf_bytes)
    field_key = f"/{field_name} ".encode("ascii")
    appended = {info_num: info_dict[:-2] + field_key + _pdf_text_string(value) + b">>"}
    return _finalize(pdf_bytes, parsed, appended=appended, info_num=info_num)
