"""Shared hardened XML parsing for every lxml/Saxon entry point that touches request- or
document-derived bytes (`apx.asp_mock`, `apx.einvoice`): rejects any document containing a
`<!DOCTYPE` marker outright before parsing (removing both the XXE local-file-disclosure and
billion-laughs entity-expansion attack classes at the source, rather than trusting parser flags
alone), enforces a size cap, and never resolves entities, loads a DTD or touches the network even
for a document that somehow slips past the DOCTYPE scan (security fix)."""

from __future__ import annotations

import re

from lxml import etree  # type: ignore[import-untyped]

from apx.common.errors import ApxError

MAX_XML_BYTES = 5 * 1024 * 1024
"""Size cap for an untrusted XML document (5 MB); larger payloads are rejected before parsing."""

_DOCTYPE_RE = re.compile(rb"<!DOCTYPE", re.IGNORECASE)
"""Matches a DOCTYPE marker anywhere in the raw bytes. Both external-entity XXE and internal
entity-expansion ("billion laughs") payloads require a `<!DOCTYPE ...>` declaration per the XML
grammar, so refusing any document containing one -- checked on the raw bytes, before the
document is handed to any parser -- removes both attack classes regardless of parser flags or
library version behaviour."""


class XmlTooLargeError(ApxError):
    """Untrusted XML payload exceeds `MAX_XML_BYTES`."""


class XmlDoctypeRejectedError(ApxError):
    """Untrusted XML declares a DOCTYPE; rejected outright, before parsing (XXE / entity-
    expansion hardening)."""


class XmlSyntaxRejectedError(ApxError):
    """Untrusted XML is not well-formed."""


def _hardened_parser() -> etree.XMLParser:
    """A fresh hardened `lxml.etree.XMLParser`: never resolves entities, never loads a DTD or
    validates against one, never touches the network, and keeps libxml2's default
    (`huge_tree=False`) limits on document size/entity-expansion amplification. Callers must
    still reject any DOCTYPE-bearing document before parsing (see `parse_untrusted_xml`); these
    flags are defense in depth, not the primary control."""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        huge_tree=False,
    )


def parse_untrusted_xml(data: bytes, *, max_bytes: int = MAX_XML_BYTES) -> etree._Element:
    """Parses `data` -- bytes from an untrusted source (an HTTP request body, a submitted
    document) -- into an lxml `Element`, the single hardened entry point for every module in
    scope. Raises `XmlTooLargeError` if `data` exceeds `max_bytes`, `XmlDoctypeRejectedError` if
    it contains a `<!DOCTYPE` marker anywhere (checked before any parsing occurs), or
    `XmlSyntaxRejectedError` if it is not well-formed XML once parsed with the hardened parser."""
    if len(data) > max_bytes:
        raise XmlTooLargeError(
            f"XML payload is {len(data)} bytes, exceeds the {max_bytes} byte cap"
        )
    if _DOCTYPE_RE.search(data) is not None:
        raise XmlDoctypeRejectedError(
            "XML document declares a DOCTYPE; rejected outright without parsing"
        )
    try:
        root: etree._Element = etree.fromstring(data, parser=_hardened_parser())
    except etree.XMLSyntaxError as exc:
        raise XmlSyntaxRejectedError(f"document is not well-formed XML: {exc}") from exc
    return root


__all__ = [
    "MAX_XML_BYTES",
    "XmlDoctypeRejectedError",
    "XmlSyntaxRejectedError",
    "XmlTooLargeError",
    "parse_untrusted_xml",
]
