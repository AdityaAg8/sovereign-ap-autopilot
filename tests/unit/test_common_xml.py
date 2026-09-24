"""Unit tests for `apx.common.xml`: the shared hardened parser rejects an oversized payload, any
document declaring a DOCTYPE (which covers both XXE local-file-disclosure and billion-laughs
entity-expansion payloads), and malformed XML, while parsing an ordinary well-formed document
normally (PRD §8 standards, security fix for XXE / entity expansion)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from lxml import etree

from apx.common import xml as xml_module
from apx.common.xml import (
    MAX_XML_BYTES,
    XmlDoctypeRejectedError,
    XmlSyntaxRejectedError,
    XmlTooLargeError,
    parse_untrusted_xml,
)

_ORDINARY_XML = b'<?xml version="1.0" encoding="UTF-8"?>\n<root><child>value</child></root>'


def test_ordinary_document_parses_normally() -> None:
    root = parse_untrusted_xml(_ORDINARY_XML)
    assert etree.QName(root).localname == "root"
    assert root.findtext("child") == "value"


def test_oversized_payload_is_rejected_before_parsing() -> None:
    with pytest.raises(XmlTooLargeError):
        parse_untrusted_xml(_ORDINARY_XML, max_bytes=10)


def test_default_size_cap_is_five_megabytes() -> None:
    assert MAX_XML_BYTES == 5 * 1024 * 1024


def test_malformed_xml_without_doctype_is_rejected() -> None:
    with pytest.raises(XmlSyntaxRejectedError):
        parse_untrusted_xml(b"not xml at all")


def test_plain_doctype_with_no_entities_is_rejected_outright() -> None:
    payload = b'<?xml version="1.0"?>\n<!DOCTYPE root>\n<root/>'
    with pytest.raises(XmlDoctypeRejectedError):
        parse_untrusted_xml(payload)


def test_external_dtd_reference_is_rejected(tmp_path: Path) -> None:
    dtd_path = tmp_path / "evil.dtd"
    dtd_path.write_text("<!ENTITY harmless 'x'>", encoding="utf-8")
    payload = (
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE root SYSTEM "' + dtd_path.as_uri().encode("ascii") + b'">\n'
        b"<root/>"
    )
    with pytest.raises(XmlDoctypeRejectedError):
        parse_untrusted_xml(payload)


def test_xxe_local_file_disclosure_payload_never_discloses_file_content(tmp_path: Path) -> None:
    secret_path = tmp_path / "secret.txt"
    secret_marker = "TOP-SECRET-4f3c9b8a"
    secret_path.write_text(secret_marker, encoding="utf-8")
    payload = (
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "' + secret_path.as_uri().encode("ascii") + b'"> ]>\n'
        b"<foo>&xxe;</foo>"
    )

    with pytest.raises(XmlDoctypeRejectedError) as excinfo:
        parse_untrusted_xml(payload)

    assert secret_marker not in str(excinfo.value)


def _billion_laughs_payload() -> bytes:
    entities = ['<!ENTITY lol "lol">']
    for i in range(1, 10):
        expansion = f"&lol{i - 1};" if i > 1 else "&lol;"
        entities.append(f'<!ENTITY lol{i} "{expansion * 10}">')
    return (
        '<?xml version="1.0"?>\n'
        "<!DOCTYPE lolz [\n" + "\n".join(entities) + "\n]>\n<lolz>&lol9;</lolz>"
    ).encode("ascii")


def test_billion_laughs_payload_is_rejected_before_any_parser_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic proof of "before any expansion": lxml is never invoked for a DOCTYPE
    payload, so no entity can expand however large the declared tree is."""
    parser_calls: list[bytes] = []

    def _tripwire(data: bytes, *args: object, **kwargs: object) -> etree._Element:
        parser_calls.append(data)
        raise AssertionError("lxml must not see a DOCTYPE-bearing payload")

    monkeypatch.setattr(xml_module.etree, "fromstring", _tripwire)

    with pytest.raises(XmlDoctypeRejectedError):
        parse_untrusted_xml(_billion_laughs_payload())

    assert parser_calls == []


@pytest.mark.perf
def test_billion_laughs_payload_is_rejected_fast() -> None:
    """Wall-clock bound (opt-in `perf` marker: timing is load-dependent on a shared host)."""
    started = time.monotonic()
    with pytest.raises(XmlDoctypeRejectedError):
        parse_untrusted_xml(_billion_laughs_payload())
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, "billion-laughs payload must be rejected fast, before any expansion"
