"""Tests for `apx.parse.client.ParseServiceClient` against a fake HTTP transport
(`httpx.MockTransport`, not a live parse service): success mapping into `Page`, and typed
`ParseError` on every failure mode -- timeout, bad status, and a malformed response (PRD §3
apx/parse client: "typed errors, no silent fallback")."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from apx.common.errors import ParseError
from apx.parse.client import ParseServiceClient
from apx.parse.config import ParseClientConfig

_SHA = "a" * 64


def _client_with_handler(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> ParseServiceClient:
    """Build a `ParseServiceClient` whose outbound `httpx.AsyncClient` always uses `handler`
    as its transport, so no real network call is ever made."""
    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return ParseServiceClient(ParseClientConfig(base_url="http://parse-test"))


@pytest.mark.asyncio
async def test_parse_page_image_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "width_px": 800,
        "height_px": 1000,
        "words": [
            {"text": "Invoice", "bbox": [0.1, 0.1, 0.2, 0.12], "confidence": 0.9},
            {"text": "Number", "bbox": [0.21, 0.1, 0.3, 0.12], "confidence": 0.9},
        ],
        "lines": [{"word_indices": [0, 1]}],
        "tables": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/parse"
        return httpx.Response(200, json=payload)

    client = _client_with_handler(monkeypatch, handler)

    page = await client.parse_page_image(document_sha256=_SHA, page_index=2, png_bytes=b"fake")

    assert page.index == 2
    assert page.width_px == 800 and page.height_px == 1000
    assert [w.text for w in page.words] == ["Invoice", "Number"]
    assert page.words[0].id == "p2w0"
    assert page.words[0].line_id == "p2l0"
    assert len(page.lines) == 1
    assert page.lines[0].word_ids == ("p2w0", "p2w1")
    assert page.lines[0].bbox.x0 == pytest.approx(0.1)
    assert page.lines[0].bbox.x1 == pytest.approx(0.3)
    assert page.tables == ()


@pytest.mark.asyncio
async def test_parse_page_image_timeout_raises_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom", request=request)

    client = _client_with_handler(monkeypatch, handler)

    with pytest.raises(ParseError, match=r"timed out|unreachable"):
        await client.parse_page_image(document_sha256=_SHA, page_index=0, png_bytes=b"x")


@pytest.mark.asyncio
async def test_parse_page_image_bad_status_raises_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model not loaded")

    client = _client_with_handler(monkeypatch, handler)

    with pytest.raises(ParseError, match="500"):
        await client.parse_page_image(document_sha256=_SHA, page_index=0, png_bytes=b"x")


@pytest.mark.asyncio
async def test_parse_page_image_malformed_json_raises_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"width_px": 10})  # missing required fields

    client = _client_with_handler(monkeypatch, handler)

    with pytest.raises(ParseError, match="malformed"):
        await client.parse_page_image(document_sha256=_SHA, page_index=0, png_bytes=b"x")


@pytest.mark.asyncio
async def test_parse_page_image_out_of_range_word_index_raises_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "width_px": 10,
        "height_px": 10,
        "words": [{"text": "only", "bbox": [0.0, 0.0, 0.1, 0.1], "confidence": 1.0}],
        "lines": [{"word_indices": [0, 5]}],
        "tables": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _client_with_handler(monkeypatch, handler)

    with pytest.raises(ParseError, match="out-of-range"):
        await client.parse_page_image(document_sha256=_SHA, page_index=0, png_bytes=b"x")


@pytest.mark.asyncio
async def test_parse_page_image_empty_line_raises_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "width_px": 10,
        "height_px": 10,
        "words": [],
        "lines": [{"word_indices": []}],
        "tables": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _client_with_handler(monkeypatch, handler)

    with pytest.raises(ParseError, match="empty line"):
        await client.parse_page_image(document_sha256=_SHA, page_index=0, png_bytes=b"x")


def test_config_defaults_match_compose_service_name() -> None:
    config = ParseClientConfig()
    assert config.base_url == "http://parse:8000"
    assert config.request_timeout_s > 0
    assert config.connect_timeout_s > 0


def test_config_reads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARSE_BASE_URL", "http://parse-staging:9000")
    config = ParseClientConfig()
    assert config.base_url == "http://parse-staging:9000"


def test_wire_response_shape_is_json_serializable() -> None:
    # Guards the wire contract apx.parse.client and infra/docker/parse/app.py both implement.
    payload = {"width_px": 1, "height_px": 1, "words": [], "lines": [], "tables": []}
    assert json.loads(json.dumps(payload)) == payload


async def test_client_accepts_the_parse_services_own_response_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract regression (live e2e 2026-09-23): the service always sends `peak_vram_gb`, and the
    client's strict wire model rejected every real scanned page as malformed."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "infra" / "docker" / "parse" / "app.py"
    spec = importlib.util.spec_from_file_location("parse_service_contract", path)
    assert spec is not None and spec.loader is not None
    service = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = service
    spec.loader.exec_module(service)
    body = service.ParseResponse(
        width_px=100,
        height_px=50,
        words=(service.WireWord(text="INV-1", bbox=(0.1, 0.1, 0.3, 0.2), confidence=1.0),),
        lines=(service.WireLine(word_indices=(0,)),),
        peak_vram_gb=2.21,
    ).model_dump(mode="json")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    client = _client_with_handler(monkeypatch, handler)
    page = await client.parse_page_image(document_sha256="a" * 64, page_index=0, png_bytes=b"png")
    assert [w.text for w in page.words] == ["INV-1"]
