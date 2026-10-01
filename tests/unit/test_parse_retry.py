"""Unit tests for the eval harness's parse-service resilience (`evals.parse_cache.retrying`,
`evals.extraction_a`): a transient OCR timeout is retried with bounded backoff, a deterministic
failure is not, and a document that still fails becomes a typed, recorded, reported failure
instead of killing the run or vanishing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from apx.common.errors import ParseError
from apx.parse.interfaces import Page, ParsedDocument
from apx.parse.parser import PaddleOcrDocumentParser
from evals.extraction_a import (
    DocumentParseFailedError,
    _run_one,
    failures_path,
    render_report,
)
from evals.parse_cache import is_transient_parse_error, retrying
from tests.unit.test_scan_input import SHA, _FakeOcrClient, _StubExtractor, _write_doc

_REQUEST = httpx.Request("POST", "http://parse/parse")


def _chained(cause: BaseException, message: str = "parse service failed") -> ParseError:
    """A `ParseError` chained from its httpx cause, exactly as `apx.parse.client` raises it."""
    try:
        raise ParseError(message) from cause
    except ParseError as exc:
        return exc


def _timeout() -> ParseError:
    return _chained(
        httpx.ReadTimeout("read timed out", request=_REQUEST), "parse service timed out"
    )


def _status(code: int) -> ParseError:
    response = httpx.Response(code, request=_REQUEST)
    return _chained(httpx.HTTPStatusError("bad status", request=_REQUEST, response=response))


class _Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.waits.append(delay)


def test_transient_classification_reads_the_httpx_cause() -> None:
    assert is_transient_parse_error(_timeout())
    assert is_transient_parse_error(_chained(httpx.ConnectError("refused", request=_REQUEST)))
    assert is_transient_parse_error(_status(503))
    assert not is_transient_parse_error(_status(400))
    assert not is_transient_parse_error(ParseError("malformed response"))  # no httpx cause


@pytest.mark.asyncio
async def test_transient_failures_are_retried_with_the_given_backoff() -> None:
    document: Any = object()
    errors = [_timeout(), _status(502)]
    calls: list[str] = []

    async def parse(sha: str, _data: bytes) -> ParsedDocument:
        calls.append(sha)
        if errors:
            raise errors.pop(0)
        return document  # type: ignore[no-any-return]

    sleeps, logs = _Sleeps(), list[str]()
    wrapped = retrying(parse, backoff_s=(1.0, 2.0, 4.0), sleep=sleeps, log=logs.append)
    assert await wrapped("a" * 64, b"x") is document
    assert len(calls) == 3 and sleeps.waits == [1.0, 2.0]
    assert len(logs) == 2 and "retrying in 1 s" in logs[0]


@pytest.mark.asyncio
async def test_retries_are_bounded_and_the_last_error_propagates() -> None:
    calls = 0

    async def parse(_sha: str, _data: bytes) -> ParsedDocument:
        nonlocal calls
        calls += 1
        raise _timeout()

    sleeps = _Sleeps()
    with pytest.raises(ParseError, match="timed out"):
        await retrying(parse, backoff_s=(1.0, 2.0), sleep=sleeps, log=lambda _m: None)("a", b"")
    assert calls == 3 and sleeps.waits == [1.0, 2.0]


@pytest.mark.asyncio
async def test_deterministic_failures_are_not_retried() -> None:
    calls = 0

    async def parse(_sha: str, _data: bytes) -> ParsedDocument:
        nonlocal calls
        calls += 1
        raise _status(422)

    sleeps = _Sleeps()
    with pytest.raises(ParseError):
        await retrying(parse, backoff_s=(1.0,), sleep=sleeps, log=lambda _m: None)("a", b"")
    assert calls == 1 and sleeps.waits == []


class _FlakyOcrClient(_FakeOcrClient):
    """The OCR service timing out on its first `failures` page requests, then answering."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def parse_page_image(
        self, *, document_sha256: str, page_index: int, png_bytes: bytes
    ) -> Page:
        if self.failures:
            self.failures -= 1
            raise _timeout()
        return await super().parse_page_image(
            document_sha256=document_sha256, page_index=page_index, png_bytes=png_bytes
        )


@pytest.mark.asyncio
async def test_a_scan_that_times_out_once_still_gets_its_row(tmp_path: Path) -> None:
    _write_doc(tmp_path, has_scan=True)
    extractor: Any = _StubExtractor()
    flaky: Any = PaddleOcrDocumentParser(_FlakyOcrClient(failures=1))  # type: ignore[arg-type]
    steady: Any = PaddleOcrDocumentParser(_FakeOcrClient())  # type: ignore[arg-type]
    retried = await _run_one(SHA, tmp_path, extractor, flaky, None, (0.0,))
    assert retried == await _run_one(SHA, tmp_path, extractor, steady, None, (0.0,))


@pytest.mark.asyncio
async def test_a_scan_that_keeps_timing_out_is_a_typed_recorded_failure(tmp_path: Path) -> None:
    _write_doc(tmp_path, has_scan=True)
    extractor: Any = _StubExtractor()
    dead: Any = PaddleOcrDocumentParser(_FlakyOcrClient(failures=99))  # type: ignore[arg-type]
    with pytest.raises(DocumentParseFailedError) as info:
        await _run_one(SHA, tmp_path, extractor, dead, None, (0.0, 0.0))
    assert extractor.seen is None  # the model never ran on a missing parse
    record = info.value.record()
    assert record["sha"] == SHA and record["input_kind"] == "scan"
    assert record["error_type"] == "ParseError" and "timed out" in record["error"]

    report = render_report([], "extractor-b", failures=[record])
    assert "Parse failures (no row, after retries; the run fails): 1." in report
    assert SHA[:12] in report
    assert failures_path(Path("logs/x.jsonl")) == Path("logs/x.parse_failures.jsonl")
