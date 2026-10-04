"""Unit tests for the shared parse cache (`evals.parse_cache`): a cached parse equals an inline one
and the extraction harness writes identical rows from either, a hit never calls the parser, the
parser identity tracks the parse sources (so a parse fix invalidates the cache) and broken entries
fail loudly instead of re-parsing silently."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

from apx.common.errors import ParsedArtifactIntegrityError, ParseError
from apx.parse.parser import PaddleOcrDocumentParser
from evals.extraction_a import _run_one, parse_input
from evals.parse_cache import (
    MAX_CONSECUTIVE_TRANSIENT_FAILURES,
    PARSER_SOURCE_DIRS,
    REPO_ROOT,
    Contacted,
    ParseCache,
    ParseCacheError,
    ParseCacheMissError,
    ParseServiceDownError,
    cache_key,
    failure_path,
    open_cache,
    parser_identity,
    warm,
)
from evals.scan_input import document_input, load_truth
from tests.unit.test_scan_input import (
    SHA,
    _FakeOcrClient,
    _RaisingClient,
    _StubExtractor,
    _write_doc,
)

PARSER_ID = "0123456789abcdef"


@pytest.mark.asyncio
@pytest.mark.parametrize("has_scan", [True, False], ids=["scan", "digital"])
async def test_cached_parse_and_rows_equal_the_uncached_ones(
    tmp_path: Path, has_scan: bool
) -> None:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    _write_doc(data_dir, has_scan=has_scan, pages=2)
    _, data = document_input(data_dir, SHA, load_truth(data_dir, SHA))
    cache = ParseCache(tmp_path / "cache", PARSER_ID)

    inline = await parse_input(data, PaddleOcrDocumentParser(_FakeOcrClient()), None)  # type: ignore[arg-type]
    first = await parse_input(data, PaddleOcrDocumentParser(_FakeOcrClient()), cache)  # type: ignore[arg-type]
    # A hit never reaches the parse service (or pdfplumber's caller): the raising client proves it.
    hit = await parse_input(data, PaddleOcrDocumentParser(_RaisingClient()), cache)  # type: ignore[arg-type]
    assert inline == first == hit
    assert ParseCache(tmp_path / "cache", PARSER_ID).lookup(cache_key(inline.document_sha256, data))

    extractor: Any = _StubExtractor()
    live: Any = PaddleOcrDocumentParser(_FakeOcrClient())  # type: ignore[arg-type]
    offline: Any = PaddleOcrDocumentParser(_RaisingClient())  # type: ignore[arg-type]
    uncached_row = await _run_one(SHA, data_dir, extractor, live)
    cached_row = await _run_one(SHA, data_dir, extractor, offline, cache)
    assert cached_row == uncached_row


@pytest.mark.asyncio
async def test_warm_parses_each_input_once(tmp_path: Path) -> None:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    _write_doc(data_dir, has_scan=True)
    _, data = document_input(data_dir, SHA, load_truth(data_dir, SHA))
    client = _FakeOcrClient()
    parser = PaddleOcrDocumentParser(client)  # type: ignore[arg-type]
    cache = ParseCache(tmp_path / "cache", PARSER_ID)
    logs: list[str] = []
    inputs = [(SHA, hashlib.sha256(data).hexdigest(), data)]
    assert (await warm(cache, inputs, parser.parse, logs.append)).parsed == 1
    assert (await warm(cache, inputs, parser.parse, logs.append)).parsed == 0
    assert len(client.calls) == 1 and "parsed" in logs[0] and "cached" in logs[1]


def test_cache_key_names_the_bytes_and_a_foreign_document_id() -> None:
    data = b"%PDF-1.4 x"
    own = cache_key("0" * 64, data)
    assert own.endswith("." + "0" * 64) and own != cache_key("1" * 64, data)
    digest = hashlib.sha256(data).hexdigest()
    assert cache_key(digest, data) == digest


def _copy_parser_sources(root: Path) -> None:
    for rel in PARSER_SOURCE_DIRS:
        shutil.copytree(REPO_ROOT / rel, root / rel, ignore=shutil.ignore_patterns("__pycache__"))


def test_parser_identity_changes_with_any_parse_source(tmp_path: Path) -> None:
    _copy_parser_sources(tmp_path)
    before = parser_identity(tmp_path)
    assert before == parser_identity(REPO_ROOT) and len(before) == 16
    source = tmp_path / "apx" / "parse" / "digital.py"
    text = source.read_bytes()
    source.write_bytes(text.replace(b"\n", b"\r\n"))  # a CRLF checkout is the same parser
    assert parser_identity(tmp_path) == before
    source.write_bytes(text + b"\n# arabic text path fix\n")
    assert parser_identity(tmp_path) != before
    source.write_bytes(text)
    app = tmp_path / "infra" / "docker" / "parse" / "app.py"
    app.write_bytes(app.read_bytes() + b"\nX = 1\n")  # e.g. a new max-new-tokens budget
    assert parser_identity(tmp_path) != before
    shutil.rmtree(tmp_path / "infra")
    with pytest.raises(ParseCacheError, match="no parser sources"):
        parser_identity(tmp_path)


@pytest.mark.asyncio
async def test_another_parser_build_never_reads_old_parses(tmp_path: Path) -> None:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    _write_doc(data_dir, has_scan=False)
    data = (data_dir / f"{SHA}.pdf").read_bytes()
    parser = PaddleOcrDocumentParser(_RaisingClient())  # type: ignore[arg-type]
    await parse_input(data, parser, ParseCache(tmp_path / "cache", PARSER_ID))
    other = ParseCache(tmp_path / "cache", "fedcba9876543210")
    key = cache_key(hashlib.sha256(data).hexdigest(), data)
    assert key not in other and other.lookup(key) is None
    with pytest.raises(ParseCacheMissError, match="run the parse phase first"):
        other.get(key)
    assert ParseCache(tmp_path / "cache").parser_id == parser_identity()


@pytest.mark.asyncio
async def test_a_pinned_snapshot_answers_hits_and_never_parses_or_writes(tmp_path: Path) -> None:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    _write_doc(data_dir, has_scan=False)
    data = (data_dir / f"{SHA}.pdf").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    old = PaddleOcrDocumentParser(_FakeOcrClient())  # type: ignore[arg-type]
    parsed = await ParseCache(tmp_path / "cache", PARSER_ID).parse(sha, data, old.parse)
    logs: list[str] = []

    pinned = open_cache(PARSER_ID, tmp_path / "cache", log=logs.append)

    assert pinned.read_only and pinned.parser_id == PARSER_ID
    assert logs and PARSER_ID in logs[0] and parser_identity() in logs[0]

    async def never(_sha: str, _data: bytes) -> Any:
        raise AssertionError("a pinned snapshot must never parse")

    assert await pinned.parse(sha, data, never) == parsed
    other = data + b"\n%"
    with pytest.raises(ParseCacheMissError, match="pinned parse cache"):
        await pinned.parse(hashlib.sha256(other).hexdigest(), other, never)
    with pytest.raises(ParseCacheError, match="read-only"):
        pinned.put("abc", parsed)
    with pytest.raises(ParseCacheMissError, match="pinned parse cache"):
        await pinned.fill([("abc", data_dir / f"{SHA}.pdf")], never)
    assert sorted(p.name for p in (tmp_path / "cache" / PARSER_ID / "index").iterdir()) == [
        f"{cache_key(sha, data)}.json"
    ]


def test_open_cache_pins_only_an_existing_snapshot_of_another_build(tmp_path: Path) -> None:
    current = open_cache(None, tmp_path, log=pytest.fail)
    assert current.parser_id == parser_identity() and not current.read_only
    same = open_cache(parser_identity(), tmp_path, log=pytest.fail)
    assert not same.read_only
    with pytest.raises(ParseCacheError, match="no parse cache snapshot"):
        open_cache(PARSER_ID, tmp_path)
    with pytest.raises(ParseCacheError, match="16 lowercase hex"):
        open_cache("../escape", tmp_path)


@pytest.mark.asyncio
async def test_broken_entries_and_keys_fail_loudly(tmp_path: Path) -> None:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    _write_doc(data_dir, has_scan=False)
    data = (data_dir / f"{SHA}.pdf").read_bytes()
    cache = ParseCache(tmp_path / "cache", PARSER_ID)
    document = await cache.parse(SHA, data, PaddleOcrDocumentParser(_RaisingClient()).parse)  # type: ignore[arg-type]
    key = cache_key(SHA, data)
    cache.put(key, document)  # a second writer of the same key writes the same entry
    assert cache.get(key) == document
    assert not [p for p in cache.directory.rglob("*.tmp")]
    manifest = next(cache.directory.rglob("document.json"))
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ParsedArtifactIntegrityError):
        cache.lookup(key)
    (cache.directory / "index" / f"{key}.json").write_text("{", encoding="utf-8")
    with pytest.raises(ParseCacheError, match="malformed"):
        cache.lookup(key)
    for bad in ("../escape", "", "a/b", "." + "a" * 10):
        with pytest.raises(ParseCacheError, match="invalid parse cache key"):
            _ = bad in cache
    assert os.path.isdir(cache.directory / "store")


def test_split_inputs_builds_one_document_per_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evals.scan_input as scan_input
    from evals.parse_cache import SplitInputs

    built: list[str] = []

    def fake_document_input(data_dir: Path, sha: str, truth: object) -> tuple[str, bytes]:
        built.append(sha)
        return "digital", sha.encode()

    monkeypatch.setattr(scan_input, "document_input", fake_document_input)
    monkeypatch.setattr(scan_input, "load_truth", lambda data_dir, sha: {})
    inputs = SplitInputs(tmp_path, ["a", "b", "c"])
    assert len(inputs) == 3 and built == []
    assert inputs[1] == ("b", hashlib.sha256(b"b").hexdigest(), b"b")
    assert built == ["b"]
    assert len(inputs[:2]) == 2 and built == ["b"]


def _timeout() -> ParseError:
    try:
        raise httpx.ReadTimeout("page timed out after 240 s")
    except httpx.ReadTimeout as cause:
        error = ParseError("parse service request failed")
        error.__cause__ = cause
        return error


def _inputs(count: int) -> list[tuple[str, str, bytes]]:
    out = []
    for i in range(count):
        data = f"document {i}".encode()
        sha = hashlib.sha256(data).hexdigest()
        out.append((sha, sha, data))
    return out


@pytest.mark.asyncio
async def test_warm_records_a_failed_document_continues_and_a_rerun_retries_it(
    tmp_path: Path,
) -> None:
    """One document whose page times out (the 09-29 queue death at 191/750) is recorded with its
    error and the pass goes on; the next pass retries only that document and clears the record."""
    inputs = _inputs(3)
    bad = inputs[1][1]
    real = PaddleOcrDocumentParser(_FakeOcrClient())  # type: ignore[arg-type]
    template = None
    calls: list[str] = []
    failing = {bad}

    async def parse(document_sha256: str, data: bytes) -> Any:
        nonlocal template
        calls.append(document_sha256)
        if document_sha256 in failing:
            raise _timeout()
        if template is None:
            data_dir = tmp_path / "uae"
            data_dir.mkdir()
            _write_doc(data_dir, has_scan=False)
            _, pdf = document_input(data_dir, SHA, load_truth(data_dir, SHA))
            template = await real.parse(SHA, pdf)
        return template.model_copy(update={"document_sha256": document_sha256})

    cache = ParseCache(tmp_path / "cache", PARSER_ID)
    logs: list[str] = []
    first = await warm(cache, inputs, parse, logs.append)
    assert first.parsed == 2
    assert [label for label, _ in first.failed] == [bad]
    assert "ParseError" in first.failed[0][1] and "FAILED" in logs[1]
    record = failure_path(cache, cache_key(bad, inputs[1][2]))
    assert bad in record.read_text(encoding="utf-8")
    assert cache_key(bad, inputs[1][2]) not in cache

    failing.clear()
    calls.clear()
    second = await warm(cache, inputs, parse, logs.append)
    assert calls == [bad]  # only the failed document is parsed again
    assert second.parsed == 1 and second.failed == ()
    assert not record.exists()
    assert cache_key(bad, inputs[1][2]) in cache


@pytest.mark.asyncio
async def test_warm_stops_when_the_parse_service_is_down(tmp_path: Path) -> None:
    """A run of transient failures means the service is down: stop, don't mark every input."""
    inputs = _inputs(MAX_CONSECUTIVE_TRANSIENT_FAILURES + 2)
    calls: list[str] = []

    async def parse(document_sha256: str, data: bytes) -> Any:
        calls.append(document_sha256)
        raise _timeout()

    cache = ParseCache(tmp_path / "cache", PARSER_ID)
    with pytest.raises(ParseServiceDownError):
        await warm(cache, inputs, parse, lambda _m: None)
    assert len(calls) == MAX_CONSECUTIVE_TRANSIENT_FAILURES


@pytest.mark.asyncio
async def test_warm_keeps_going_past_deterministic_failures(tmp_path: Path) -> None:
    """Deterministic failures (an unreadable document) never count as the service being down."""
    inputs = _inputs(MAX_CONSECUTIVE_TRANSIENT_FAILURES + 2)

    async def parse(document_sha256: str, data: bytes) -> Any:
        raise ParseError("unreadable document")

    cache = ParseCache(tmp_path / "cache", PARSER_ID)
    result = await warm(cache, inputs, parse, lambda _m: None)
    assert result.parsed == 0 and len(result.failed) == len(inputs)


@pytest.mark.asyncio
async def test_contacted_marks_any_pass_that_reached_the_parse_service(tmp_path: Path) -> None:
    """The unload decision: a pass whose every parse failed (parsed == 0) still loaded the OCR
    model, and so did one stopped by `ParseServiceDownError`; only an all-cached pass did not."""

    async def unreadable(document_sha256: str, data: bytes) -> Any:
        raise ParseError("unreadable document")

    failing = Contacted(unreadable)
    result = await warm(ParseCache(tmp_path / "a", PARSER_ID), _inputs(2), failing, lambda _m: None)
    assert result.parsed == 0 and failing.called

    async def down(document_sha256: str, data: bytes) -> Any:
        raise _timeout()

    stopped = Contacted(down)
    with pytest.raises(ParseServiceDownError):
        await warm(
            ParseCache(tmp_path / "b", PARSER_ID),
            _inputs(MAX_CONSECUTIVE_TRANSIENT_FAILURES),
            stopped,
            lambda _m: None,
        )
    assert stopped.called

    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    _write_doc(data_dir, has_scan=False)
    _, data = document_input(data_dir, SHA, load_truth(data_dir, SHA))
    inputs = [(SHA, hashlib.sha256(data).hexdigest(), data)]
    cache = ParseCache(tmp_path / "c", PARSER_ID)
    await warm(cache, inputs, PaddleOcrDocumentParser(_FakeOcrClient()).parse, lambda _m: None)  # type: ignore[arg-type]
    cached = Contacted(unreadable)
    assert (await warm(cache, inputs, cached, lambda _m: None)).parsed == 0
    assert not cached.called
