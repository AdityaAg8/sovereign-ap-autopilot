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

import pytest

from apx.common.errors import ParsedArtifactIntegrityError
from apx.parse.parser import PaddleOcrDocumentParser
from evals.extraction_a import _run_one, parse_input
from evals.parse_cache import (
    PARSER_SOURCE_DIRS,
    REPO_ROOT,
    ParseCache,
    ParseCacheError,
    ParseCacheMissError,
    cache_key,
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
    assert await warm(cache, inputs, parser.parse, logs.append) == 1
    assert await warm(cache, inputs, parser.parse, logs.append) == 0
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
