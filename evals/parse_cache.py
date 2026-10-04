"""Shared parse cache for the GPU evals (extraction harness, Extractor B gates, M3.4 bake-off):
each document is parsed once per parser build into a content-addressed store under
data/cache/parse/<parser identity>/, so resumes and later evals never re-OCR a scan. The parser
identity hashes apx/parse and the parse service source, so a parser fix invalidates the cache.

Usage: python -m evals.parse_cache --splits test,cal [--data-root data/uae] [--root DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, overload

import httpx
from pydantic import ValidationError as PydanticValidationError

from apx.common.errors import ApxError, ParseError
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.parse.parser import SERVICE_PARSER_VERSION
from apx.parse.store import FileParsedDocumentStore

ParseFn = Callable[[str, bytes], Awaitable[ParsedDocument]]
"""`DocumentParser.parse`: (document sha256, file bytes) -> parse."""

PARSE_RETRY_BACKOFF_S: Final[tuple[float, ...]] = (5.0, 20.0, 60.0)
"""Waits before each retry of a transient parse-service failure (so at most 4 attempts): the OCR
service shares the GPU with llama-server, and a page can time out while the LLM holds it."""
TRANSIENT_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_parse_error(exc: ParseError) -> bool:
    """A parse-service failure worth retrying: a timeout, a transport error (refused, reset) or a
    busy/5xx status, read from the httpx cause `apx.parse.client` chains. A malformed response,
    an unreadable document or a raster-cap refusal is deterministic and never retried."""
    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return cause.response.status_code in TRANSIENT_STATUS
    return isinstance(cause, httpx.TransportError)


def retrying(
    parse: ParseFn,
    *,
    backoff_s: Sequence[float] = PARSE_RETRY_BACKOFF_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: Callable[[str], None] = print,
) -> ParseFn:
    """`parse` with bounded retries: a transient `ParseError` is retried after each wait in
    `backoff_s`; the last failure (or any deterministic one) propagates unchanged, typed, so the
    caller records it. Every retry is logged, never silent."""

    async def call(document_sha256: str, data: bytes) -> ParsedDocument:
        waits = list(backoff_s)
        attempt = 1
        while True:
            try:
                return await parse(document_sha256, data)
            except ParseError as exc:
                if not waits or not is_transient_parse_error(exc):
                    raise
                delay = waits.pop(0)
                log(
                    f"parse {document_sha256[:12]} attempt {attempt} failed ({exc}); "
                    f"retrying in {delay:g} s"
                )
                attempt += 1
                await sleep(delay)

    return call


class Contacted:
    """`parse` that remembers whether it was ever called. A pass that sent any page to the parse
    service (parsed or failed, or stopped by an exception) must unload the OCR model before the
    GPU evals start; only a pass served wholly from the cache may skip the unload."""

    def __init__(self, parse: ParseFn) -> None:
        self._parse = parse
        self.called = False

    async def __call__(self, document_sha256: str, data: bytes) -> ParsedDocument:
        self.called = True
        return await self._parse(document_sha256, data)


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_ROOT: Final[Path] = Path("data/cache/parse")
PARSER_SOURCE_DIRS: Final[tuple[str, ...]] = ("apx/parse", "infra/docker/parse")
"""Everything that decides what a parse contains: the client-side router and pdfplumber path,
and the OCR service (model, max new tokens, repetition penalty, image budget)."""
_KEY: Final = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,199}")


class ParseCacheError(ApxError):
    """The parse cache cannot answer: an unknown key, a bad key or unreadable parser sources."""


class ParseCacheMissError(ParseCacheError):
    """A key that was required to be cached (`ParseCache.get`) is not."""


def _is_parser_source(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and (path.suffix == ".py" or path.name == "Dockerfile")
    )


def parser_identity(repo_root: Path = REPO_ROOT) -> str:
    """16-hex hash of `SERVICE_PARSER_VERSION` plus every parser source file (path and bytes,
    line endings normalised so a Windows and a Linux checkout agree). Any edit to the parse
    path, a comment included, gives a new identity: a stale cache is never trusted."""
    digest = hashlib.sha256(SERVICE_PARSER_VERSION.encode("utf-8") + b"\0")
    for rel in PARSER_SOURCE_DIRS:
        base = repo_root / rel
        files = sorted(p for p in base.rglob("*") if _is_parser_source(p)) if base.is_dir() else []
        if not files:
            raise ParseCacheError(f"no parser sources under {base}; cannot identify the parser")
        for path in files:
            digest.update(path.relative_to(repo_root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


def cache_key(document_sha256: str, data: bytes) -> str:
    """The key of one parse input: the sha256 of the bytes parsed, plus the document id the
    caller stamps on the parse when that differs (the id is part of the stored parse)."""
    data_sha = hashlib.sha256(data).hexdigest()
    return data_sha if document_sha256 == data_sha else f"{data_sha}.{document_sha256}"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ParseCache:
    """`<root>/<parser identity>/store/...` (a `FileParsedDocumentStore`) and `index/<key>.json`
    (the `ParsedDocumentRef` of that key). Every file is written by temp file + `os.replace` and
    the index entry only after its store entry, so concurrent readers and writers (two eval jobs
    on the same split) never see a torn entry; two writers of one key write the same parse."""

    def __init__(
        self, root: Path = DEFAULT_ROOT, parser_id: str | None = None, *, read_only: bool = False
    ) -> None:
        self.parser_id = parser_id or parser_identity()
        self.directory = root / self.parser_id
        self.read_only = read_only
        """A pinned snapshot (`open_cache`): answered from, never written to, so another parser
        build's output can never be stored under this identity."""
        self._store = FileParsedDocumentStore(self.directory / "store")
        self._index = self.directory / "index"

    def _entry(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ParseCacheError(f"invalid parse cache key {key!r}")
        return self._index / f"{key}.json"

    def __contains__(self, key: str) -> bool:
        return self._entry(key).is_file()

    def lookup(self, key: str) -> ParsedDocument | None:
        """The cached parse of `key`, or None. A corrupt entry raises (never re-parsed silently:
        `ParsedArtifactIntegrityError`/`ParsedArtifactMissingError` or `ParseCacheError`)."""
        entry = self._entry(key)
        try:
            raw = entry.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            ref = ParsedDocumentRef.model_validate_json(raw)
        except PydanticValidationError as exc:
            raise ParseCacheError(f"parse cache index entry {entry} is malformed") from exc
        return self._store.load(ref)

    def get(self, key: str) -> ParsedDocument:
        document = self.lookup(key)
        if document is None:
            raise ParseCacheMissError(
                f"{key[:12]} is not in the parse cache {self.directory}; run the parse phase first"
            )
        return document

    def put(self, key: str, document: ParsedDocument) -> None:
        if self.read_only:
            raise ParseCacheError(f"parse cache {self.directory} is a pinned read-only snapshot")
        entry = self._entry(key)
        ref = self._store.put(document)
        entry.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(entry, ref.model_dump_json().encode("utf-8"))

    async def parse(self, document_sha256: str, data: bytes, parse: ParseFn) -> ParsedDocument:
        """`parse(document_sha256, data)`, answered from the cache when this input was parsed
        before by the same parser build."""
        key = cache_key(document_sha256, data)
        cached = self.lookup(key)
        if cached is not None:
            return cached
        if self.read_only:
            raise ParseCacheMissError(
                f"{key[:12]} is not in the pinned parse cache {self.directory}; a pinned snapshot "
                "is never filled by the current parser build"
            )
        document = await parse(document_sha256, data)
        self.put(key, document)
        return document

    async def fill(
        self,
        items: Sequence[tuple[str, Path]],
        parse: ParseFn,
        log: Callable[[str], None] = print,
    ) -> int:
        """Parse every (key, file) not cached yet, keyed by the name the caller gives it (the
        bake-off's form); returns how many were parsed now."""
        pending = [(key, path) for key, path in items if key not in self]
        if pending and self.read_only:
            raise ParseCacheMissError(
                f"{len(pending)} of {len(items)} inputs are not in the pinned parse cache "
                f"{self.directory} (e.g. {pending[0][0][:12]})"
            )
        done = len(items) - len(pending)
        for key, path in pending:
            document = await parse(key, path.read_bytes())
            self.put(key, document)
            done += 1
            log(f"[{done}/{len(items)}] parse {key[:12]} pages={len(document.pages)}")
        return len(pending)


_PARSER_ID: Final = re.compile(r"[0-9a-f]{16}")


def open_cache(
    parser_id: str | None = None,
    root: Path = DEFAULT_ROOT,
    log: Callable[[str], None] = print,
) -> ParseCache:
    """The shared cache of the current parser build, or, with `parser_id`, an earlier build's
    snapshot pinned so an eval reads exactly the parses the arms it is compared with were
    measured on (ADR-0019 amendment 3). A pinned snapshot of another build is read-only: a miss
    raises instead of parsing, and the pin is logged with both identities."""
    current = parser_identity()
    if parser_id is None or parser_id == current:
        return ParseCache(root)
    if not _PARSER_ID.fullmatch(parser_id):
        raise ParseCacheError(f"parser id {parser_id!r} is not 16 lowercase hex characters")
    if not (root / parser_id / "index").is_dir():
        raise ParseCacheError(f"no parse cache snapshot {parser_id} under {root}")
    log(f"parse cache PINNED to {parser_id} (read-only); the current parser build is {current}")
    return ParseCache(root, parser_id, read_only=True)


MAX_CONSECUTIVE_TRANSIENT_FAILURES: Final = 3
"""Transient failures in a row (after each one's retries) that mean the parse service itself is
down, not one hard document: `warm` stops instead of recording every remaining input as failed."""


class ParseServiceDownError(ParseCacheError):
    """`warm` saw `MAX_CONSECUTIVE_TRANSIENT_FAILURES` transient parse failures in a row."""


@dataclass(frozen=True)
class WarmResult:
    """What one `warm` pass did: inputs parsed now, and the (label, error) of each input whose
    parse failed (recorded under `failed/`, retried by the next pass)."""

    parsed: int
    failed: tuple[tuple[str, str], ...]


def failure_path(cache: ParseCache, key: str) -> Path:
    """The failure record of `key`: written when its parse fails, removed when one succeeds."""
    return cache.directory / "failed" / f"{cache._entry(key).name}"


async def warm(
    cache: ParseCache,
    inputs: Sequence[tuple[str, str, bytes]],
    parse: ParseFn,
    log: Callable[[str], None] = print,
) -> WarmResult:
    """Parse every (label, document sha256, bytes) input into `cache` ahead of the GPU evals.
    A `ParseError` on one input (a page timeout under contention, an unreadable scan) is recorded
    in `failed/<key>.json` with its error and the pass continues: the input stays uncached, so
    the next pass retries it and the evals that read the cache parse it live and count its
    failure in their denominators. Only a run of transient failures (the service is down) stops
    the pass, raised as `ParseServiceDownError`."""
    parsed = 0
    failed: list[tuple[str, str]] = []
    transient_run = 0
    for position, (label, document_sha256, data) in enumerate(inputs, start=1):
        key = cache_key(document_sha256, data)
        hit = key in cache
        record = failure_path(cache, key)
        try:
            document = await cache.parse(document_sha256, data, parse)
        except ParseError as exc:
            error = f"{type(exc).__name__}: {exc}"
            failed.append((label, error))
            record.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(
                record,
                json.dumps(
                    {"label": label, "document_sha256": document_sha256, "error": error}
                ).encode("utf-8"),
            )
            log(f"[{position}/{len(inputs)}] {label[:12]} FAILED {error}")
            transient_run = transient_run + 1 if is_transient_parse_error(exc) else 0
            if transient_run >= MAX_CONSECUTIVE_TRANSIENT_FAILURES:
                raise ParseServiceDownError(
                    f"parse service failed {transient_run} times in a row; stopping the pass"
                ) from exc
            continue
        transient_run = 0
        record.unlink(missing_ok=True)
        parsed += not hit
        log(
            f"[{position}/{len(inputs)}] {label[:12]} {'cached' if hit else 'parsed'} "
            f"pages={len(document.pages)}"
        )
    return WarmResult(parsed=parsed, failed=tuple(failed))


class SplitInputs(Sequence[tuple[str, str, bytes]]):
    """The parse inputs `evals.extraction_a` feeds for a UAE split, in its order: the runtime
    form of each document (scan PDF for `has_scan`) under the sha256 of those bytes. Built one
    document at a time on access: a split's scan PDFs hold raw page pixels, so materialising
    all of them up front ran the host out of memory."""

    def __init__(self, data_dir: Path, shas: Sequence[str]) -> None:
        self._data_dir = data_dir
        self._shas = list(shas)

    def __len__(self) -> int:
        return len(self._shas)

    @overload
    def __getitem__(self, index: int) -> tuple[str, str, bytes]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[str, str, bytes]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[str, str, bytes] | Sequence[tuple[str, str, bytes]]:
        if isinstance(index, slice):
            return SplitInputs(self._data_dir, self._shas[index])
        from evals.scan_input import document_input, load_truth

        sha = self._shas[index]
        _, data = document_input(self._data_dir, sha, load_truth(self._data_dir, sha))
        return sha, hashlib.sha256(data).hexdigest(), data


def split_inputs(data_dir: Path, split: str, limit: int = 0) -> SplitInputs:
    """`SplitInputs` for the UAE `split` documents present in `data_dir` (first `limit`)."""
    from evals.splits import load_splits

    shas = [s for s in sorted(load_splits().sets["uae"][split]) if (data_dir / f"{s}.pdf").exists()]
    return SplitInputs(data_dir, shas[:limit] if limit else shas)


def _log(message: str) -> None:
    print(message, flush=True)


async def _run(args: argparse.Namespace) -> int:  # pragma: no cover - needs the parse service
    from apx.parse.client import ParseServiceClient
    from apx.parse.parser import PaddleOcrDocumentParser

    cache = ParseCache(args.root)
    client = ParseServiceClient()
    parser = PaddleOcrDocumentParser(client)
    parse = Contacted(retrying(parser.parse, log=_log))
    parsed = 0
    failed: list[tuple[str, str]] = []
    try:
        for split in args.splits.split(","):
            inputs = split_inputs(args.data_root / split, split, args.limit)
            if not inputs:
                print(f"no UAE {split} documents in {args.data_root / split}", file=sys.stderr)
                return 2
            print(f"==> {split}: {len(inputs)} documents -> {cache.directory}", flush=True)
            result = await warm(cache, inputs, parse, _log)
            parsed += result.parsed
            failed.extend(result.failed)
    finally:
        if parse.called:
            await client.unload()  # free the OCR model's VRAM before the GPU evals start
    print(f"parse cache {cache.parser_id}: {parsed} parsed now, {len(failed)} failed")
    for label, error in failed:  # recorded under failed/; a rerun retries them
        print(f"  failed {label[:12]}: {error}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    parser = argparse.ArgumentParser(prog="python -m evals.parse_cache", description=__doc__)
    parser.add_argument("--splits", default="test,cal", help="comma-separated UAE splits")
    parser.add_argument("--data-root", type=Path, default=Path("data/uae"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
