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
import os
import re
import sys
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Final

from pydantic import ValidationError as PydanticValidationError

from apx.common.errors import ApxError
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.parse.parser import SERVICE_PARSER_VERSION
from apx.parse.store import FileParsedDocumentStore

ParseFn = Callable[[str, bytes], Awaitable[ParsedDocument]]
"""`DocumentParser.parse`: (document sha256, file bytes) -> parse."""

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

    def __init__(self, root: Path = DEFAULT_ROOT, parser_id: str | None = None) -> None:
        self.parser_id = parser_id or parser_identity()
        self.directory = root / self.parser_id
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
        done = len(items) - len(pending)
        for key, path in pending:
            document = await parse(key, path.read_bytes())
            self.put(key, document)
            done += 1
            log(f"[{done}/{len(items)}] parse {key[:12]} pages={len(document.pages)}")
        return len(pending)


async def warm(
    cache: ParseCache,
    inputs: Sequence[tuple[str, str, bytes]],
    parse: ParseFn,
    log: Callable[[str], None] = print,
) -> int:
    """Parse every (label, document sha256, bytes) input into `cache` ahead of the GPU evals;
    returns how many were parsed now (the rest were cached)."""
    parsed = 0
    for position, (label, document_sha256, data) in enumerate(inputs, start=1):
        hit = cache_key(document_sha256, data) in cache
        document = await cache.parse(document_sha256, data, parse)
        parsed += not hit
        log(
            f"[{position}/{len(inputs)}] {label[:12]} {'cached' if hit else 'parsed'} "
            f"pages={len(document.pages)}"
        )
    return parsed


def split_inputs(data_dir: Path, split: str, limit: int = 0) -> list[tuple[str, str, bytes]]:
    """The parse inputs `evals.extraction_a` feeds for a UAE split, in its order: the runtime
    form of each document (scan PDF for `has_scan`) under the sha256 of those bytes."""
    from evals.scan_input import document_input, load_truth
    from evals.splits import load_splits

    shas = [s for s in sorted(load_splits().sets["uae"][split]) if (data_dir / f"{s}.pdf").exists()]
    inputs: list[tuple[str, str, bytes]] = []
    for sha in shas[:limit] if limit else shas:
        _, data = document_input(data_dir, sha, load_truth(data_dir, sha))
        inputs.append((sha, hashlib.sha256(data).hexdigest(), data))
    return inputs


async def _run(args: argparse.Namespace) -> int:  # pragma: no cover - needs the parse service
    from apx.parse.client import ParseServiceClient
    from apx.parse.parser import PaddleOcrDocumentParser

    cache = ParseCache(args.root)
    client = ParseServiceClient()
    parser = PaddleOcrDocumentParser(client)
    parsed = 0
    try:
        for split in args.splits.split(","):
            inputs = split_inputs(args.data_root / split, split, args.limit)
            if not inputs:
                print(f"no UAE {split} documents in {args.data_root / split}", file=sys.stderr)
                return 2
            print(f"==> {split}: {len(inputs)} documents -> {cache.directory}", flush=True)
            parsed += await warm(cache, inputs, parser.parse, lambda m: print(m, flush=True))
    finally:
        if parsed:
            await client.unload()  # free the OCR model's VRAM before the GPU evals start
    print(f"parse cache {cache.parser_id}: {parsed} parsed now")
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
