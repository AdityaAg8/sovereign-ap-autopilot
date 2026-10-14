"""Content-addressed on-disk `ParsedDocumentStore`: the parse activity writes each parse here
once and hands downstream activities a small `ParsedDocumentRef`, so page PNGs never cross the
Temporal payload boundary. Layout: `<root>/<document_sha256>/<content_sha256>/document.json`
plus `page-<index>.png`."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Final

from pydantic import JsonValue
from pydantic import ValidationError as PydanticValidationError

from apx.common.errors import ParsedArtifactIntegrityError, ParsedArtifactMissingError
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef, ParsedDocumentStore

_DOCUMENT_FILE: Final[str] = "document.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _page_file(position: int) -> str:
    """By position in `pages`, not `Page.index`, so two pages can never share a file."""
    return f"page-{position:04d}.png"


def canonical_manifest(document: ParsedDocument) -> bytes:
    """The stored form of `document` minus its page images, which are referenced by sha256.
    Canonical (sorted keys, sorted languages) so the same parse hashes identically in every
    process regardless of set iteration order; its sha256 is the ref's `content_sha256`. A page's
    `truncated` key is written only when True, so every parse without a truncated page hashes
    exactly as it did before the flag existed (stored refs stay valid and re-parses match them)."""
    stripped = document.model_copy(
        update={"pages": tuple(p.model_copy(update={"image_png": b""}) for p in document.pages)}
    )
    body: dict[str, JsonValue] = stripped.model_dump(mode="json")
    pages = body["pages"]
    for page in pages if isinstance(pages, list) else ():
        if isinstance(page, dict) and page.get("truncated") is False:
            del page["truncated"]
    languages: list[JsonValue] = [language.value for language in sorted(document.languages)]
    body["languages"] = languages
    manifest: dict[str, JsonValue] = {
        "document": body,
        "page_png_sha256": [_sha256(page.image_png) for page in document.pages],
    }
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a unique temp file and `os.replace`, so a crashed or concurrent (retried)
    writer never leaves a torn file under the final name."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class FileParsedDocumentStore(ParsedDocumentStore):
    """`ParsedDocumentStore` under `root` (created on first use). Entries are immutable and
    content-addressed, so `put` is idempotent and safe under Temporal activity retries; the
    manifest is written last, so its presence means every page PNG was written before it."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _dir(self, ref: ParsedDocumentRef) -> Path:
        return self._root / ref.document_sha256 / ref.content_sha256

    def put(self, document: ParsedDocument) -> ParsedDocumentRef:
        manifest = canonical_manifest(document)
        ref = ParsedDocumentRef(
            document_sha256=document.document_sha256,
            content_sha256=_sha256(manifest),
            parser_version=document.parser_version,
            page_count=len(document.pages),
        )
        directory = self._dir(ref)
        directory.mkdir(parents=True, exist_ok=True)
        for position, page in enumerate(document.pages):
            _write_atomic(directory / _page_file(position), page.image_png)
        _write_atomic(directory / _DOCUMENT_FILE, manifest)
        return ref

    def load(self, ref: ParsedDocumentRef) -> ParsedDocument:
        directory = self._dir(ref)
        manifest = self._read(directory / _DOCUMENT_FILE, ref)
        if _sha256(manifest) != ref.content_sha256:
            raise ParsedArtifactIntegrityError(f"stored parse {directory} does not match its ref")
        try:
            body = json.loads(manifest)
            document = ParsedDocument.model_validate(body["document"])
            png_digests = [str(d) for d in body["page_png_sha256"]]
        except (ValueError, KeyError, TypeError, PydanticValidationError) as exc:
            raise ParsedArtifactIntegrityError(f"stored parse {directory} is malformed") from exc
        if (
            document.document_sha256 != ref.document_sha256
            or document.parser_version != ref.parser_version
            or len(document.pages) != ref.page_count
            or len(png_digests) != ref.page_count
        ):
            raise ParsedArtifactIntegrityError(f"stored parse {directory} does not match its ref")
        pages = []
        for position, (page, digest) in enumerate(zip(document.pages, png_digests, strict=True)):
            png = self._read(directory / _page_file(position), ref)
            if _sha256(png) != digest:
                raise ParsedArtifactIntegrityError(
                    f"page {page.index} image of stored parse {directory} is corrupt"
                )
            pages.append(page.model_copy(update={"image_png": png}))
        return document.model_copy(update={"pages": tuple(pages)})

    @staticmethod
    def _read(path: Path, ref: ParsedDocumentRef) -> bytes:
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ParsedArtifactMissingError(
                f"no stored parse file {path.name} for document {ref.document_sha256} "
                f"(content {ref.content_sha256})"
            ) from exc
