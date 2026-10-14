"""Content-addressed `DocumentStore` on local disk: validates a document against every upload
hardening rule (`apx.ingest.validate`), then writes it under its sha256 exactly once, so a byte
-identical document dropped in the watch folder or uploaded through the API is stored a single
time."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from apx.common.types import Sha256Hex
from apx.ingest.errors import DocumentNotFoundError
from apx.ingest.interfaces import DocumentStore, IngestedDocument, sha256_hex
from apx.ingest.validate import validate


class FileDocumentStore(DocumentStore):
    """`DocumentStore` backed by a flat directory of `<sha256>` files under `root` (created on
    first use). Not safe to point two different `root`s at the same directory as another store
    with different limits; `put` is otherwise idempotent and safe to call concurrently for the
    same bytes, since the final `write_bytes` is the only step touching a shared path and
    writing identical content twice is harmless."""

    def __init__(self, root: Path, *, max_bytes: int, max_pages: int) -> None:
        self._root = root
        self._max_bytes = max_bytes
        self._max_pages = max_pages

    def _path(self, sha256: Sha256Hex) -> Path:
        return self._root / sha256

    def put(
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestedDocument:
        media_type = validate(data, max_bytes=self._max_bytes, max_pages=self._max_pages)
        digest = sha256_hex(data)
        self._root.mkdir(parents=True, exist_ok=True)
        dest = self._path(digest)
        if not dest.exists():
            dest.write_bytes(data)
        return IngestedDocument(
            sha256=digest,
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
            received_at=datetime.now(UTC),
            source=source,
        )

    def get(self, sha256: Sha256Hex) -> bytes:
        path = self._path(sha256)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise DocumentNotFoundError(f"no stored document for sha256={sha256}") from exc

    def exists(self, sha256: Sha256Hex) -> bool:
        return self._path(sha256).is_file()
