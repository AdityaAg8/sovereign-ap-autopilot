"""Unit tests for `apx.ingest.store.FileDocumentStore`: content-addressed storage, idempotent
writes for the same bytes, `get`/`exists` and that hardening failures propagate as typed errors
before anything is written to disk (PRD §10 M5.5). No database is used.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from apx.ingest.errors import DocumentNotFoundError
from apx.ingest.interfaces import sha256_hex
from apx.ingest.store import FileDocumentStore
from apx.ingest.validate import PayloadTooLargeError, UnsupportedMediaTypeError

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 30


def _make_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 700, "hello")
    c.showPage()
    c.save()
    return buf.getvalue()


def _store(tmp_path: Path, **overrides: int) -> FileDocumentStore:
    return FileDocumentStore(
        tmp_path / "documents",
        max_bytes=overrides.get("max_bytes", MAX_BYTES),
        max_pages=overrides.get("max_pages", MAX_PAGES),
    )


def test_put_stores_the_document_under_its_sha256(tmp_path: Path) -> None:
    data = _make_pdf()
    store = _store(tmp_path)
    document = store.put(data, "invoice.pdf", "upload")
    assert document.sha256 == sha256_hex(data)
    assert document.media_type == "application/pdf"
    assert document.size_bytes == len(data)
    assert document.filename == "invoice.pdf"
    assert document.source == "upload"
    assert (tmp_path / "documents" / document.sha256).read_bytes() == data


def test_put_is_idempotent_for_identical_bytes(tmp_path: Path) -> None:
    data = _make_pdf()
    store = _store(tmp_path)
    first = store.put(data, "a.pdf", "upload")
    second = store.put(data, "b.pdf", "watch")
    assert first.sha256 == second.sha256
    # Filename/source reflect each call, not the first write; only the bytes are deduped.
    assert second.filename == "b.pdf"
    assert second.source == "watch"


def test_put_rejects_oversized_document_without_writing_it(tmp_path: Path) -> None:
    store = _store(tmp_path, max_bytes=10)
    with pytest.raises(PayloadTooLargeError):
        store.put(_make_pdf(), "invoice.pdf", "upload")
    assert not (tmp_path / "documents").exists()


def test_put_rejects_unsupported_media_type(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(UnsupportedMediaTypeError):
        store.put(b"plain text, not a document", "notes.txt", "upload")


def test_exists_and_get_round_trip(tmp_path: Path) -> None:
    data = _make_pdf()
    store = _store(tmp_path)
    document = store.put(data, "invoice.pdf", "upload")
    assert store.exists(document.sha256) is True
    assert store.get(document.sha256) == data


def test_exists_is_false_for_an_unknown_sha256(tmp_path: Path) -> None:
    store = _store(tmp_path)
    unknown = "0" * 64
    assert store.exists(unknown) is False


def test_get_raises_typed_error_for_an_unknown_sha256(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(DocumentNotFoundError):
        store.get("0" * 64)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    image_format=st.sampled_from(["PNG", "JPEG"]),
    suffix=st.binary(max_size=200),
)
def test_put_always_returns_the_true_sha256_of_the_bytes_it_stored(
    image_format: str, suffix: bytes, tmp_path: Path
) -> None:
    # PNG/JPEG are accepted on a readable, in-cap header (REVIEW H-03); trailing bytes after
    # the image are never read, so any suffix reaches `put` and must round-trip its sha256.
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format=image_format)
    data = buf.getvalue() + suffix
    store = _store(tmp_path)
    document = store.put(data, "f", "upload")
    assert document.sha256 == sha256_hex(data)
    assert store.get(document.sha256) == data
