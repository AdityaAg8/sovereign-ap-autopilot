"""Unit tests for `apx.parse.store.FileParsedDocumentStore`: a 5-page parse (real PNGs) round
-trips exactly, `put` is idempotent and content-addressed, and a missing or tampered entry
raises the matching typed error instead of returning partial data (PRD §10 M5.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from apx.common.errors import ParsedArtifactIntegrityError, ParsedArtifactMissingError
from apx.common.types import Language
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.parse.store import FileParsedDocumentStore
from tests.unit import _payload_fixtures as fx


@pytest.fixture(scope="module")
def document() -> ParsedDocument:
    return fx.parsed_document()


def _entry(root: Path, ref: ParsedDocumentRef) -> Path:
    return root / ref.document_sha256 / ref.content_sha256


def test_round_trips_every_page_image_and_word(tmp_path: Path, document: ParsedDocument) -> None:
    store = FileParsedDocumentStore(tmp_path)

    ref = store.put(document)

    assert ref.document_sha256 == document.document_sha256
    assert ref.page_count == fx.PAGES
    assert ref.parser_version == document.parser_version
    assert store.load(ref) == document


def test_put_is_idempotent_and_content_addressed(tmp_path: Path, document: ParsedDocument) -> None:
    store = FileParsedDocumentStore(tmp_path)

    first = store.put(document)
    again = store.put(document)
    reordered = store.put(
        document.model_copy(update={"languages": frozenset([Language.AR, Language.EN])})
    )
    changed = store.put(document.model_copy(update={"reading_order": ()}))

    assert first == again == reordered
    assert changed.content_sha256 != first.content_sha256
    assert not list(tmp_path.rglob("*.tmp"))


def test_missing_manifest_is_a_typed_miss(tmp_path: Path, document: ParsedDocument) -> None:
    store = FileParsedDocumentStore(tmp_path)
    ref = store.put(document)
    (_entry(tmp_path, ref) / "document.json").unlink()

    with pytest.raises(ParsedArtifactMissingError):
        store.load(ref)


def test_missing_page_image_is_a_typed_miss(tmp_path: Path, document: ParsedDocument) -> None:
    store = FileParsedDocumentStore(tmp_path)
    ref = store.put(document)
    (_entry(tmp_path, ref) / "page-0003.png").unlink()

    with pytest.raises(ParsedArtifactMissingError):
        store.load(ref)


def test_never_stored_ref_is_a_typed_miss(tmp_path: Path) -> None:
    with pytest.raises(ParsedArtifactMissingError):
        FileParsedDocumentStore(tmp_path).load(fx.parsed_ref())


def test_corrupt_page_image_is_an_integrity_error(tmp_path: Path, document: ParsedDocument) -> None:
    store = FileParsedDocumentStore(tmp_path)
    ref = store.put(document)
    (_entry(tmp_path, ref) / "page-0002.png").write_bytes(b"\x89PNG tampered")

    with pytest.raises(ParsedArtifactIntegrityError):
        store.load(ref)


def test_tampered_manifest_is_an_integrity_error(tmp_path: Path, document: ParsedDocument) -> None:
    store = FileParsedDocumentStore(tmp_path)
    ref = store.put(document)
    manifest = _entry(tmp_path, ref) / "document.json"
    manifest.write_bytes(manifest.read_bytes().replace(b"AED10.00", b"AED99.00", 1))

    with pytest.raises(ParsedArtifactIntegrityError):
        store.load(ref)


def test_ref_with_other_metadata_is_an_integrity_error(
    tmp_path: Path, document: ParsedDocument
) -> None:
    store = FileParsedDocumentStore(tmp_path)
    ref = store.put(document)

    with pytest.raises(ParsedArtifactIntegrityError):
        store.load(ref.model_copy(update={"parser_version": "other/2"}))
