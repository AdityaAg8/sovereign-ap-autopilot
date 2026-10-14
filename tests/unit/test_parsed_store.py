"""Unit tests for `apx.parse.store.FileParsedDocumentStore`: a 5-page parse (real PNGs) round
-trips exactly, `put` is idempotent and content-addressed, and a missing or tampered entry
raises the matching typed error instead of returning partial data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apx.common.errors import ParsedArtifactIntegrityError, ParsedArtifactMissingError
from apx.common.types import Language
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.parse.store import FileParsedDocumentStore, canonical_manifest
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


def test_untruncated_parse_hashes_as_before_the_flag_existed(document: ParsedDocument) -> None:
    """ADR-0013 amendment 1: `Page.truncated` is written only when True, so a complete parse's
    manifest bytes (and so every stored ref and every re-parse check) are unchanged."""
    assert all(not page.truncated for page in document.pages)
    assert b'"truncated"' not in canonical_manifest(document)


def test_old_manifest_without_the_flag_loads_as_untruncated(
    tmp_path: Path, document: ParsedDocument
) -> None:
    store = FileParsedDocumentStore(tmp_path)
    ref = store.put(document)
    body = json.loads((_entry(tmp_path, ref) / "document.json").read_bytes())
    assert all("truncated" not in page for page in body["document"]["pages"])

    loaded = store.load(ref)
    assert loaded.truncated_pages == ()


def test_truncated_page_round_trips_and_changes_the_content_address(
    tmp_path: Path, document: ParsedDocument
) -> None:
    store = FileParsedDocumentStore(tmp_path)
    pages = list(document.pages)
    pages[1] = pages[1].model_copy(update={"truncated": True})
    cut = document.model_copy(update={"pages": tuple(pages)})

    ref = store.put(cut)

    assert ref.content_sha256 != store.put(document).content_sha256
    assert store.load(ref).truncated_pages == (pages[1].index,)
