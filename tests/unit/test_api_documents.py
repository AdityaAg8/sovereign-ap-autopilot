"""Unit tests for the document-serving digest guard in `apx.api.service.DbInvoiceService.
get_document` (docs/security.md B1): the sha256-shape check that stops a
corrupted or otherwise-unexpected `document_sha256` from ever reaching `FileDocumentStore.get`
with a value that is not exactly 64 lowercase hex characters -- the only shape it turns into an
on-disk path. No database is needed for this part; the full route (including a real invoice row
and a real stored file) is covered end to end in `tests/integration/test_api_authz.py`."""

from __future__ import annotations

import pytest

from apx.api.service import _DOCUMENT_EXTENSION, _SHA256_RE

NOT_A_VALID_DIGEST = [
    "../../../etc/passwd",
    "..\\..\\windows\\system32\\config\\sam",
    "%2e%2e%2fetc%2fpasswd",
    "",
    "not-a-hash",
    "a" * 63,
    "a" * 65,
    "A" * 64,  # uppercase hex is not the stored (lowercase-only) form
    "a" * 63 + "/",
]


@pytest.mark.parametrize("payload", NOT_A_VALID_DIGEST)
def test_sha256_guard_rejects_anything_that_is_not_64_lowercase_hex_chars(payload: str) -> None:
    assert _SHA256_RE.fullmatch(payload) is None


def test_sha256_guard_accepts_a_real_sha256_digest_shape() -> None:
    assert _SHA256_RE.fullmatch("a" * 64) is not None


def test_every_accepted_media_type_maps_to_a_plain_alphanumeric_extension() -> None:
    # A `Content-Disposition` filename is built as `f"{digest}.{extension}"` (a validated hex
    # digest plus this extension); the extension must never be able to inject a header value or
    # a path segment.
    for media_type, extension in _DOCUMENT_EXTENSION.items():
        assert extension.isalnum()
        assert media_type.startswith(("application/", "image/"))
