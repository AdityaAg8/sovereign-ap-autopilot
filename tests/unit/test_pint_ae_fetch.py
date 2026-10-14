"""Unit tests for evals/pint_ae/fetch.py: pin.json loading, sha256 verification (mismatch is a
typed, non-silent PinMismatchError -- httpx is monkeypatched so this needs no network), and the
already-extracted-so-skip-the-network idempotency check."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from evals.pint_ae import fetch as fetch_module
from evals.pint_ae.fetch import Pin, PinMismatchError, load_pin, sha256_of


def test_load_pin_reads_committed_pin_json() -> None:
    pin = load_pin()
    assert pin.pint_ae.version
    assert pin.pint_ae.resources_url.startswith("https://")
    assert len(pin.pint_ae.resources_sha256) == 64
    assert pin.schxslt.version
    assert pin.schxslt.url.startswith("https://")
    assert len(pin.schxslt.sha256) == 64


def test_load_pin_round_trips_a_custom_file(tmp_path: Path) -> None:
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(
        json.dumps(
            {
                "pint_ae": {
                    "version": "9.9.9",
                    "resources_url": "https://example.invalid/resources.zip",
                    "resources_sha256": "a" * 64,
                },
                "schxslt": {
                    "version": "0.0.0",
                    "url": "https://example.invalid/schxslt.zip",
                    "sha256": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    pin = load_pin(pin_path)
    assert pin.pint_ae.version == "9.9.9"
    assert pin.schxslt.sha256 == "b" * 64


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"hello world")
    import hashlib

    assert sha256_of(target) == hashlib.sha256(b"hello world").hexdigest()


@contextmanager
def _fake_stream(body: bytes) -> Any:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self) -> Iterator[bytes]:
            yield body

    yield _FakeResponse()


def test_download_raises_pin_mismatch_and_deletes_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fetch_module.httpx, "stream", lambda *a, **kw: _fake_stream(b"not the real content")
    )
    dest = tmp_path / "download.bin"

    with pytest.raises(PinMismatchError, match="expected sha256"):
        fetch_module._download("https://example.invalid/file", dest, "f" * 64)

    assert not dest.exists()


def test_ensure_pint_ae_skips_network_when_already_extracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = Pin(
        pint_ae=fetch_module.PintAePin(
            version="1.2.3",
            resources_url="https://example.invalid/r.zip",
            resources_sha256="c" * 64,
        ),
        schxslt=fetch_module.SchXsltPin(
            version="4.5.6", url="https://example.invalid/s.zip", sha256="d" * 64
        ),
    )
    marker = (
        tmp_path
        / "1.2.3"
        / "resources"
        / "trn-invoice"
        / "schematron"
        / "PINT-jurisdiction-aligned-rules.sch"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("<schema/>", encoding="utf-8")

    def _fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("network should not be called when already extracted")

    monkeypatch.setattr(fetch_module, "_download", _fail_download)

    result = fetch_module.ensure_pint_ae(tmp_path, pin)
    assert result == marker.parent.parent.parent
