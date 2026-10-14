"""Unit tests for evals/datasets/common.py: sha256 hashing, resumable range-request downloads
(against httpx.MockTransport, no real network) and SETUP_TODO.md entry appending."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from evals.datasets import common


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    payload = b"hello sovereign ap autopilot" * 1000
    path.write_bytes(payload)

    assert common.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_repo_relative_posix_uses_forward_slashes() -> None:
    path = common.REPO_ROOT / "data" / "fatura" / "FATURA.zip"
    assert common.repo_relative_posix(path) == "data/fatura/FATURA.zip"


def test_download_resumable_writes_full_body(tmp_path: Path) -> None:
    body = b"0123456789" * 100

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    dest = tmp_path / "out.bin"
    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = common.download_resumable("https://example.org/f.bin", dest, client=client)

    assert result == dest
    assert dest.read_bytes() == body
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_resumable_skips_when_dest_already_present(tmp_path: Path) -> None:
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"already here")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"should not be fetched")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    common.download_resumable("https://example.org/f.bin", dest, client=client)

    assert calls == 0
    assert dest.read_bytes() == b"already here"


def test_download_resumable_resumes_partial_file_via_range(tmp_path: Path) -> None:
    full_body = b"abcdefghij" * 50
    dest = tmp_path / "out.bin"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(full_body[:200])
    seen_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range"))
        return httpx.Response(206, content=full_body[200:])

    client = httpx.Client(transport=httpx.MockTransport(handler))

    common.download_resumable("https://example.org/f.bin", dest, client=client)

    assert seen_ranges == ["bytes=200-"]
    assert dest.read_bytes() == full_body
    assert not part.exists()


def test_download_resumable_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    full_body = b"z" * 500
    dest = tmp_path / "out.bin"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"stale-partial-data")

    def handler(request: httpx.Request) -> httpx.Response:
        # Server does not honor Range and sends the whole body with 200.
        return httpx.Response(200, content=full_body)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    common.download_resumable("https://example.org/f.bin", dest, client=client)

    assert dest.read_bytes() == full_body


def test_append_setup_todo_writes_expected_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_todo_md = tmp_path / "SETUP_TODO.md"
    setup_todo_md.write_text(common.SETUP_TODO_HEADER, encoding="utf-8", newline="\n")
    monkeypatch.setattr(common, "SETUP_TODO_MD", setup_todo_md)

    common.append_setup_todo(
        tag="dataset:docile",
        title="Needs a token",
        why="DOCILE_TOKEN is not set.",
        command="DOCILE_TOKEN=... uv run python -m evals.datasets fetch --set docile",
    )

    text = setup_todo_md.read_text(encoding="utf-8")
    assert "### [dataset:docile] Needs a token" in text
    assert "- **Why:** DOCILE_TOKEN is not set." in text
    assert "fetch --set docile" in text


def test_append_setup_todo_is_idempotent_per_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_todo_md = tmp_path / "SETUP_TODO.md"
    setup_todo_md.write_text(common.SETUP_TODO_HEADER, encoding="utf-8", newline="\n")
    monkeypatch.setattr(common, "SETUP_TODO_MD", setup_todo_md)

    common.append_setup_todo(tag="dataset:x", title="A", why="a", command="cmd-a")
    common.append_setup_todo(tag="dataset:x", title="B", why="b", command="cmd-b")

    text = setup_todo_md.read_text(encoding="utf-8")
    assert text.count("[dataset:x]") == 1
    assert "cmd-b" not in text


def test_append_setup_todo_starts_a_new_file_with_the_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_todo_md = tmp_path / "SETUP_TODO.md"
    monkeypatch.setattr(common, "SETUP_TODO_MD", setup_todo_md)

    common.append_setup_todo(tag="dataset:x", title="A", why="a", command="cmd-a")

    text = setup_todo_md.read_text(encoding="utf-8")
    assert text.startswith("# Setup to-do\n\nManual setup steps the scripts couldn't do for you")
    assert text.count("[dataset:x]") == 1
