"""Shared helpers for evals.datasets fetchers: sha256 hashing, resumable range-request
downloads over httpx, and appending BLOCKED.md entries for human-only gaps so a missing
token or package degrades the fetch run instead of failing it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED_MD = REPO_ROOT / "BLOCKED.md"
CHUNK_SIZE = 1 << 20  # 1 MiB


def repo_relative_posix(path: Path) -> str:
    """Repo-relative, forward-slash path for a manifest entry."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's contents, hex-encoded lowercase."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    timeout: float = 60.0,
) -> Path:
    """Downloads `url` to `dest`, resuming a partial `dest.part` file via a Range request.
    Renames `.part` to `dest` only once the transfer completes, so a killed job resumes
    cleanly on the next run; a `dest` that already exists is left untouched (already fetched)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
        request_headers = dict(headers or {})
        mode = "wb"
        if part.exists() and part.stat().st_size > 0:
            request_headers["Range"] = f"bytes={part.stat().st_size}-"
            mode = "ab"
        with http.stream("GET", url, headers=request_headers) as response:
            if response.status_code == 416:
                # Server says the range is beyond the resource: what we have is complete.
                part.rename(dest)
                return dest
            if mode == "ab" and response.status_code == 200:
                # Server ignored the Range header; restart the file from scratch.
                mode = "wb"
            response.raise_for_status()
            with part.open(mode) as handle:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    handle.write(chunk)
        part.rename(dest)
    finally:
        if owns_client:
            http.close()
    return dest


def append_blocked_entry(tag: str, title: str, why: str, command: str) -> None:
    """Appends a BLOCKED.md entry unless one with this `[tag]` is already present,
    so re-running a deferred fetch does not duplicate the entry."""
    marker = f"[{tag}]"
    existing = BLOCKED_MD.read_text(encoding="utf-8") if BLOCKED_MD.exists() else "# BLOCKED\n"
    if marker in existing:
        return
    entry = f"\n### {marker} {title}\n- **Why:** {why}\n- **Command:** `{command}`\n"
    with BLOCKED_MD.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(entry)
