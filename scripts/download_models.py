"""Resumable downloader for the models pinned in infra/models.lock.json, with sha256
verification; this is what `make models` runs. It talks to the public
Hugging Face CDN over the host's normal internet path -- never over the `app` Docker
network, which is `internal: true` at runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Protocol

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "infra" / "models.lock.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


class DownloadError(RuntimeError):
    """Raised when a downloaded file fails size or sha256 verification against the lock."""


class Downloader(Protocol):
    def __call__(self, *, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path: ...


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
    # hf_hub_download resumes partial downloads automatically via its cache's .incomplete files.
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(local_dir),
    )
    return Path(path)


def load_lock(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def download_model(
    model: dict[str, Any],
    models_dir: Path,
    downloader: Downloader | None = None,
) -> list[str]:
    """Downloads and verifies every file for one lock entry; returns the filenames written
    (already-verified files on disk are left alone, so this is safe to re-run)."""
    if downloader is None:
        # Looked up at call time (not bound as a default) so tests can monkeypatch it.
        downloader = _default_downloader
    repo_id = model["repo_id"]
    revision = model["revision_sha"] or model["revision_requested"]
    local_dir = models_dir / repo_id
    written: list[str] = []
    for file_entry in model["files"]:
        filename = file_entry["filename"]
        expected_sha = file_entry.get("sha256")
        expected_size = file_entry.get("size_bytes")
        dest = local_dir / filename

        if dest.exists() and expected_sha and sha256_of(dest) == expected_sha:
            written.append(filename)
            continue

        path = Path(
            downloader(repo_id=repo_id, filename=filename, revision=revision, local_dir=local_dir)
        )

        actual_size = path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise DownloadError(
                f"{repo_id}/{filename}: size mismatch (expected {expected_size}, got {actual_size})"
            )
        if expected_sha:
            actual_sha = sha256_of(path)
            if actual_sha != expected_sha:
                raise DownloadError(
                    f"{repo_id}/{filename}: sha256 mismatch "
                    f"(expected {expected_sha}, got {actual_sha})"
                )
        written.append(filename)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Skip offline-only models (PRD §4 offline table)",
    )
    parser.add_argument(
        "--intent",
        action="append",
        default=None,
        help="Only these lock intents (repeatable), e.g. --intent extractor_b_train_base",
    )
    args = parser.parse_args(argv)

    lock = load_lock(args.lock)
    if args.intent:
        unknown = sorted(set(args.intent) - {m["intent"] for m in lock["models"]})
        if unknown:
            print(f"error: intents not in {args.lock}: {unknown}", file=sys.stderr)
            return 1
    failures: list[str] = []
    done = 0
    for model in lock["models"]:
        if args.runtime_only and not model.get("runtime", False):
            continue
        if args.intent and model["intent"] not in args.intent:
            continue
        done += 1
        print(f"==> {model['intent']} ({model['repo_id']})", flush=True)
        try:
            written = download_model(model, args.models_dir)
            print(f"    ok: {written}", flush=True)
        except DownloadError as exc:
            print(f"    FAILED: {exc}", file=sys.stderr, flush=True)
            failures.append(model["intent"])

    if failures:
        print(f"error: {len(failures)} model(s) failed verification: {failures}", file=sys.stderr)
        return 1
    print(f"all {done} model(s) verified under {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
