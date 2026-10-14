"""Fetches the public deepset/prompt-injections dataset (Apache-2.0) at a pinned Hub revision,
verifying every parquet file's sha256, for the M6.4 injection-guard training/eval mix.
Runs on the host at fetch time only; nothing at runtime downloads it."""

from __future__ import annotations

from pathlib import Path

from evals.datasets.common import download_resumable, repo_relative_posix, sha256_file
from evals.datasets.manifest import DatasetEntry, DatasetStatus, ManifestFile, Split

DATASET_NAME = "prompt_injections"
REPO_ID = "deepset/prompt-injections"
REVISION = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
LICENSE = "Apache-2.0 (deepset/prompt-injections)"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"

# (Hub path, split, sha256, size) resolved from the Hub API at REVISION; never trusted blindly:
# each download is re-hashed and rejected on mismatch.
PINNED_FILES: tuple[tuple[str, Split, str, int], ...] = (
    (
        "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
        "train",
        "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
        40323,
    ),
    (
        "data/test-00000-of-00001-701d16158af87368.parquet",
        "test",
        "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
        10892,
    ),
)


class ChecksumMismatchError(RuntimeError):
    """A downloaded file's sha256 differs from the pinned value; the file is removed."""


def dataset_dir(data_root: Path) -> Path:
    return data_root / "external" / "prompt_injections"


def local_path(data_root: Path, split: Split) -> Path:
    """Where the pinned parquet for `split` lives after a fetch."""
    return dataset_dir(data_root) / f"{split}.parquet"


def fetch(data_root: Path) -> DatasetEntry:
    files: list[ManifestFile] = []
    for hub_path, split, sha, size in PINNED_FILES:
        url = f"{SOURCE_URL}/resolve/{REVISION}/{hub_path}"
        dest = download_resumable(url, local_path(data_root, split))
        actual = sha256_file(dest)
        if actual != sha:
            dest.unlink()
            raise ChecksumMismatchError(f"{hub_path}: expected sha256 {sha}, got {actual}")
        files.append(
            ManifestFile(
                path=repo_relative_posix(dest),
                sha256=sha,
                bytes=size,
                split=split,
            )
        )
    return DatasetEntry(
        status=DatasetStatus.PRESENT,
        source=SOURCE_URL,
        version=REVISION,
        license=LICENSE,
        blocked=None,
        files=files,
    )
