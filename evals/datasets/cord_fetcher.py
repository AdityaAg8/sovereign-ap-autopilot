"""Fetches CORD v2 (HF `naver-clova-ix/cord-v2`, CC-BY-4.0) at a pinned revision into data/cord/,
verifies each parquet shard against its pinned sha256, and unpacks one image plus one index line
per receipt so the real-data track can parse and score them (ADR-0019 §1, §8).
The 1.8 GB train split needs `allow_large` (the >1 GB download rule)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Final, Literal

from apx.common.errors import ApxError
from evals.datasets.common import sha256_file
from evals.datasets.manifest import DatasetEntry, DatasetStatus, ManifestFile, Split

DATASET_NAME: Final = "cord"
HF_REPO_ID: Final = "naver-clova-ix/cord-v2"
# Pinned dataset commit (HF `dataset_info(...).sha`, read 2026-09-29).
REVISION: Final = "7f0115a4b758a71d6473b8d085751692da2fef98"
SOURCE_URL: Final = f"https://huggingface.co/datasets/{HF_REPO_ID}"
LICENSE: Final = "CC-BY-4.0"

CordSplit = Literal["validation", "test", "train"]
CORD_SPLITS: Final[tuple[CordSplit, ...]] = ("validation", "test", "train")
LARGE_SPLITS: Final[frozenset[CordSplit]] = frozenset({"train"})
MANIFEST_SPLIT: Final[dict[CordSplit, Split]] = {
    "validation": "val",
    "test": "test",
    "train": "train",
}
INDEX_NAME: Final = "index.jsonl"


class CordShard:
    """One pinned parquet shard: its repo path, LFS sha256 and size at REVISION."""

    __slots__ = ("bytes", "filename", "sha256")

    def __init__(self, filename: str, sha256: str, size: int) -> None:
        self.filename = filename
        self.sha256 = sha256
        self.bytes = size


SHARDS: Final[dict[CordSplit, tuple[CordShard, ...]]] = {
    "test": (
        CordShard(
            "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
            "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7",
            234202795,
        ),
    ),
    "validation": (
        CordShard(
            "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
            "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15",
            242080800,
        ),
    ),
    "train": (
        CordShard(
            "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
            "da3994eee1bf9bd3c57f0d53a72c3a6812c8696c5ba26245987949ddf73483cc",
            490224630,
        ),
        CordShard(
            "data/train-00001-of-00004-7dbbe248962764c5.parquet",
            "cce4def16a0d6a6c75f80be712f7494c56c318a8829b712f5c62650155c9e58e",
            441418432,
        ),
        CordShard(
            "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
            "591e2db8fe8b1d364b054f46e8c375b7f00e72578914ba46a573b6858162cab2",
            443802181,
        ),
        CordShard(
            "data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
            "1ffd9de8d6fbcee7630fd4cdfedff05b9b7fabc0fae4fc17557c5fe7cf178748",
            455555434,
        ),
    ),
}

# (repo filename, local dir) -> local path; the default downloads through huggingface_hub.
Downloader = Callable[[str, Path], Path]


class CordFetchError(ApxError):
    """A CORD shard is missing, does not match its pinned sha256, or a large split was
    requested without `allow_large`."""


def _hf_download(filename: str, local_dir: Path) -> Path:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=REVISION,
            local_dir=local_dir,
        )
    )


def _image_suffix(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    raise CordFetchError("CORD image is neither PNG nor JPEG")


def iter_rows(parquet: Path) -> Iterator[tuple[bytes, str]]:
    """(image bytes, ground_truth JSON string) per row of one CORD parquet shard."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["image", "ground_truth"])
    for image, truth in zip(
        table.column("image").to_pylist(), table.column("ground_truth").to_pylist(), strict=True
    ):
        data = image["bytes"] if isinstance(image, dict) else image
        if not isinstance(data, bytes) or not isinstance(truth, str):
            raise CordFetchError(f"{parquet.name}: unexpected row types")
        yield data, truth


def _relative(path: Path, data_root: Path) -> str:
    """Manifest path `data/...`, relative to the data root's parent (the repo root)."""
    return path.resolve().relative_to(data_root.resolve().parent).as_posix()


def _unpack_split(
    split: CordSplit, parquets: Sequence[Path], data_root: Path
) -> list[ManifestFile]:
    """Writes `<sha256><ext>` images and index.jsonl (row, sha256, image, ground_truth) for one
    split; returns one per-document manifest entry per distinct image."""
    out_dir = data_root / DATASET_NAME / split
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    files: dict[str, ManifestFile] = {}
    row = 0
    for parquet in parquets:
        for data, truth in iter_rows(parquet):
            sha = hashlib.sha256(data).hexdigest()
            image = out_dir / f"{sha}{_image_suffix(data)}"
            if not image.exists():
                image.write_bytes(data)
            lines.append(
                json.dumps(
                    {
                        "row": row,
                        "sha256": sha,
                        "image": image.name,
                        "ground_truth": json.loads(truth),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            files.setdefault(
                sha,
                ManifestFile(
                    path=_relative(image, data_root),
                    sha256=sha,
                    bytes=len(data),
                    split=MANIFEST_SPLIT[split],
                    doc_id=f"{split}-{row:04d}",
                ),
            )
            row += 1
    index = out_dir / INDEX_NAME
    index.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return list(files.values())


def fetch_cord(
    splits: Sequence[CordSplit],
    *,
    allow_large: bool = False,
    data_root: Path,
    download: Downloader | None = None,
) -> DatasetEntry:
    """Downloads the pinned parquet shards of `splits` into data/cord/parquet/, checks each
    sha256, unpacks images plus index.jsonl per split, and returns the manifest entry (shards
    with `doc_id=None`, one entry per receipt image). 'train' needs `allow_large`."""
    requested = list(dict.fromkeys(splits))
    unknown = [s for s in requested if s not in SHARDS]
    if unknown:
        raise CordFetchError(f"unknown CORD split(s) {unknown}; choices: {list(CORD_SPLITS)}")
    large = [s for s in requested if s in LARGE_SPLITS]
    if large and not allow_large:
        raise CordFetchError(
            f"CORD split(s) {large} exceed 1 GB; pass --allow-large once the owner approved it"
        )
    fetch = download or _hf_download
    parquet_dir = data_root / DATASET_NAME / "parquet"
    files: list[ManifestFile] = []
    for split in requested:
        paths: list[Path] = []
        for shard in SHARDS[split]:
            path = fetch(shard.filename, parquet_dir)
            actual = sha256_file(path)
            if actual != shard.sha256:
                raise CordFetchError(
                    f"{shard.filename}: sha256 {actual} != pinned {shard.sha256} "
                    f"(revision {REVISION})"
                )
            paths.append(path)
            files.append(
                ManifestFile(
                    path=_relative(path, data_root),
                    sha256=actual,
                    bytes=path.stat().st_size,
                    split=MANIFEST_SPLIT[split],
                    doc_id=None,
                )
            )
        files.extend(_unpack_split(split, paths, data_root))
    return DatasetEntry(
        status=DatasetStatus.PRESENT,
        source=SOURCE_URL,
        version=f"{REVISION}:{','.join(requested)}",
        license=LICENSE,
        blocked=None,
        files=files,
    )


def merge_entry(old: DatasetEntry | None, new: DatasetEntry) -> DatasetEntry:
    """Keeps files of splits fetched earlier (e.g. test before train) next to the new ones, so
    a later `--splits train` run never drops test/validation from the manifest."""
    if old is None or old.status is not DatasetStatus.PRESENT:
        return new
    fetched = {f.split for f in new.files}
    kept = [f for f in old.files if f.split not in fetched]
    names = {*old.version.partition(":")[2].split(","), *new.version.partition(":")[2].split(",")}
    splits = [s for s in CORD_SPLITS if s in names]
    return new.model_copy(
        update={"files": [*kept, *new.files], "version": f"{REVISION}:{','.join(splits)}"}
    )


def fetch(data_root: Path) -> DatasetEntry:
    """Default `fetch --set cord` (no --splits): the two small hold-out splits only."""
    return fetch_cord(("test", "validation"), data_root=data_root)
