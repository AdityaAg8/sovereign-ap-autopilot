"""Unit tests for evals.datasets.cord_fetcher (ADR-0019 §8): the >1 GB train split needs
`allow_large`, every shard is checked against its pinned sha256, receipts are unpacked to one image
plus an index line, and a later fetch keeps earlier splits in the manifest. The downloader is a
local fake that serves a tiny generated parquet; no network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evals.datasets import cord_fetcher
from evals.datasets.cord_fetcher import CordFetchError, CordShard, fetch_cord, merge_entry
from evals.datasets.manifest import DatasetStatus

PNG = b"\x89PNG\r\n\x1a\n" + b"receipt-1"
JPG = b"\xff\xd8" + b"receipt-2"


def _parquet(path: Path, images: list[bytes]) -> None:
    truth = [
        json.dumps({"gt_parse": {"total": {"total_price": str(i)}}}) for i in range(len(images))
    ]
    table = pa.table(
        {
            "image": [{"bytes": b, "path": None} for b in images],
            "ground_truth": truth,
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """A fake hub: one shard per split, the pinned shas replaced by the generated files'."""
    hub = tmp_path / "hub"
    hub.mkdir()
    shards: dict[str, tuple[CordShard, ...]] = {}
    for split, images in (("test", [PNG, JPG, PNG]), ("validation", [JPG]), ("train", [PNG])):
        name = f"data/{split}-0.parquet"
        source = hub / f"{split}.parquet"
        _parquet(source, images)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        shards[split] = (CordShard(name, digest, source.stat().st_size),)
    monkeypatch.setattr(cord_fetcher, "SHARDS", shards)
    fetched: list[str] = []

    def download(filename: str, local_dir: Path) -> Path:
        fetched.append(filename)
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((hub / f"{filename.split('/')[1].split('-')[0]}.parquet").read_bytes())
        return target

    monkeypatch.setattr(cord_fetcher, "_hf_download", download)
    return tmp_path / "data", fetched


def test_train_needs_allow_large(served: tuple[Path, list[str]]) -> None:
    data_root, fetched = served
    with pytest.raises(CordFetchError, match="allow-large"):
        fetch_cord(["train"], data_root=data_root)
    assert fetched == []


def test_fetch_unpacks_images_index_and_manifest(served: tuple[Path, list[str]]) -> None:
    data_root, _ = served
    entry = fetch_cord(["test"], data_root=data_root)
    assert entry.status is DatasetStatus.PRESENT
    assert entry.version.endswith(":test")
    index = (data_root / "cord" / "test" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index) == 3
    first = json.loads(index[0])
    assert first["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert (data_root / "cord" / "test" / first["image"]).read_bytes() == PNG
    assert json.loads(index[1])["image"].endswith(".jpg")
    docs = [f for f in entry.files if f.doc_id is not None]
    assert len(docs) == 2  # the duplicated image is one document
    assert {f.split for f in entry.files} == {"test"}
    assert all(f.path.startswith("data/cord/") for f in entry.files)


def test_sha_mismatch_is_refused(
    served: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _ = served
    shard = cord_fetcher.SHARDS["test"][0]
    monkeypatch.setitem(
        cord_fetcher.SHARDS, "test", (CordShard(shard.filename, "0" * 64, shard.bytes),)
    )
    with pytest.raises(CordFetchError, match="pinned"):
        fetch_cord(["test"], data_root=data_root)


def test_merge_keeps_earlier_splits(served: tuple[Path, list[str]]) -> None:
    data_root, _ = served
    first = fetch_cord(["test", "validation"], data_root=data_root)
    later = fetch_cord(["train"], allow_large=True, data_root=data_root)
    merged = merge_entry(first, later)
    assert {f.split for f in merged.files} == {"test", "val", "train"}
    assert merged.version.endswith(":validation,test,train")
    assert merge_entry(None, later) == later


def test_unknown_split_is_refused(served: tuple[Path, list[str]]) -> None:
    with pytest.raises(CordFetchError, match="unknown"):
        fetch_cord(["dev"], data_root=served[0])  # type: ignore[list-item]


def test_pinned_revision_and_shards_are_complete() -> None:
    assert len(cord_fetcher.REVISION) == 40
    assert sum(s.bytes for shards in cord_fetcher.SHARDS.values() for s in shards) > 2_000_000_000
    assert len(cord_fetcher.SHARDS["train"]) == 4
