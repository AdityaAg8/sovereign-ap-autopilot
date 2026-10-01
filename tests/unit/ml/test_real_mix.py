"""Tests for the ADR-0019 §9.2 `REAL_MIX` rounds in Extractor B data prep: the CORD v2 train
loader over a synthetic data root, the R1/R2 source sets with typed refusals when a required
source is missing (before any page is rendered), the UAE >= 45% floor at the ADR's exact counts,
and UAE/FATURA rows that stay byte-identical to the v1 prep."""

from __future__ import annotations

import io
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ml.extractor_b import data
from ml.extractor_b.guards import manifest_line
from ml.extractor_b.interfaces import (
    DataLeakError,
    MixSourceMissingError,
    SampleSource,
    TrainConfig,
    TrainingError,
    TrainingSample,
)
from tests.unit.ml.conftest import StubCounter, sha
from tests.unit.ml.test_data import _fatura_doc, _uae_doc

GT: dict[str, Any] = {
    "menu": [{"nm": "NASI GORENG", "cnt": "2", "price": "50.000"}],
    "sub_total": {"subtotal_price": "50.000", "tax_price": "5.000"},
    "total": {"total_price": "55.000", "cashprice": "60.000"},
}


def _receipt_image(seed: str) -> Image.Image:
    """A 480x960 receipt stand-in whose 8x8 block pattern (and so pHash) is unique per seed."""
    rng = random.Random(seed)
    small = Image.new("L", (8, 16))
    small.putdata([rng.choice((40, 230)) for _ in range(8 * 16)])
    return small.resize((480, 960), Image.Resampling.NEAREST).convert("RGB")


def _cord_doc(
    root: Path, doc_id: str, *, split: str = "train", truth: bool = True
) -> dict[str, Any]:
    """One receipt in the `evals/datasets/cord_fetcher.py` layout: `<sha256>.png` plus a line in
    the split's `index.jsonl` (unless `truth` is False)."""
    folder = root / "cord" / split
    folder.mkdir(parents=True, exist_ok=True)
    digest = sha(f"cord-{split}-{doc_id}")
    _receipt_image(digest).save(folder / f"{digest}.png")
    if truth:
        ground_truth = {"gt_parse": GT, "meta": {"split": split}, "valid_line": []}
        record = {
            "row": 0,
            "sha256": digest,
            "image": f"{digest}.png",
            "ground_truth": ground_truth,
        }
        with (folder / "index.jsonl").open("a", encoding="utf-8") as index:
            index.write(json.dumps(record) + "\n")
    return {
        "path": f"data/cord/{split}/{digest}.png",
        "sha256": digest,
        "bytes": 1,
        "split": split,
        "doc_id": doc_id,
    }


def _manifest(root: Path, fatura: list[dict[str, Any]], cord: list[dict[str, Any]]) -> None:
    entry = {"status": "present", "version": "v", "license": "CC-BY-4.0", "blocked": None}
    datasets: dict[str, Any] = {"fatura": {**entry, "source": "zenodo", "files": fatura}}
    if cord:
        datasets["cord"] = {**entry, "source": "naver-clova-ix/cord-v2", "files": cord}
    body = {"generated_at": "2026-09-29T00:00:00Z", "datasets": datasets}
    (root / "manifest.json").write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "data"
    r.mkdir()
    return r


def _holdouts(root: Path) -> None:
    """CORD validation and test indexes: the hold-out copy check needs both."""
    for split in ("validation", "test"):
        _cord_doc(root, f"receipt_{split}_h", split=split)


def _world(root: Path, *, n_cord: int = 3) -> frozenset[str]:
    uae = [_uae_doc(root, str(n)) for n in range(40)]
    fatura = [_fatura_doc(root, f"Template1_Instance{n}") for n in range(3)]
    cord = [_cord_doc(root, f"receipt_{n:05d}") for n in range(n_cord)]
    cord.append(_cord_doc(root, "receipt_test_0", split="test"))  # a hold-out: never read
    _cord_doc(root, "receipt_val_0", split="validation")
    _manifest(root, fatura, cord)
    return frozenset(uae) | {str(f["sha256"]) for f in fatura + cord if f["split"] == "train"}


def test_cord_docs_reads_every_train_receipt_with_its_gt_parse(root: Path) -> None:
    shas = _world(root)
    docs = list(data.cord_docs(root, shas))
    assert [d.source for d in docs] == [SampleSource.CORD] * 3
    assert all(d.n_pages == 1 and d.document_sha256 in shas for d in docs)
    assert docs[0].target["total"] == "55.000"
    assert docs[0].target["line_items"][0]["description"] == "NASI GORENG"
    with Image.open(io.BytesIO(docs[0].pages()[0])) as img:
        assert img.size == (480, 960)


def test_cord_train_image_off_the_split_stops_prep(root: Path) -> None:
    shas = _world(root)
    stale = frozenset(s for s in shas if s != sha("cord-train-receipt_00001"))
    with pytest.raises(DataLeakError, match="1/3 CORD train images"):
        list(data.cord_docs(root, stale))


def test_cord_image_without_ground_truth_stops_prep(root: Path) -> None:
    files = [_cord_doc(root, "receipt_00000", truth=False)]
    _manifest(root, [], files)
    _holdouts(root)
    shas = frozenset({str(files[0]["sha256"])})
    with pytest.raises(TrainingError, match="unreadable CORD index"):
        list(data.cord_docs(root, shas))
    _cord_doc(root, "receipt_99999", split="train")  # an index without this receipt's row
    with pytest.raises(TrainingError, match="no gt_parse for CORD train doc receipt_00000"):
        list(data.cord_docs(root, shas))


def test_cord_duplicate_image_takes_its_first_row(root: Path) -> None:
    """CORD train has 800 rows but 798 images: the manifest lists each image once and its
    target comes from the first index row with that sha256."""
    files = [_cord_doc(root, "receipt_00000")]
    _holdouts(root)
    _manifest(root, [], files)
    second = {**GT, "total": {"total_price": "99.000"}}
    record = {
        "row": 1,
        "sha256": files[0]["sha256"],
        "image": "x.png",
        "ground_truth": {"gt_parse": second},
    }
    with (root / "cord" / "train" / "index.jsonl").open("a", encoding="utf-8") as index:
        index.write(json.dumps(record) + "\n")
    docs = list(data.cord_docs(root, frozenset({str(files[0]["sha256"])})))
    assert len(docs) == 1 and docs[0].target["total"] == "55.000"


def test_cord_absent_yields_nothing(root: Path) -> None:
    _manifest(root, [], [])
    assert list(data.cord_docs(root, frozenset())) == []


def test_r1_adds_cord_and_keeps_uae_and_fatura_rows_byte_identical(
    root: Path, tmp_path: Path
) -> None:
    shas = _world(root)
    v1_cfg, r1_cfg = TrainConfig(data_dir=tmp_path / "v1"), TrainConfig(data_dir=tmp_path / "r1")
    v1 = data.build_dataset(v1_cfg, root, train_shas=shas, counter=StubCounter())
    r1 = data.build_dataset(r1_cfg, root, train_shas=shas, counter=StubCounter(), mix="r1")

    assert {r.source for r in v1} == {SampleSource.UAE, SampleSource.FATURA}
    assert {r.source for r in r1} == {SampleSource.UAE, SampleSource.FATURA, SampleSource.CORD}
    cord = [r for r in r1 if r.source is SampleSource.CORD]
    assert len(cord) == 3 and all(r.role == "train" for r in cord)
    old = sorted(manifest_line(r) for r in v1)
    new = sorted(manifest_line(r) for r in r1 if r.source is not SampleSource.CORD)
    assert old == new
    for row in v1:
        for rel in row.image_paths:
            assert (v1_cfg.data_dir / rel).read_bytes() == (r1_cfg.data_dir / rel).read_bytes()
    v1_train = (v1_cfg.data_dir / "raw_train.jsonl").read_text("utf-8").splitlines()
    r1_train = (r1_cfg.data_dir / "raw_train.jsonl").read_text("utf-8").splitlines()
    assert [line for line in r1_train if '"source":"cord"' not in line] == v1_train
    assert (v1_cfg.data_dir / "raw_dev.jsonl").read_bytes() == (
        r1_cfg.data_dir / "raw_dev.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(("mix", "missing"), [("r1", "cord"), ("r2", "cord, docile")])
def test_a_round_without_its_real_source_fails_typed_before_rendering(
    root: Path, tmp_path: Path, mix: data.RealMix, missing: str
) -> None:
    shas = _world(root, n_cord=0)
    cfg = TrainConfig(data_dir=tmp_path / "prep")
    with pytest.raises(MixSourceMissingError, match=f"needs {missing}"):
        data.build_dataset(cfg, root, train_shas=shas, counter=StubCounter(), mix=mix)
    assert not cfg.data_dir.exists() or not any(cfg.data_dir.rglob("*.jpg"))


def test_r2_without_docile_fails_typed(root: Path, tmp_path: Path) -> None:
    shas = _world(root)  # CORD present, DocILE never fetched
    cfg = TrainConfig(data_dir=tmp_path / "prep")
    with pytest.raises(MixSourceMissingError, match="needs docile"):
        data.build_dataset(cfg, root, train_shas=shas, counter=StubCounter(), mix="r2")


# ---- check_mix at the ADR-0019 §9.2 counts --------------------------------------------------


def _rows(counts: dict[SampleSource, int], dev: int = 175) -> list[TrainingSample]:
    rows = [
        TrainingSample(
            sample_id=f"{source.value}-{n}",
            document_sha256=sha(f"{source.value}-{n}"),
            source=source,
            role="train",
            image_paths=(Path("p.jpg"),),
            target_json="{}",
            n_prompt_tokens=1,
            n_target_tokens=1,
        )
        for source, count in counts.items()
        for n in range(count)
    ]
    rows += [
        row.model_copy(update={"sample_id": f"dev-{n}", "role": "dev"})
        for n, row in enumerate(rows[:dev])
    ]
    return rows


R1 = {SampleSource.UAE: 3322, SampleSource.FATURA: 752, SampleSource.CORD: 800}
R2 = {**R1, SampleSource.DOCILE: 2500}


def _shares(counts: dict[SampleSource, int]) -> dict[SampleSource, int]:
    total = sum(counts.values())
    return {s: round(100 * n / total) for s, n in counts.items()}


def test_the_adr_mixes_pass_with_the_adr_shares() -> None:
    assert sum(R1.values()) == 4874 and sum(R2.values()) == 7374
    assert _shares(R1) == {SampleSource.UAE: 68, SampleSource.FATURA: 15, SampleSource.CORD: 16}
    assert _shares(R2) == {
        SampleSource.UAE: 45,
        SampleSource.FATURA: 10,
        SampleSource.CORD: 11,
        SampleSource.DOCILE: 34,
    }
    assert 0.45 < R2[SampleSource.UAE] / sum(R2.values()) < 0.451  # 45.05%
    data.check_mix(_rows(R1), "r1")
    data.check_mix(_rows(R2), "r2")


def test_uae_floor_holds_for_r2() -> None:
    below = {**R2, SampleSource.UAE: 2700}  # 2700 / 6752 = 40%
    with pytest.raises(TrainingError, match="below 45%"):
        data.check_mix(_rows(below), "r2")


def test_check_mix_refuses_a_missing_or_foreign_source() -> None:
    with pytest.raises(MixSourceMissingError, match="needs train rows from docile"):
        data.check_mix(_rows(R1), "r2")
    with pytest.raises(TrainingError, match="does not include docile"):
        data.check_mix(_rows(R2), "r1")
    with pytest.raises(TrainingError, match="does not include cord"):
        data.check_mix(_rows(R1), "v1")


def test_v1_is_the_default_and_unchanged() -> None:
    data.check_mix(_rows({SampleSource.UAE: 100, SampleSource.FATURA: 20}))
    assert data.MIX_SOURCES["v1"] == {SampleSource.UAE, SampleSource.DOCILE, SampleSource.FATURA}


def test_real_mix_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REAL_MIX", "r3")
    with pytest.raises(SystemExit) as exc:
        data.main(["--data-root", "nowhere"])
    assert exc.value.code == 2  # argparse rejects an unknown round before any work


def test_cord_copy_exclusions_drop_holdout_copies_redundant_and_conflicting_copies() -> None:
    """ADR-0019 amendment 2, on the image hash alone: a train photo that copies a hold-out photo
    is dropped (re-blurred copies of CORD test receipts sit in CORD train), identical-label copies
    keep the smallest sha256, and copies whose labels disagree are all dropped."""
    h = {
        "a": 0b0,
        "b": 0b1,
        "c": 0xFF00,
        "d": 0xFF01,
        "e": 0xF0F0F0,
        "f": 0xF0F0F1,
        "g": 0x0F0F0F0F0F,
        "t1": 0x0F0F0F0F0F ^ 0b11,
    }
    train = [
        ("a", Path("a"), "X"),
        ("b", Path("b"), "X"),  # one photo, same label
        ("c", Path("c"), "Y"),
        ("d", Path("d"), "Z"),  # one photo, labels disagree
        ("e", Path("e"), "W"),
        ("f", Path("f"), "W"),
        ("g", Path("g"), "V"),  # a copy of test photo t1
    ]
    copies = data.cord_copy_exclusions(
        train, [("test", "t1", Path("t1"))], phash=lambda p: h[str(p)]
    )
    assert copies.excluded == {
        "b": "redundant_copy:a",
        "c": "conflicting_copy",
        "d": "conflicting_copy",
        "f": "redundant_copy:e",
        "g": "holdout_copy:test:t1",
    }
    assert (copies.holdout_copies, copies.redundant_copies, copies.conflicting_copies) == (1, 2, 2)


def test_cord_far_photos_are_all_kept() -> None:
    train = [(s, Path(s), "X") for s in ("a", "b")]
    h = {"a": 0, "b": (1 << 5) - 1, "t": (1 << 40) - 1}  # distance 5 > 4
    copies = data.cord_copy_exclusions(train, [("test", "t", Path("t"))], phash=lambda p: h[str(p)])
    assert copies.excluded == {}


def test_cord_without_holdout_indexes_stops_prep(root: Path) -> None:
    """The copy check needs CORD validation and test; without them CORD train is never used."""
    files = [_cord_doc(root, "receipt_00000")]
    _manifest(root, [], files)
    with pytest.raises(TrainingError, match="hold-out copy check"):
        list(data.cord_docs(root, frozenset({str(files[0]["sha256"])})))


def test_cord_docs_drop_a_train_copy_of_a_test_photo(root: Path) -> None:
    files = [_cord_doc(root, "receipt_00000"), _cord_doc(root, "receipt_00001")]
    _manifest(root, [], files)
    _holdouts(root)
    leaked = str(files[1]["sha256"])
    shutil.copy(
        root / "cord" / "train" / f"{leaked}.png", next((root / "cord" / "test").glob("*.png"))
    )
    docs = list(data.cord_docs(root, frozenset(str(f["sha256"]) for f in files)))
    assert [d.document_sha256 for d in docs] == [str(files[0]["sha256"])]
