"""Tests for Extractor B data prep (ADR-0009 §1.2-1.4) on a synthetic data root: train-split
filtering, the UAE dev holdout, FATURA sampling and eligibility, page JPEGs from the serving
resize, raw manifests that pass the guards, the UAE share check, the stale-build guard and the
classified targets from the factory truth (ADR-0012)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ml.extractor_b import data
from ml.extractor_b.guards import compute_page1_phashes, parse_raw_rows, run_guards
from ml.extractor_b.interfaces import DataLeakError, SampleSource, TrainConfig, TrainingError
from tests.unit.ml.conftest import StubCounter, sha


def _uae_doc(root: Path, name: str, *, pages: int = 2, tax_case: str | None = "exempt") -> str:
    """A PDF rendered by PIL plus a factory-style label set; returns its sha."""
    doc_sha = sha(f"uae-{name}")
    folder = root / "uae" / "train"
    folder.mkdir(parents=True, exist_ok=True)
    images = [Image.new("RGB", (620, 877), (255, 255 - 20 * i, 255)) for i in range(pages)]
    images[0].save(folder / f"{doc_sha}.pdf", save_all=True, append_images=images[1:])
    labels = {
        "layout_id": "t",
        "n_pages": pages,
        "labels": [  # header on page 1, totals on the last page, as the factory prints them
            {"field": "invoice_number", "value": f"INV-{name}", "line_index": None, "page": 0},
            {"field": "issue_date", "value": "2026-01-05", "line_index": None, "page": 0},
            {"field": "supplier_name", "value": "Acme LLC", "line_index": None, "page": 0},
            {"field": "subtotal", "value": "100.00", "line_index": None, "page": pages - 1},
            {"field": "vat_total", "value": "5.00", "line_index": None, "page": pages - 1},
            {"field": "total", "value": "105.00", "line_index": None, "page": pages - 1},
            {"field": "description", "value": "Item", "line_index": 0, "page": 0},
        ],
    }
    (folder / f"{doc_sha}.labels.json").write_text(json.dumps(labels), encoding="utf-8")
    if tax_case is not None:
        truth = {"sha256": doc_sha, "tax_case": tax_case}
        (folder / f"{doc_sha}.truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return doc_sha


def _fatura_doc(root: Path, doc_id: str, *, discount: bool = False) -> dict[str, object]:
    ann_dir = root / data.FATURA_ANNOTATIONS
    img_dir = root / "fatura" / "invoices_dataset_final" / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    annotation: dict[str, object] = {
        "NUMBER": {"text": f"INVOICE # {doc_id}"},
        "DATE": {"text": "Date: 03-Jan-1994"},
        "SELLER_NAME": {"text": "Mclean-Cochran"},
        "TOTAL": {"text": "TOTAL : 441.14 EUR"},
    }
    if discount:
        annotation["DISCOUNT"] = {"text": "DISCOUNT(2%): (-) 9.39"}
    (ann_dir / f"{doc_id}.json").write_text(json.dumps(annotation), encoding="utf-8")
    Image.new("RGB", (595, 842), (250, 250, 250)).save(img_dir / f"{doc_id}.jpg")
    return {
        "path": f"data/fatura/invoices_dataset_final/images/{doc_id}.jpg",
        "sha256": sha(f"fatura-{doc_id}"),
        "bytes": 1,
        "split": "train",
        "doc_id": doc_id,
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "data"
    r.mkdir()
    return r


def _manifest(root: Path, fatura_files: list[dict[str, object]]) -> None:
    manifest = {
        "generated_at": "2026-09-23T00:00:00Z",
        "datasets": {
            "fatura": {
                "status": "present",
                "source": "https://zenodo.org/records/8261508",
                "version": "8261508",
                "license": "CC-BY-4.0",
                "blocked": None,
                "files": fatura_files,
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dev_holdout_is_stable_and_about_five_percent() -> None:
    picks = [data.is_dev(sha(str(n))) for n in range(4000)]
    assert 0.04 < sum(picks) / len(picks) < 0.06
    assert picks == [data.is_dev(sha(str(n))) for n in range(4000)]


def test_build_dataset_end_to_end(root: Path, tmp_path: Path) -> None:
    uae = [_uae_doc(root, str(n)) for n in range(40)]
    files = [_fatura_doc(root, f"Template1_Instance{n}") for n in range(3)]
    files.append(_fatura_doc(root, "Template1_Instance9", discount=True))
    files.append(_fatura_doc(root, "Template2_Instance0"))
    _manifest(root, files)
    train_shas = frozenset(uae) | frozenset(str(f["sha256"]) for f in files)

    cfg = TrainConfig(data_dir=tmp_path / "prep")
    rows = data.build_dataset(cfg, root, train_shas=train_shas, counter=StubCounter())

    by_source = {s: [r for r in rows if r.source is s] for s in SampleSource}
    assert len(by_source[SampleSource.UAE]) == 40
    assert len(by_source[SampleSource.FATURA]) == 4  # the discount one is ineligible
    assert all(r.role == "train" for r in by_source[SampleSource.FATURA])
    assert {r.role for r in by_source[SampleSource.UAE]} == {"train", "dev"}

    first = next(r for r in rows if r.source is SampleSource.UAE)
    assert len(first.image_paths) == 2
    with Image.open(cfg.data_dir / first.image_paths[0]) as img:
        assert img.format == "JPEG" and img.width % 32 == 0 and img.height % 32 == 0
    target = json.loads(first.target_json)
    assert target["line_items"][0]["description"] == "Item"
    assert target["line_items"][0]["vat_category"] == "E"
    assert target["invoice_type_code"] == "380"
    fatura = json.loads(by_source[SampleSource.FATURA][0].target_json)
    assert fatura["invoice_type_code"] == "380" and fatura["line_items"] == []
    # Exact lengths come from the counter (ADR-0011 amendment 3): visual + overhead, target.
    visual = 0
    for rel in first.image_paths:
        with Image.open(cfg.data_dir / rel) as img:
            visual += (img.width // 32) * (img.height // 32)
    assert first.n_prompt_tokens == visual + StubCounter().prompt_overhead(2)
    assert first.n_target_tokens == StubCounter().target_tokens(first.target_json)
    # Compact target JSON: no whitespace between tokens, schema key order, nulls kept.
    assert first.target_json == json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    assert ", " not in first.target_json and '": ' not in first.target_json

    raw = [
        line
        for name in ("raw_train.jsonl", "raw_dev.jsonl")
        for line in (cfg.data_dir / name).read_text(encoding="utf-8").splitlines()
    ]
    parsed, invalid = parse_raw_rows(raw)
    assert not invalid and sorted(r.sample_id for r in parsed) == sorted(r.sample_id for r in rows)
    kept, quarantined, _ = run_guards(
        parsed,
        train_split_sha256=train_shas,
        page1_phash=compute_page1_phashes(parsed, cfg.data_dir),
        cfg=cfg.guards.model_copy(update={"max_quarantine_rate": 1.0}),
        max_seq_len=4096,
        image_root=cfg.data_dir,
    )
    assert len(kept) + len(quarantined) == len(rows)
    assert quarantined == []


def test_uae_share_below_45_percent_stops(root: Path, tmp_path: Path) -> None:
    uae = [_uae_doc(root, str(n)) for n in range(2)]
    files = [_fatura_doc(root, f"Template{n}_Instance0") for n in range(5)]
    _manifest(root, files)
    shas = frozenset(uae) | frozenset(str(f["sha256"]) for f in files)
    with pytest.raises(TrainingError, match="below 45%"):
        data.build_dataset(
            TrainConfig(data_dir=tmp_path / "prep"), root, train_shas=shas, counter=StubCounter()
        )


def test_fatura_sampling_is_seeded_and_capped_per_template(root: Path) -> None:
    files = [_fatura_doc(root, f"Template3_Instance{n}") for n in range(8)]
    _manifest(root, files)
    shas = frozenset(str(f["sha256"]) for f in files)
    a = [d.document_sha256 for d in data.fatura_docs(root, shas, per_template=3)]
    b = [d.document_sha256 for d in data.fatura_docs(root, shas, per_template=3)]
    assert a == b and len(a) == 3


def test_docile_absent_yields_nothing(root: Path) -> None:
    _manifest(root, [])
    assert list(data.docile_docs(root, frozenset())) == []


def test_uae_doc_off_the_train_split_stops_prep(root: Path) -> None:
    """A factory build other than the one evals/splits.json was made from (ADR-0012 §2)."""
    on_split = _uae_doc(root, "on")
    _uae_doc(root, "not-on-train-split")
    with pytest.raises(DataLeakError, match="1/2 documents"):
        list(data.uae_docs(root, frozenset({on_split})))


def test_uae_doc_without_truth_tax_case_stops_prep(root: Path) -> None:
    doc = _uae_doc(root, "no-truth", tax_case=None)
    with pytest.raises(TrainingError, match="no tax_case"):
        list(data.uae_docs(root, frozenset({doc})))
