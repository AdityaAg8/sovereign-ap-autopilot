"""Unit tests for `factory.build`: deterministic exact-count stratified splits/fractions, plan
construction, and a small-N (<=5 invoices) end-to-end `generate_dataset` run covering the
manifest/splits registration path (PRD §10 M2.3)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from evals.splits import empty_splits, load_splits, register_documents, validate_splits
from factory.build import (
    _AR_FIRST_LAYOUT_IDS,
    _OTHER_LAYOUT_IDS,
    _assign_splits,
    _stratified_subset,
    build_plan,
    generate_dataset,
    update_manifest_and_splits,
)
from factory.layouts import TaxCase


def test_stratified_subset_exact_count() -> None:
    subset = _stratified_subset(200, 0.25, seed=1, salt="arabic_first")
    assert len(subset) == 50
    assert all(0 <= i < 200 for i in subset)


def test_stratified_subset_is_deterministic() -> None:
    a = _stratified_subset(200, 0.3, seed=1, salt="scan")
    b = _stratified_subset(200, 0.3, seed=1, salt="scan")
    assert a == b


def test_stratified_subset_different_salt_gives_a_different_partition() -> None:
    a = _stratified_subset(200, 0.25, seed=1, salt="arabic_first")
    b = _stratified_subset(200, 0.25, seed=1, salt="scan")
    assert a != b


def test_stratified_subset_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        _stratified_subset(10, 1.5, seed=1, salt="x")


def test_assign_splits_exact_counts() -> None:
    splits = _assign_splits(5000, 3500, 750, 750, seed=1)
    counts = Counter(splits)
    assert counts["train"] == 3500
    assert counts["cal"] == 750
    assert counts["test"] == 750
    assert len(splits) == 5000


def test_assign_splits_is_deterministic() -> None:
    a = _assign_splits(500, 350, 75, 75, seed=7)
    b = _assign_splits(500, 350, 75, 75, seed=7)
    assert a == b


def test_assign_splits_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="must equal n_total"):
        _assign_splits(10, 5, 3, 1, seed=1)


def test_build_plan_exact_fractions_and_ranges() -> None:
    plans = build_plan(
        n_total=200,
        n_train=140,
        n_cal=30,
        n_test=30,
        arabic_first_fraction=0.25,
        scan_fraction=0.30,
        n_vendors=5,
        seed=123,
    )
    assert len(plans) == 200
    arabic_first_plans = [p for p in plans if p.arabic_first]
    assert len(arabic_first_plans) == 50
    assert all(p.layout_id in _AR_FIRST_LAYOUT_IDS for p in arabic_first_plans)
    other_plans = [p for p in plans if not p.arabic_first]
    assert all(p.layout_id in _OTHER_LAYOUT_IDS for p in other_plans)
    assert sum(1 for p in plans if p.is_scan) == 60

    for plan in plans:
        assert 1 <= plan.n_lines <= 40
        assert 0 <= plan.vendor_index < 5
        assert TaxCase(plan.tax_case) in TaxCase

    counts = Counter(p.split for p in plans)
    assert counts["train"] == 140
    assert counts["cal"] == 30
    assert counts["test"] == 30


def test_build_plan_is_deterministic_for_the_same_seed() -> None:
    kwargs = dict(
        n_total=50,
        n_train=30,
        n_cal=10,
        n_test=10,
        arabic_first_fraction=0.25,
        scan_fraction=0.3,
        n_vendors=5,
        seed=9,
    )
    assert build_plan(**kwargs) == build_plan(**kwargs)  # type: ignore[arg-type]


def test_build_plan_rejects_non_positive_vendor_count() -> None:
    with pytest.raises(ValueError):
        build_plan(
            n_total=10,
            n_train=8,
            n_cal=1,
            n_test=1,
            arabic_first_fraction=0.25,
            scan_fraction=0.3,
            n_vendors=0,
            seed=1,
        )


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> list:  # type: ignore[type-arg]
    out_dir = tmp_path_factory.mktemp("uae_dataset")
    return generate_dataset(
        seed=555,
        out_dir=out_dir,
        n_total=5,
        n_train=3,
        n_cal=1,
        n_test=1,
        n_vendors=4,
        n_users=4,
        workers=1,
        scale=1.0,
    )


def test_generate_dataset_small_n_writes_expected_files(small_dataset: list) -> None:  # type: ignore[type-arg]
    assert len(small_dataset) == 5
    counts = Counter(r.split for r in small_dataset)
    assert counts["train"] == 3
    assert counts["cal"] == 1
    assert counts["test"] == 1

    all_problems = [problem for r in small_dataset for problem in r.bbox_problems]
    assert all_problems == []
    for result in small_dataset:
        assert len(result.sha256) == 64
        int(result.sha256, 16)  # valid hex


def test_generate_dataset_files_on_disk(tmp_path: Path) -> None:
    results = generate_dataset(
        seed=777,
        out_dir=tmp_path,
        n_total=3,
        n_train=1,
        n_cal=1,
        n_test=1,
        n_vendors=3,
        n_users=4,
        workers=1,
        scale=1.0,
    )
    assert len(results) == 3
    for result in results:
        split_dir = tmp_path / result.split
        pdf_path = split_dir / f"{result.sha256}.pdf"
        labels_path = split_dir / f"{result.sha256}.labels.json"
        truth_path = split_dir / f"{result.sha256}.truth.json"
        assert pdf_path.is_file()
        assert labels_path.is_file()
        assert truth_path.is_file()
        assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == result.sha256
        if result.is_scan:
            scan_pages = sorted(split_dir.glob(f"{result.sha256}.page*.scan.png"))
            assert len(scan_pages) == result.n_pages


def test_generate_dataset_is_deterministic_for_the_same_seed(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    kwargs = dict(
        seed=42,
        n_total=3,
        n_train=1,
        n_cal=1,
        n_test=1,
        n_vendors=3,
        n_users=4,
        workers=1,
        scale=1.0,
    )
    first = generate_dataset(out_dir=tmp_path_factory.mktemp("a"), **kwargs)  # type: ignore[arg-type]
    second = generate_dataset(out_dir=tmp_path_factory.mktemp("b"), **kwargs)  # type: ignore[arg-type]
    assert [r.sha256 for r in first] == [r.sha256 for r in second]
    assert [r.split for r in first] == [r.split for r in second]


def test_update_manifest_and_splits_registers_documents_without_leakage(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "uae"
    results = generate_dataset(
        seed=321,
        out_dir=out_dir,
        n_total=4,
        n_train=2,
        n_cal=1,
        n_test=1,
        n_vendors=3,
        n_users=4,
        workers=1,
        scale=1.0,
    )
    manifest_path = tmp_path / "data" / "manifest.json"
    splits_path = tmp_path / "evals" / "splits.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    update_manifest_and_splits(
        results,
        out_dir=out_dir,
        manifest_path=manifest_path,
        splits_path=splits_path,
        seed=321,
    )

    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["datasets"]["uae"]["status"] == "present"
    assert len(manifest_data["datasets"]["uae"]["files"]) == 4

    splits_file = load_splits(splits_path)
    uae = splits_file.sets["uae"]
    assert len(uae["train"]) == 2
    assert len(uae["cal"]) == 1
    assert len(uae["test"]) == 1
    assert validate_splits(splits_file) == []


def test_register_documents_never_reshuffles_an_existing_sha() -> None:
    splits = empty_splits()
    splits = register_documents(splits, "uae", {"a" * 64: "train"})
    splits = register_documents(splits, "uae", {"a" * 64: "test"})
    assert "a" * 64 in splits.sets["uae"]["train"]
    assert "a" * 64 not in splits.sets["uae"]["test"]
