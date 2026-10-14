"""Fetches the FATURA template-generated invoice dataset from Zenodo record 8261508 into
data/fatura/, extracting the archive and listing one manifest entry per invoice image with its
official Strat#1 train/test split, or a seeded deterministic fallback for any image the
official CSVs omit (I5)."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

from evals.datasets.common import download_resumable, repo_relative_posix, sha256_file
from evals.datasets.manifest import DatasetEntry, DatasetStatus, ManifestFile, Split

DATASET_NAME = "fatura"
ZENODO_RECORD = "8261508"
DOWNLOAD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/FATURA.zip/content"
SOURCE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"
LICENSE = "CC-BY-4.0"

# Layout inside the extracted archive (FATURA's own directory names).
IMAGES_DIR = "invoices_dataset_final/images"
STRAT1_TRAIN_CSV = "invoices_dataset_final/strat1_train.csv"
STRAT1_TEST_CSV = "invoices_dataset_final/strat1_test.csv"

# Any image the official Strat#1 CSVs do not list falls back to this seeded, deterministic
# split so every document still gets a stable assignment (PRD ADR-0005: splits are fixed by
# document identity and never reshuffled). ~1/7 test matches Strat#1's own ~14% test share.
FALLBACK_SPLIT_SALT = "fatura-strat1-fallback-v1"
FALLBACK_TEST_MODULUS = 7


def _read_split_csv(path: Path) -> frozenset[str]:
    """Returns the set of image stems (filename without extension) listed in a
    strat1_{train,test}.csv `img_path` column; empty if the file is absent."""
    if not path.exists():
        return frozenset()
    with path.open(newline="", encoding="utf-8") as handle:
        return frozenset(Path(row["img_path"]).stem for row in csv.DictReader(handle))


def _fallback_split(doc_id: str) -> Split:
    """Deterministic split for a document the official CSVs do not cover: a pure function of
    `doc_id`, so registering it never depends on, or disturbs, any other document's split."""
    digest = hashlib.sha256(f"{FALLBACK_SPLIT_SALT}:{doc_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % FALLBACK_TEST_MODULUS
    return "test" if bucket == 0 else "train"


def fetch(data_root: Path) -> DatasetEntry:
    target = data_root / DATASET_NAME
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "FATURA.zip"
    download_resumable(DOWNLOAD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)

    train_ids = _read_split_csv(target / STRAT1_TRAIN_CSV)
    test_ids = _read_split_csv(target / STRAT1_TEST_CSV)

    files = [
        ManifestFile(
            path=repo_relative_posix(zip_path),
            sha256=sha256_file(zip_path),
            bytes=zip_path.stat().st_size,
            split=None,
            doc_id=None,
        )
    ]
    images_dir = target / IMAGES_DIR
    for image_path in sorted(images_dir.glob("*.jpg")):
        doc_id = image_path.stem
        split: Split
        if doc_id in test_ids:
            split = "test"
        elif doc_id in train_ids:
            split = "train"
        else:
            split = _fallback_split(doc_id)
        files.append(
            ManifestFile(
                path=repo_relative_posix(image_path),
                sha256=sha256_file(image_path),
                bytes=image_path.stat().st_size,
                split=split,
                doc_id=doc_id,
            )
        )

    return DatasetEntry(
        status=DatasetStatus.PRESENT,
        source=SOURCE_URL,
        version=ZENODO_RECORD,
        license=LICENSE,
        blocked=None,
        files=files,
    )
