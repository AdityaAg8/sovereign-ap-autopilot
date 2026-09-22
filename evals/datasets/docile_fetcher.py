"""Fetches the DocILE annotated-trainval and test splits (never the 1M-doc unlabeled set)
from the token-gated Rossum S3 bucket into data/docile/, deferring with a BLOCKED.md entry
when DOCILE_TOKEN is absent instead of failing the run (PRD §5 data table, §10 M1.1).
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from evals.datasets.common import (
    append_blocked_entry,
    download_resumable,
    repo_relative_posix,
    sha256_file,
)
from evals.datasets.manifest import DatasetEntry, DatasetStatus, ManifestFile, Split

DATASET_NAME = "docile"
ENV_VAR = "DOCILE_TOKEN"
SOURCE_URL = "https://docile.rossum.ai/"
LICENSE = "Research use per DocILE terms (docile.rossum.ai); see rossumai/docile for details"
DOWNLOAD_URL_TEMPLATE = (
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{token}/{dataset}.zip"
)
# Split identifiers as accepted by rossumai/docile's download_dataset.sh; the 1M-doc
# "unlabeled" set is intentionally excluded (PRD §5).
REMOTE_SPLITS: tuple[tuple[str, Split | None], ...] = (
    ("annotated-trainval", None),  # bundles both train and val documents
    ("test", "test"),
)
FETCH_COMMAND = f"{ENV_VAR}=<your-token> uv run python -m evals.datasets fetch --set docile"


def fetch(data_root: Path) -> DatasetEntry:
    token = os.environ.get(ENV_VAR, "").strip()
    if not token:
        append_blocked_entry(
            tag="dataset:docile",
            title="DocILE annotated-trainval+test needs a DOCILE_TOKEN",
            why=(
                "DocILE is token-gated (PRD §5); DOCILE_TOKEN is not set, so the "
                "annotated-trainval and test splits cannot be downloaded. Extractor B "
                "training/eval degrades gracefully without it (PRD §9)."
            ),
            command=FETCH_COMMAND,
        )
        return DatasetEntry(
            status=DatasetStatus.DEFERRED,
            source=SOURCE_URL,
            version="annotated-trainval+test",
            license=LICENSE,
            blocked=f"needs {ENV_VAR} (see docs/runbook.md)",
            files=[],
        )

    target = data_root / DATASET_NAME
    target.mkdir(parents=True, exist_ok=True)
    files: list[ManifestFile] = []
    for remote_name, split in REMOTE_SPLITS:
        url = DOWNLOAD_URL_TEMPLATE.format(token=token, dataset=remote_name)
        zip_path = target / f"{remote_name}.zip"
        download_resumable(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
        files.append(
            ManifestFile(
                path=repo_relative_posix(zip_path),
                sha256=sha256_file(zip_path),
                bytes=zip_path.stat().st_size,
                split=split,
                doc_id=None,
            )
        )
    return DatasetEntry(
        status=DatasetStatus.PRESENT,
        source=SOURCE_URL,
        version="annotated-trainval+test",
        license=LICENSE,
        blocked=None,
        files=files,
    )
