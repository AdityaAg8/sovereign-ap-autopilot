"""Fetches the ExtractBench short split (252 docs: metadata plus source PDFs) from HF
`llamaindex/ExtractBench` and clones the official `run-llama/ExtractBench` scoring framework
pinned to a fixed commit into data/external/ (PRD §5 data table, §10 M1.1). Eval-only; not
used for training.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from evals.datasets.common import repo_relative_posix, sha256_file
from evals.datasets.manifest import DatasetEntry, DatasetStatus, ManifestFile

DATASET_NAME = "extractbench"
HF_REPO_ID = "llamaindex/ExtractBench"
# Only the "short" config/split (252 docs, PRD §5): its metadata file and referenced PDFs.
HF_ALLOW_PATTERNS = ["short.jsonl", "docs/short/*"]
FRAMEWORK_REPO = "https://github.com/run-llama/ExtractBench.git"
FRAMEWORK_COMMIT = "11937d102cad9d9b815044888d55e7ce343d3924"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
LICENSE = "Apache-2.0"


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def _clone_framework(external_dir: Path) -> str:
    """Clones (or checks out) run-llama/ExtractBench at FRAMEWORK_COMMIT under external_dir;
    returns the checked-out commit sha."""
    dest = external_dir / "ExtractBench"
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", FRAMEWORK_REPO, str(dest))
    head = _git("-C", str(dest), "rev-parse", "HEAD")
    if head != FRAMEWORK_COMMIT:
        _git("-C", str(dest), "fetch", "--depth", "1", "origin", FRAMEWORK_COMMIT)
        _git("-C", str(dest), "checkout", FRAMEWORK_COMMIT)
        head = _git("-C", str(dest), "rev-parse", "HEAD")
    return head


def fetch(data_root: Path) -> DatasetEntry:
    target = data_root / DATASET_NAME
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=HF_ALLOW_PATTERNS,
        local_dir=target,
    )
    files = [
        ManifestFile(
            path=repo_relative_posix(path),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            split="eval",
            doc_id=path.stem if path.suffix == ".pdf" else None,
        )
        for path in sorted(target.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(target).parts
    ]
    framework_commit = _clone_framework(data_root / "external")
    return DatasetEntry(
        status=DatasetStatus.PRESENT,
        source=SOURCE_URL,
        version=f"short@framework-{framework_commit[:12]}",
        license=LICENSE,
        blocked=None,
        files=files,
    )
