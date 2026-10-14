"""Pydantic model and (de)serialization for data/manifest.json: the per-dataset record of
whether an eval/training set is present or deferred, with a sha256 per downloaded file.
`load_manifest()`/`save_manifest()` are the contract the evals/splits.json generation and
the scorers build on top of."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from apx.common.types import Sha256Hex

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

Split = Literal["train", "val", "test", "eval"]


class DatasetStatus(StrEnum):
    """Whether a dataset in the manifest was actually fetched or deferred
    (loop: human-only gaps degrade gracefully rather than failing the run)."""

    PRESENT = "present"
    DEFERRED = "deferred"


class ManifestFile(BaseModel):
    """One downloaded (or extracted) file tracked in the manifest, identified by its own
    sha256 rather than the archive it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    """Repo-relative, forward-slash path, e.g. `data/fatura/FATURA.zip`."""
    sha256: Sha256Hex
    bytes: int = Field(ge=0)
    split: Split | None = None
    doc_id: str | None = None
    """Set for single-document files (e.g. one ExtractBench PDF); null for archive-level
    entries such as a dataset zip that bundles many documents."""


class DatasetEntry(BaseModel):
    """Manifest record for one dataset set (docile, fatura, extractbench, agentdojo, ...)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: DatasetStatus
    source: str
    version: str
    license: str
    blocked: str | None = None
    files: list[ManifestFile] = Field(default_factory=list)


class Manifest(BaseModel):
    """Root of data/manifest.json."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime
    datasets: dict[str, DatasetEntry] = Field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(json.loads(self.model_dump_json()), indent=2) + "\n"


def load_manifest(path: Path | None = None) -> Manifest:
    """Loads data/manifest.json, or an empty manifest if it does not exist yet."""
    manifest_path = path or MANIFEST_PATH
    if not manifest_path.exists():
        return Manifest(generated_at=datetime.now(UTC), datasets={})
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)


def save_manifest(manifest: Manifest, path: Path | None = None) -> Path:
    """Writes the manifest as pretty-printed JSON with LF line endings, creating parent dirs."""
    manifest_path = path or MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.to_json(), encoding="utf-8", newline="\n")
    return manifest_path
