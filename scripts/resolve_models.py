"""Resolves every model in infra/models.yaml against the live Hugging Face Hub API into
infra/models.lock.json: exact revision commit sha, per-file sha256/size and license, so
`make models` (scripts/download_models.py) never fetches an unverified blob (PRD §4, §10
M0.4: "never guess an id").
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import ModelInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "infra" / "models.yaml"
DEFAULT_LOCK = REPO_ROOT / "infra" / "models.lock.json"


class ModelResolutionError(RuntimeError):
    """Raised when a manifest entry cannot be verified against the Hub: unknown repo,
    unknown revision, or an expected file missing from that revision's file list."""


class HubClient(Protocol):
    """The slice of huggingface_hub.HfApi this module needs; a Protocol so tests can pass
    a fake client instead of hitting the network."""

    def model_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> ModelInfo: ...


@dataclass(frozen=True)
class ManifestEntry:
    intent: str
    role: str
    stage: str
    runtime: bool
    repo_id: str
    revision: str
    files: list[str] | None
    vram_gb_est: float | None
    notes: str


def load_manifest(path: Path) -> list[ManifestEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries: list[ManifestEntry] = []
    for item in raw["models"]:
        entries.append(
            ManifestEntry(
                intent=item["intent"],
                role=item["role"],
                stage=item["stage"],
                runtime=bool(item["runtime"]),
                repo_id=item["repo_id"],
                revision=item.get("revision") or "main",
                files=item.get("files"),
                vram_gb_est=item.get("vram_gb_est"),
                notes=item.get("notes", ""),
            )
        )
    return entries


def _license_of(info: ModelInfo) -> str | None:
    card = info.card_data
    if card is not None:
        lic = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
        if lic:
            return str(lic)
    for tag in info.tags or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _sibling_record(sib: Any) -> dict[str, Any]:
    size = getattr(sib, "size", None)
    sha256 = None
    lfs = getattr(sib, "lfs", None)
    if lfs is not None:
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if size is None:
            size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    return {"filename": sib.rfilename, "size_bytes": size, "sha256": sha256}


def resolve_entry(client: HubClient, entry: ManifestEntry) -> dict[str, Any]:
    try:
        info = client.model_info(entry.repo_id, revision=entry.revision, files_metadata=True)
    except HfHubHTTPError as exc:
        raise ModelResolutionError(
            f"{entry.intent}: cannot resolve {entry.repo_id}@{entry.revision} "
            f"against the Hub: {exc}"
        ) from exc

    available = {sib.rfilename: sib for sib in info.siblings or []}
    if entry.files is None:
        chosen = list(available.values())
    else:
        missing = [f for f in entry.files if f not in available]
        if missing:
            raise ModelResolutionError(
                f"{entry.intent}: {entry.repo_id}@{entry.revision} is missing expected "
                f"file(s) {missing}; available: {sorted(available)}"
            )
        chosen = [available[f] for f in entry.files]

    files = [_sibling_record(sib) for sib in chosen]
    total = sum(f["size_bytes"] for f in files if f["size_bytes"] is not None)

    return {
        "intent": entry.intent,
        "role": entry.role,
        "stage": entry.stage,
        "runtime": entry.runtime,
        "repo_id": entry.repo_id,
        "revision_requested": entry.revision,
        "revision_sha": info.sha,
        "license": _license_of(info),
        "vram_gb_est": entry.vram_gb_est,
        "notes": entry.notes,
        "files": files,
        "total_size_bytes": total,
    }


def resolve_all(client: HubClient, entries: list[ManifestEntry]) -> dict[str, Any]:
    resolved = [resolve_entry(client, entry) for entry in entries]
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_manifest": "infra/models.yaml",
        "models": resolved,
    }


def carry_local_models(lock: dict[str, Any], previous: Path) -> dict[str, Any]:
    """Keep the `local_models` entries of the lock being replaced (promoted local builds that
    `ml.extractor_b.promote` records, ADR-0009 §1.8): they are not on the Hub, so re-resolving
    `infra/models.yaml` must not drop them."""
    if not previous.is_file():
        return lock
    local = json.loads(previous.read_text(encoding="utf-8")).get("local_models")
    return lock if local is None else {**lock, "local_models": local}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)

    entries = load_manifest(args.manifest)
    client = HfApi()
    try:
        lock = resolve_all(client, entries)
    except ModelResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    lock = carry_local_models(lock, args.out)
    args.out.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(lock['models'])} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
