"""Fixed evaluation/training splits keyed by document sha256, written to the committed
evals/splits.json (PRD §5, §6.2 I5, ADR-0005). Builds the fatura/extractbench/docile groups
deterministically from data/manifest.json, and exposes `register_documents` so a later
milestone (e.g. M2's UAE factory) can add newly generated documents to `uae`/`attacks`
without ever reshuffling a sha256 that was already assigned.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from apx.common.types import Sha256Hex
from evals.datasets.manifest import Manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = REPO_ROOT / "evals" / "splits.json"

SetName = Literal["uae", "docile", "fatura", "extractbench", "attacks"]

# The split names each set is allowed to use, in the fixed order evals/splits.json writes
# them (so regenerating the file never reorders unrelated keys).
SET_SPLIT_NAMES: dict[SetName, tuple[str, ...]] = {
    "uae": ("train", "cal", "test"),
    "docile": ("train", "val"),
    "fatura": ("train", "test"),
    "extractbench": ("eval",),
    "attacks": ("eval",),
}

# Which (set, split) pairs pool into each leakage-check bucket (PRD §6.2 I5: eval, cal and
# test data must never reach training). Every named split feeds exactly one pool.
POOL_MEMBERS: dict[str, tuple[tuple[SetName, str], ...]] = {
    "train": (("uae", "train"), ("docile", "train"), ("fatura", "train")),
    "cal": (("uae", "cal"),),
    "test": (("uae", "test"), ("fatura", "test")),
    "docile_val": (("docile", "val"),),
    "extractbench_eval": (("extractbench", "eval"),),
    "attacks_eval": (("attacks", "eval"),),
}

SplitGroups = dict[str, list[Sha256Hex]]


class SplitsFile(BaseModel):
    """Root of evals/splits.json: one group of named sha256 lists per set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime
    sets: dict[SetName, SplitGroups] = Field(default_factory=dict)


def empty_splits(*, now: datetime | None = None) -> SplitsFile:
    """A SplitsFile with every known set/split present but empty (the M1 starting point for
    `uae`, `docile` and `attacks`, which are populated by later milestones)."""
    return SplitsFile(
        generated_at=now or datetime.now(UTC),
        sets={name: {split: [] for split in splits} for name, splits in SET_SPLIT_NAMES.items()},
    )


def load_splits(path: Path | None = None) -> SplitsFile:
    """Loads evals/splits.json, or an empty (but fully-shaped) SplitsFile if it does not exist
    yet. Does not sort or de-duplicate: callers that need clean data should validate first."""
    splits_path = path or SPLITS_PATH
    if not splits_path.exists():
        return empty_splits()
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    return SplitsFile.model_validate(data)


def save_splits(splits: SplitsFile, path: Path | None = None) -> Path:
    """Writes evals/splits.json with LF endings and a fixed key order (SET_SPLIT_NAMES) and
    sorted, de-duplicated sha256 lists per split, so regenerating it produces a clean diff."""
    splits_path = path or SPLITS_PATH
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": splits.generated_at.isoformat(),
        "sets": {
            name: {
                split: sorted(set(splits.sets.get(name, {}).get(split, [])))
                for split in SET_SPLIT_NAMES[name]
            }
            for name in SET_SPLIT_NAMES
        },
    }
    splits_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return splits_path


def _fatura_groups(manifest: Manifest) -> SplitGroups:
    groups: SplitGroups = {split: [] for split in SET_SPLIT_NAMES["fatura"]}
    entry = manifest.datasets.get("fatura")
    if entry is None:
        return groups
    for file in entry.files:
        if file.doc_id is None or file.split not in groups:
            continue  # archive-level entries (e.g. the zip itself) are not documents
        groups[file.split].append(file.sha256)
    return groups


def _extractbench_groups(manifest: Manifest) -> SplitGroups:
    groups: SplitGroups = {split: [] for split in SET_SPLIT_NAMES["extractbench"]}
    entry = manifest.datasets.get("extractbench")
    if entry is None:
        return groups
    for file in entry.files:
        if file.doc_id is None or file.split not in groups:
            continue
        groups[file.split].append(file.sha256)
    return groups


def _docile_groups(manifest: Manifest) -> SplitGroups:
    """DocILE is deferred until M1.1's fetch has a DOCILE_TOKEN and is extracted per-document;
    until then this always returns empty train/val lists."""
    groups: SplitGroups = {split: [] for split in SET_SPLIT_NAMES["docile"]}
    entry = manifest.datasets.get("docile")
    if entry is None:
        return groups
    for file in entry.files:
        if file.doc_id is None or file.split not in groups:
            continue
        groups[file.split].append(file.sha256)
    return groups


def build_from_manifest(manifest: Manifest, existing: SplitsFile | None = None) -> SplitsFile:
    """Recomputes fatura/extractbench/docile fully from data/manifest.json (a pure function of
    each document's own sha256 and split, so re-running reproduces the same output), while
    leaving `uae` and `attacks` exactly as `existing` had them: those two are populated later
    via `register_documents`, never re-derived from the manifest."""
    base = existing or empty_splits()
    sets: dict[SetName, SplitGroups] = {
        "uae": copy.deepcopy(base.sets.get("uae", {s: [] for s in SET_SPLIT_NAMES["uae"]})),
        "attacks": copy.deepcopy(
            base.sets.get("attacks", {s: [] for s in SET_SPLIT_NAMES["attacks"]})
        ),
        "docile": _docile_groups(manifest),
        "fatura": _fatura_groups(manifest),
        "extractbench": _extractbench_groups(manifest),
    }
    return SplitsFile(generated_at=datetime.now(UTC), sets=sets)


def register_documents(
    splits: SplitsFile, set_name: SetName, assignments: Mapping[str, str]
) -> SplitsFile:
    """Adds `{sha256: split}` assignments into `set_name` (e.g. once the M2 UAE factory
    produces documents). A sha256 already present anywhere in the set, in any split, is left
    untouched: registering only adds new documents, it never reshuffles an existing one
    (PRD ADR-0005)."""
    valid_splits = SET_SPLIT_NAMES[set_name]
    current: SplitGroups = copy.deepcopy(
        splits.sets.get(set_name, {split: [] for split in valid_splits})
    )
    for split in valid_splits:
        current.setdefault(split, [])
    already_assigned = {sha for members in current.values() for sha in members}

    for sha256, split in assignments.items():
        if split not in valid_splits:
            raise ValueError(f"{set_name!r} has no {split!r} split; expected one of {valid_splits}")
        if sha256 in already_assigned:
            continue
        current[split].append(sha256)
        already_assigned.add(sha256)

    new_sets = copy.deepcopy(splits.sets)
    new_sets[set_name] = {split: sorted(set(members)) for split, members in current.items()}
    return SplitsFile(generated_at=datetime.now(UTC), sets=new_sets)


def _pool(splits: SplitsFile, members: tuple[tuple[SetName, str], ...]) -> set[str]:
    result: set[str] = set()
    for set_name, split_name in members:
        result.update(splits.sets.get(set_name, {}).get(split_name, []))
    return result


def find_duplicate_documents(splits: SplitsFile) -> list[str]:
    """Sha256s listed more than once within a single (set, split) list."""
    problems: list[str] = []
    for set_name, groups in splits.sets.items():
        for split_name, members in groups.items():
            seen: set[str] = set()
            for sha in members:
                if sha in seen:
                    problems.append(f"{set_name}.{split_name} lists {sha} more than once")
                seen.add(sha)
    return problems


def find_leaks(splits: SplitsFile) -> list[str]:
    """Cross-pool sha256 overlaps: any document reachable from more than one of
    train/cal/test/docile_val/extractbench_eval/attacks_eval would mean eval, cal or test data
    reaching training (or cal/test mixing with each other) (PRD §6.2 I5). Empty means clean."""
    pools = {name: _pool(splits, members) for name, members in POOL_MEMBERS.items()}
    names = sorted(pools)
    problems: list[str] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for sha in sorted(pools[name_a] & pools[name_b]):
                problems.append(f"{sha} is in both {name_a!r} and {name_b!r}")
    return problems


def validate_splits(splits: SplitsFile) -> list[str]:
    """All problems `tests/test_no_leakage.py` treats as a leakage failure: within-split
    duplicates plus cross-pool overlaps."""
    return find_duplicate_documents(splits) + find_leaks(splits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.splits")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "build", help="Rebuild evals/splits.json's fatura/extractbench/docile groups"
    )
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest = load_manifest()
        existing = load_splits()
        updated = build_from_manifest(manifest, existing)
        problems = validate_splits(updated)
        if problems:
            for problem in problems:
                print(f"leakage: {problem}")
            return 1
        save_splits(updated)
        print(f"wrote {SPLITS_PATH}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
