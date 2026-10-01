"""Moves eval row files written under another parser build to logs/archive/ before a GPU queue
(ADR-0018 §5), so a resume never refuses them mid-queue and the trust job never mixes parses.
Files are moved with their `.parse_failures.jsonl` sibling, never deleted or rewritten.

Usage: python -m evals.archive_stale_rows [--dry-run] [--archive-dir logs/archive] ROWS.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from evals.parse_cache import parser_identity

PARSER_KEY: Final[str] = "parser_identity"
NO_IDENTITY: Final[str] = "none"
"""Label of rows that carry no parser identity (written before rows were stamped): stale."""


@dataclass(frozen=True)
class Verdict:
    path: Path
    identities: frozenset[str]
    """Parser identities the file's rows carry (NO_IDENTITY for unstamped rows)."""
    moved_to: Path | None = None

    @property
    def stale(self) -> bool:
        return self.moved_to is not None


def row_identities(path: Path) -> frozenset[str]:
    """Every parser identity in a JSONL row file. An unparseable line raises ValueError: a file
    we cannot read is neither kept as current nor moved blindly."""
    found: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not JSON: {exc}") from exc
        value = row.get(PARSER_KEY) if isinstance(row, dict) else None
        found.add(str(value) if value else NO_IDENTITY)
    return frozenset(found)


def _free_destination(archive_dir: Path, name: str) -> Path:
    candidate = archive_dir / name
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    while candidate.exists():
        candidate = archive_dir / f"{stem}.{n}{suffix}"
        n += 1
    return candidate


def archive_stale(
    paths: Sequence[Path], current: str, archive_dir: Path, *, dry_run: bool = False
) -> list[Verdict]:
    """Keep files whose every row has parser identity `current`; move the others (whole file,
    plus `<stem>.parse_failures.jsonl`) to `archive_dir/<stem>.<old identity>.jsonl`. Absent
    files are skipped. An empty file has no stale rows and stays."""
    verdicts: list[Verdict] = []
    for path in paths:
        if not path.is_file():
            continue
        identities = row_identities(path)
        stale = identities - {current}
        if not stale:
            verdicts.append(Verdict(path, identities))
            continue
        label = next(iter(stale)) if len(stale) == 1 else "mixed"
        destination = _free_destination(archive_dir, f"{path.stem}.{label}{path.suffix}")
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            failures = path.with_name(path.stem + ".parse_failures.jsonl")
            if failures.is_file():
                failures.rename(
                    _free_destination(archive_dir, f"{failures.stem}.{label}{failures.suffix}")
                )
            path.rename(destination)
        verdicts.append(Verdict(path, identities, destination))
    return verdicts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.archive_stale_rows", description=__doc__)
    parser.add_argument("rows", nargs="+", type=Path)
    parser.add_argument("--archive-dir", type=Path, default=Path("logs/archive"))
    parser.add_argument("--parser-identity", default="", help="default: the current parser build")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    current = args.parser_identity or parser_identity()
    try:
        verdicts = archive_stale(args.rows, current, args.archive_dir, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"archive-stale-rows FAILED: {exc}", file=sys.stderr)
        return 2
    verb = "would move" if args.dry_run else "moved"
    print(f"current parser identity {current}")
    for v in verdicts:
        seen = ", ".join(sorted(v.identities)) or "no rows"
        if v.moved_to is None:
            print(f"  keep   {v.path}  ({seen})")
        else:
            print(f"  {verb:<6} {v.path} -> {v.moved_to}  ({seen})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
