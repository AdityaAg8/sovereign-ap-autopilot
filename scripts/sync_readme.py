#!/usr/bin/env python3
"""Rewrites the README.md block between `<!-- SCOREBOARD:START -->` and `<!-- SCOREBOARD:END
-->` with the current contents of reports/scoreboard.md, so the README never drifts from the
generated scoreboard (ADR-0005)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
SCOREBOARD_PATH = REPO_ROOT / "reports" / "scoreboard.md"
START_MARKER = "<!-- SCOREBOARD:START -->"
END_MARKER = "<!-- SCOREBOARD:END -->"


class MarkerError(ValueError):
    """Raised when README.md is missing a scoreboard marker, or they are out of order -- sync
    must never silently write outside the intended block."""


def sync_readme(readme_text: str, scoreboard_text: str) -> str:
    """Returns `readme_text` with everything between the markers replaced by `scoreboard_text`,
    keeping the markers themselves and everything outside them untouched."""
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise MarkerError(f"README.md must contain {START_MARKER} before {END_MARKER}")
    before = readme_text[: start + len(START_MARKER)]
    after = readme_text[end:]
    body = scoreboard_text.strip("\n")
    return f"{before}\n{body}\n{after}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/sync_readme.py", description=__doc__)
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument("--scoreboard", type=Path, default=SCOREBOARD_PATH)
    args = parser.parse_args(argv)

    if not args.scoreboard.exists():
        print(
            f"error: {args.scoreboard} does not exist -- run `make eval-quick` first",
            file=sys.stderr,
        )
        return 1

    readme_text = args.readme.read_text(encoding="utf-8")
    scoreboard_text = args.scoreboard.read_text(encoding="utf-8")
    try:
        updated = sync_readme(readme_text, scoreboard_text)
    except MarkerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.readme.write_text(updated, encoding="utf-8", newline="\n")
    print(f"synced {args.scoreboard} into {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
