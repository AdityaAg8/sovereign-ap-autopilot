"""Entry point for `uv run python -m evals.datasets fetch [--set NAME]` (PRD §10 M1.1)."""

from __future__ import annotations

import sys

from evals.datasets.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
