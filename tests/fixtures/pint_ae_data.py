"""Shared helper for `tests/unit/` modules that exercise the real, pinned PINT AE artifacts
under `data/pint_ae/`: resolves the data root relative to the repo root (never a hardcoded host
path) and, when a selected `artifacts`-marked test finds the bundle missing, fails it loudly
with the exact fetch command instead of a confusing file-not-found trace."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PINT_AE_DATA_ROOT = REPO_ROOT / "data" / "pint_ae"
"""The repo-relative pinned PINT AE data root. Safe to reference at module import time (pure
path arithmetic, no I/O) even when `data/` does not exist -- only `require_pint_ae_data_root()`
touches the filesystem."""

MISSING_PINT_AE_DATA = (
    "data/pint_ae missing: run `uv run python -m evals.pint_ae.fetch` first, then re-run with "
    "`uv run pytest -m artifacts` (PRD §10 M1.3/M4.3)"
)


def require_pint_ae_data_root() -> Path:
    """Returns `PINT_AE_DATA_ROOT` if the pinned bundle has already been fetched, else fails the
    calling test loudly. Call this from inside a test body or fixture -- never at module import
    time -- so collection succeeds whether or not `data/` is present."""
    if not PINT_AE_DATA_ROOT.is_dir():
        pytest.fail(MISSING_PINT_AE_DATA)
    return PINT_AE_DATA_ROOT
