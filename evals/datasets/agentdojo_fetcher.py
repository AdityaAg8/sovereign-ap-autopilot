"""Checks whether the AgentDojo banking benchmark package is installed. `agentdojo` is a
declared dev dependency (pyproject.toml's dev group), so this normally succeeds once `uv sync`
has run; this fetcher only reports availability and defers with a BLOCKED.md entry on the rare
environment where it is still missing, rather than installing anything.

"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from evals.datasets.common import append_blocked_entry
from evals.datasets.manifest import DatasetEntry, DatasetStatus

DATASET_NAME = "agentdojo"
SOURCE_URL = "https://pypi.org/project/agentdojo/"
LICENSE = "MIT (agentdojo)"
FETCH_COMMAND = "uv add agentdojo && uv run python -m evals.datasets fetch --set agentdojo"


def fetch(data_root: Path) -> DatasetEntry:
    del data_root  # AgentDojo is a pip package, not a data directory.
    try:
        version = importlib.metadata.version("agentdojo")
    except importlib.metadata.PackageNotFoundError:
        append_blocked_entry(
            tag="dataset:agentdojo",
            title="AgentDojo banking suite needs `agentdojo` added to pyproject.toml",
            why=(
                "The eval plan lists AgentDojo banking via `pip install agentdojo`; the package is "
                "declared in pyproject.toml's dev group but is not importable in this "
                "environment (run `uv sync --all-groups`), so the agent security eval set "
                "cannot be loaded yet."
            ),
            command=FETCH_COMMAND,
        )
        return DatasetEntry(
            status=DatasetStatus.DEFERRED,
            source=SOURCE_URL,
            version="unknown",
            license=LICENSE,
            blocked="agentdojo not installed; see BLOCKED.md [dataset:agentdojo]",
            files=[],
        )
    return DatasetEntry(
        status=DatasetStatus.PRESENT,
        source=SOURCE_URL,
        version=version,
        license=LICENSE,
        blocked=None,
        files=[],
    )
