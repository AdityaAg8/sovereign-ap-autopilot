"""CLI logic for `uv run python -m evals.datasets fetch [--set NAME]`: runs the DocILE,
FATURA, ExtractBench, AgentDojo and prompt-injections fetchers and (re)writes
data/manifest.json, one dataset entry per set, present or deferred (PRD §5 data table, §10 M1.1).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import structlog

from evals.datasets import (
    agentdojo_fetcher,
    docile_fetcher,
    extractbench_fetcher,
    fatura_fetcher,
    prompt_injections_fetcher,
)
from evals.datasets.manifest import DatasetEntry, Manifest, load_manifest, save_manifest

log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

Fetcher = Callable[[Path], DatasetEntry]

FETCHERS: dict[str, Fetcher] = {
    "docile": docile_fetcher.fetch,
    "fatura": fatura_fetcher.fetch,
    "extractbench": extractbench_fetcher.fetch,
    "agentdojo": agentdojo_fetcher.fetch,
    "prompt_injections": prompt_injections_fetcher.fetch,
}


def fetch_all(names: list[str] | None = None, *, data_root: Path = DATA_ROOT) -> Manifest:
    """Runs the selected fetchers (all by default), merges results into the existing
    manifest, and writes data/manifest.json. Never raises for a deferred dataset: only an
    unexpected fetcher exception propagates."""
    selected = names or list(FETCHERS)
    unknown = sorted(set(selected) - set(FETCHERS))
    if unknown:
        raise ValueError(f"Unknown dataset name(s): {unknown}; choices: {sorted(FETCHERS)}")

    manifest = load_manifest()
    datasets = dict(manifest.datasets)
    for name in selected:
        log.info("dataset.fetch.start", dataset=name)
        entry = FETCHERS[name](data_root)
        datasets[name] = entry
        log.info("dataset.fetch.done", dataset=name, status=entry.status.value)

    result = Manifest(generated_at=datetime.now(UTC), datasets=datasets)
    save_manifest(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch eval datasets, writing data/manifest.json"
    )
    fetch_parser.add_argument(
        "--set",
        dest="set_name",
        choices=sorted(FETCHERS),
        default=None,
        help="Fetch only this dataset (default: all)",
    )
    args = parser.parse_args(argv)

    if args.command == "fetch":
        names = [args.set_name] if args.set_name else None
        manifest = fetch_all(names)
        for name, entry in manifest.datasets.items():
            print(f"{name}: {entry.status.value}")
        return 0
    return 1
