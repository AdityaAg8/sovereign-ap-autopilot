"""CLI logic for `uv run python -m evals.datasets fetch [--set NAME]`: runs the DocILE,
FATURA, ExtractBench, AgentDojo, prompt-injections and CORD v2 fetchers and (re)writes
data/manifest.json, one dataset entry per set, present or deferred (PRD §5 data table, §10 M1.1;
ADR-0019 §8 for CORD's `--splits` and `--allow-large`).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import structlog

from evals.datasets import (
    agentdojo_fetcher,
    cord_fetcher,
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
    "cord": cord_fetcher.fetch,
}


def fetch_cord_splits(
    splits: list[cord_fetcher.CordSplit], *, allow_large: bool, data_root: Path = DATA_ROOT
) -> Manifest:
    """`fetch --set cord --splits ...`: fetches those CORD splits into `data_root` and merges
    them into `data_root/manifest.json`, keeping splits an earlier run fetched."""
    manifest_path = data_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    datasets = dict(manifest.datasets)
    entry = cord_fetcher.fetch_cord(splits, allow_large=allow_large, data_root=data_root)
    datasets["cord"] = cord_fetcher.merge_entry(datasets.get("cord"), entry)
    result = Manifest(generated_at=datetime.now(UTC), datasets=datasets)
    save_manifest(result, manifest_path)
    return result


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
        if name == "cord":
            entry = cord_fetcher.merge_entry(datasets.get(name), entry)
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
    fetch_parser.add_argument(
        "--splits", default=None, help="CORD only: comma-separated validation,test,train"
    )
    fetch_parser.add_argument(
        "--allow-large",
        action="store_true",
        help="CORD only: allow the >1 GB train split (owner-approved download)",
    )
    fetch_parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="CORD only: data directory whose manifest.json is updated (default: repo data/)",
    )
    args = parser.parse_args(argv)

    if args.command == "fetch" and args.splits is not None:
        if args.set_name != "cord":
            parser.error("--splits applies to --set cord only")
        wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
        bad = sorted(wanted - set(cord_fetcher.CORD_SPLITS))
        if bad or not wanted:
            parser.error(f"--splits needs CORD splits from {list(cord_fetcher.CORD_SPLITS)}")
        manifest = fetch_cord_splits(
            [s for s in cord_fetcher.CORD_SPLITS if s in wanted],
            allow_large=args.allow_large,
            data_root=args.data_root,
        )
        print(f"cord: {manifest.datasets['cord'].version}")
        return 0
    if args.command == "fetch":
        names = [args.set_name] if args.set_name else None
        manifest = fetch_all(names)
        for name, entry in manifest.datasets.items():
            print(f"{name}: {entry.status.value}")
        return 0
    return 1
