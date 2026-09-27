"""M3.4 bake-off runner (PRD §10 M3.4): parses every bake-off document once into the parse cache,
then runs each probed, in-budget arm strictly sequentially over the fixed 200-document UAE subset
(a scanned document as its image-only scan PDF, as the runtime receives it) and ExtractBench
short, appending identity-stamped rows (resumable) and one whole-card VRAM
session per invocation under logs/bakeoff/. A challenger stops early only under the pre-registered
futility rule (`evals.bakeoff.sequential`); the baseline may import A test rows of the same
identity. `evals.bakeoff.report` turns the rows into the report.

Usage: python -m evals.bakeoff.run [--arms a,q38-27b,...] [--data-dir data/uae/test] [--limit N]
       [--reuse-a-rows logs/extraction_a.jsonl] [--request-timeout-s 300]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TextIO

from apx.common.errors import ModelServingError
from apx.extract.extractor import LlmExtractor
from apx.extract.interfaces import ExtractorId
from apx.llm.interfaces import LlmClient
from apx.parse.interfaces import ParsedDocument
from evals.bakeoff.arms import BASELINE, Arm, ArmKind, BakeoffError, invoice_prompt_builder
from evals.bakeoff.extractbench import (
    REQUEST_TIMEOUT_S,
    ExtractBenchPromptBuilder,
    extractbench_row,
    is_request_rejection,
)
from evals.bakeoff.parsed import ParseCache
from evals.bakeoff.probe import ProbeResult
from evals.bakeoff.sequential import LookResult, futility, look_sizes
from evals.extraction_a import extract_row, is_current_row, load_rows
from evals.extraction_identity import (
    IDENTITY_KEYS,
    RowIdentity,
    prompt_version,
    require_identity,
    served_gguf_sha256,
)
from evals.scan_input import document_input, input_kind, load_truth
from ml.extractor_b.gates import StrictVramSampler, VramUnavailableError

ROWS_DIR: Final[Path] = Path("logs/bakeoff")
PARSE_CACHE_DIR: Final[Path] = Path("data/bakeoff/parsed")

MAX_CONSECUTIVE_SERVER_FAILURES: Final[int] = 3
"""Timeout/5xx rows in a row (each followed by an unload) before the run ends: past this the
server, not a document, is the likely cause, so the run stops for a human to look."""

SampleFn = Callable[[], float | None]
UnloadFn = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ArmFiles:
    uae: Path
    extractbench: Path
    vram: Path


def arm_files(rows_dir: Path, arm: Arm) -> ArmFiles:
    return ArmFiles(
        uae=rows_dir / f"{arm.key}.uae.jsonl",
        extractbench=rows_dir / f"{arm.key}.extractbench.jsonl",
        vram=rows_dir / f"{arm.key}.vram.jsonl",
    )


@dataclass(frozen=True, slots=True)
class ArmIdentities:
    uae: RowIdentity
    extractbench: RowIdentity


def arm_identities(
    arm: Arm, *, llama_swap_config: Path, models_lock: Path, models_root: Path
) -> ArmIdentities:
    """Alias + served GGUF sha256 + prompt version, per row file (the two prompts differ)."""
    gguf = served_gguf_sha256(
        arm.alias,
        llama_swap_config=llama_swap_config,
        models_lock=models_lock,
        models_root=models_root,
    )
    return ArmIdentities(
        uae=RowIdentity(
            model=arm.alias,
            gguf_sha256=gguf,
            prompt_version=prompt_version(invoice_prompt_builder(arm)),
        ),
        extractbench=RowIdentity(
            model=arm.alias,
            gguf_sha256=gguf,
            prompt_version=ExtractBenchPromptBuilder(images=arm.images).version,
        ),
    )


def require_probe_ok(probe: Mapping[str, ProbeResult], arm: Arm) -> None:
    """No eval hours for an arm that was not probed or did not fit the budget (PRD M3.4)."""
    result = probe.get(arm.key)
    if result is None:
        raise BakeoffError(f"{arm.key}: no VRAM probe on record; run `make bakeoff-probe` first")
    if result.status != "ok":
        raise BakeoffError(f"{arm.key}: disqualified by the VRAM probe ({result.reason})")


def load_extractbench_rows(path: Path) -> dict[str, dict[str, Any]]:
    """ExtractBench rows keyed by ExtractBench doc id; the last row wins."""
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["doc_id"]: row for row in rows}


def load_vram_sessions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def uae_row(
    sha: str, document: ParsedDocument, data_dir: Path, extractor: LlmExtractor
) -> dict[str, Any]:
    """`evals.extraction_a.extract_row`, except that a request llama-server rejects for this
    document (context overflow, HTTP 400) is an invalid row with its error, not a dead run."""
    truth = load_truth(data_dir, sha)
    kind = input_kind(truth)
    try:
        return {**await extract_row(sha, document, data_dir, extractor, kind), "error": None}
    except ModelServingError as exc:
        if not is_request_rejection(exc):
            raise
        return {
            "sha": sha,
            "input_kind": kind,
            "arabic_first": bool(truth["arabic_first"]),
            "valid_json": False,
            "latency_ms": None,
            "pages": len(document.pages),
            "arabic": "ar" in {str(language) for language in document.languages},
            "fields": {},
            "error": str(exc),
        }


def uae_inputs(
    shas: Sequence[str], data_dir: Path, scan_dir: Path
) -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """The parse input of each bake-off UAE document in the form the runtime receives it
    (`evals.scan_input.document_input`: a has_scan document is its image-only scan PDF, written
    under `scan_dir`), keyed by the sha256 of those bytes as the runtime keys an upload; also
    the factory sha -> cache key map."""
    items: list[tuple[str, Path]] = []
    keys: dict[str, str] = {}
    for sha in shas:
        kind, data = document_input(data_dir, sha, load_truth(data_dir, sha))
        key = hashlib.sha256(data).hexdigest()
        path = data_dir / f"{sha}.pdf"
        if kind == "scan":
            path = scan_dir / f"{key}.pdf"
            if not path.is_file():
                scan_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        items.append((key, path))
        keys[sha] = key
    return items, keys


def reuse_a_rows(
    source: Path,
    target: Path,
    identity: RowIdentity,
    shas: Sequence[str],
    log: Callable[[str], None] = print,
) -> int:
    """Append the A test run's rows (`evals.extraction_a` over the same factory documents) for
    the bake-off subset to the baseline's UAE row file, so arm a is not re-run (ADR-0016 amdt 1).
    All or nothing: every row in `source` must carry exactly the baseline's identity (alias,
    served GGUF sha256, prompt version and whatever else `IDENTITY_KEYS` covers) and the current
    row schema; otherwise nothing is imported and arm a runs. Returns how many rows were added."""
    rows = load_rows(source)
    if not rows:
        log(f"a: no A test rows at {source}; arm a runs")
        return 0
    want = identity.as_row_fields()
    foreign = [sha for sha, row in rows.items() if {k: row.get(k) for k in IDENTITY_KEYS} != want]
    stale = [sha for sha, row in rows.items() if not is_current_row(row)]
    if foreign or stale:
        first = rows[(foreign or stale)[0]]
        found = {k: first.get(k) for k in IDENTITY_KEYS}
        log(
            f"a: not reusing {source}: {len(foreign)} rows of another identity, {len(stale)} of an "
            f"old row schema (e.g. {found}, expected {want}); arm a runs"
        )
        return 0
    present = load_rows(target)
    require_identity(present, identity, target)
    new = [sha for sha in shas if sha in rows and sha not in present]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as sink:
        for sha in new:
            _write(sink, {**rows[sha], "error": None, "reused_from": source.as_posix()})
    log(f"a: reused {len(new)} A test rows from {source} ({len(present)} already present)")
    return len(new)


def futility_stop(
    arm: Arm,
    arm_rows: Mapping[str, Mapping[str, Any]],
    baseline_rows: Mapping[str, Mapping[str, Any]] | None,
    data_dir: Path,
    shas: Sequence[str],
) -> LookResult | None:
    """The look at which the pre-registered futility rule stops a challenger, if any. The
    baseline and the latency arm never stop; nor does anything without baseline rows."""
    if arm.kind is not ArmKind.CHALLENGER or baseline_rows is None:
        return None
    return futility(arm.key, arm_rows, baseline_rows, data_dir, shas).stopped


def _write(sink: TextIO, row: Mapping[str, Any]) -> None:
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    sink.flush()


async def run_arm(
    arm: Arm,
    *,
    uae_shas: Sequence[str],
    uae_data_dir: Path,
    extractbench_records: Mapping[str, Mapping[str, Any]],
    extractbench_shas: Mapping[str, str],
    files: ArmFiles,
    identities: ArmIdentities,
    client: LlmClient,
    unload: UnloadFn,
    sample_fn: SampleFn,
    cache: ParseCache,
    uae_cache_keys: Mapping[str, str] | None = None,
    baseline_rows: Mapping[str, Mapping[str, Any]] | None = None,
    request_timeout_s: float = REQUEST_TIMEOUT_S,
    log: Callable[[str], None] = print,
    interval_s: float = 0.2,
) -> LookResult | None:
    """Every pending UAE then ExtractBench document of `arm`, one at a time, inside one
    whole-card VRAM window that starts from an empty llama-swap (so it includes the load).
    `uae_cache_keys` maps a factory sha to its parse-cache key (`uae_inputs`; default: itself).
    With `baseline_rows`, a challenger is tested at each interim look and stops (no further UAE
    or ExtractBench documents) when the futility rule says it can no longer win; the stopping
    look is returned. Rows are written in subset order, so a resume re-checks the same looks."""
    uae_rows = load_rows(files.uae)
    require_identity(uae_rows, identities.uae, files.uae)
    eb_rows = load_extractbench_rows(files.extractbench)
    require_identity(eb_rows, identities.extractbench, files.extractbench)
    stopped = futility_stop(arm, uae_rows, baseline_rows, uae_data_dir, uae_shas)
    if stopped is not None:
        log(f"{arm.key}: stopped for futility at look {stopped.look} (n={stopped.n}) earlier")
        return stopped
    uae_pending = [sha for sha in uae_shas if sha not in uae_rows]
    eb_ids = list(extractbench_records)
    eb_pending = [doc_id for doc_id in eb_ids if doc_id not in eb_rows]
    if not uae_pending and not eb_pending:
        log(f"{arm.key}: all {len(uae_shas)} UAE and {len(eb_ids)} ExtractBench rows present")
        return None
    files.uae.parent.mkdir(parents=True, exist_ok=True)
    await unload()
    idle = sample_fn()
    extractor = LlmExtractor(
        client, invoice_prompt_builder(arm), model=arm.alias, extractor=ExtractorId.A
    )
    eb_builder = ExtractBenchPromptBuilder(images=arm.images)
    started = datetime.now(UTC).isoformat()
    uae_done, eb_done = len(uae_shas) - len(uae_pending), len(eb_ids) - len(eb_pending)
    looks = set(look_sizes(len(uae_shas)))
    sampler = StrictVramSampler(sample_fn, interval_s=interval_s)
    try:
        with sampler:
            with files.uae.open("a", encoding="utf-8") as sink:
                for sha in uae_pending:
                    document = cache.get((uae_cache_keys or {}).get(sha, sha))
                    row = await uae_row(sha, document, uae_data_dir, extractor)
                    _write(sink, {**row, **identities.uae.as_row_fields()})
                    uae_rows[sha] = row
                    uae_done += 1
                    log(
                        f"[{uae_done}/{len(uae_shas)}] {arm.key} uae {sha[:12]} "
                        f"valid={row['valid_json']} {row['latency_ms'] or 0:.0f} ms"
                    )
                    if uae_done not in looks:
                        continue
                    stopped = futility_stop(arm, uae_rows, baseline_rows, uae_data_dir, uae_shas)
                    if stopped is not None:
                        log(
                            f"{arm.key}: stopped for futility at look {stopped.look} "
                            f"(n={stopped.n}): {'; '.join(stopped.reasons)}"
                        )
                        return stopped
            failures: list[str] = []
            with files.extractbench.open("a", encoding="utf-8") as sink:
                for doc_id in eb_pending:
                    document = cache.get(extractbench_shas[doc_id])
                    row = await extractbench_row(
                        extractbench_records[doc_id],
                        document,
                        client,
                        eb_builder,
                        arm.alias,
                        timeout_s=request_timeout_s,
                    )
                    _write(sink, {**row, **identities.extractbench.as_row_fields()})
                    eb_done += 1
                    kind = row.get("error_kind")
                    log(
                        f"[{eb_done}/{len(eb_ids)}] {arm.key} extractbench {doc_id} "
                        f"valid={row['valid_json']} {row['latency_ms'] or 0:.0f} ms"
                        + (f" {kind}" if kind else "")
                    )
                    if kind not in ("timeout", "server_error"):
                        failures.clear()
                        continue
                    failures.append(doc_id)
                    await unload()  # a stalled or crashed slot must not delay the next document
                    if len(failures) >= MAX_CONSECUTIVE_SERVER_FAILURES:
                        raise BakeoffError(
                            f"{arm.key}: {len(failures)} ExtractBench requests in a row timed "
                            f"out or got HTTP 5xx ({', '.join(failures)}); check llama-swap, "
                            f"then delete those rows from {files.extractbench} if the server "
                            "was at fault, and resume"
                        )
    finally:
        await unload()  # arm done or interrupted: free the card for the next arm
        try:
            peak: float | None = sampler.peak_gb
        except VramUnavailableError:
            peak = None  # recorded as a gap; the report shows the arm's peak as unmeasured
        session = {
            "started_at": started,
            "idle_gb": idle,
            "peak_gb": peak,
            "samples": sampler.samples,
            "misses": sampler.misses,
            "uae_done": uae_done,
            "extractbench_done": eb_done,
            "stopped_at_look": None if stopped is None else stopped.look,
        }
        with files.vram.open("a", encoding="utf-8") as sink:
            _write(sink, session)
    return None


def extractbench_inputs(
    data_root: Path, limit: int | None
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Path]]:
    """ExtractBench short eval documents in doc-id order: records, sha256s and PDF paths, from
    data/manifest.json and evals/splits.json (the same selection `evals.run` scores)."""
    from evals.datasets.manifest import load_manifest
    from evals.run import _extractbench_doc_ids, _load_extractbench_records
    from evals.splits import load_splits

    manifest = load_manifest(data_root / "manifest.json")
    doc_ids = _extractbench_doc_ids(manifest, load_splits(), limit)
    if not doc_ids:
        raise BakeoffError("no ExtractBench short documents in data/manifest.json + splits.json")
    records = _load_extractbench_records(data_root)
    entry = manifest.datasets["extractbench"]
    shas = {f.doc_id: f.sha256 for f in entry.files if f.doc_id}
    missing = [doc_id for doc_id in doc_ids if doc_id not in records or doc_id not in shas]
    if missing:
        raise BakeoffError(
            f"{len(missing)} ExtractBench documents lack a record, e.g. {missing[0]}"
        )
    ordered = {doc_id: records[doc_id] for doc_id in doc_ids}
    paths = {doc_id: data_root / "extractbench" / str(records[doc_id]["pdf"]) for doc_id in doc_ids}
    return ordered, {doc_id: shas[doc_id] for doc_id in doc_ids}, paths


async def _live(args: argparse.Namespace) -> int:  # pragma: no cover - needs GPU + llama-swap
    from scripts.smoke import sample_vram_gb

    from apx.llm.client import LlamaSwapClient, LlmSettings
    from apx.parse.client import ParseServiceClient
    from apx.parse.parser import PaddleOcrDocumentParser
    from evals.bakeoff.probe import PROBE_PATH, load_probe, parse_arms
    from evals.bakeoff.subset import uae_subset
    from evals.splits import load_splits

    uae_shas = uae_subset(load_splits().sets["uae"]["test"])
    if args.limit:
        uae_shas = uae_shas[: args.limit]
    missing = [sha for sha in uae_shas if not (args.data_dir / f"{sha}.pdf").exists()]
    if missing:
        raise BakeoffError(f"{len(missing)} bake-off UAE PDFs missing in {args.data_dir}")
    records, eb_shas, eb_paths = extractbench_inputs(args.data_root, args.limit or None)

    cache = ParseCache(args.parse_cache)
    parse_client = ParseServiceClient()
    parser = PaddleOcrDocumentParser(parse_client)
    items, uae_keys = uae_inputs(uae_shas, args.data_dir, args.parse_cache / "scan_inputs")
    items += [(eb_shas[doc_id], eb_paths[doc_id]) for doc_id in records]
    if await cache.fill(items, parser.parse):
        await parse_client.unload()  # parse stage over before any arm's VRAM window opens

    probe = load_probe(PROBE_PATH)
    # The baseline first: challengers are tested against its rows at every interim look.
    arms = sorted(parse_arms(args.arms), key=lambda arm: arm.kind is not ArmKind.BASELINE)
    client = LlamaSwapClient(
        LlmSettings(base_url=args.base_url, allowed_models=frozenset(a.alias for a in arms))
    )

    def identities(arm: Arm) -> ArmIdentities:
        return arm_identities(
            arm,
            llama_swap_config=args.llama_swap_config,
            models_lock=args.models_lock,
            models_root=args.models_root,
        )

    def baseline_rows() -> dict[str, dict[str, Any]] | None:
        path = arm_files(args.rows_dir, BASELINE).uae
        rows = load_rows(path)
        if not rows:
            return None
        require_identity(rows, identities(BASELINE).uae, path)
        return rows

    skipped = 0
    try:
        for arm in arms:
            try:
                require_probe_ok(probe, arm)
            except BakeoffError as exc:
                print(f"skip {arm.key}: {exc}", flush=True)
                skipped += 1
                continue
            files = arm_files(args.rows_dir, arm)
            if arm.kind is ArmKind.BASELINE and args.reuse_a_rows is not None:
                reuse_a_rows(args.reuse_a_rows, files.uae, identities(arm).uae, uae_shas)
            await run_arm(
                arm,
                uae_shas=uae_shas,
                uae_data_dir=args.data_dir,
                uae_cache_keys=uae_keys,
                extractbench_records=records,
                extractbench_shas=eb_shas,
                files=files,
                identities=identities(arm),
                client=client,
                unload=client.unload,
                sample_fn=sample_vram_gb,
                cache=cache,
                baseline_rows=baseline_rows(),
                request_timeout_s=args.request_timeout_s,
            )
    finally:
        await client.aclose()
    return 0 if skipped < len(arms) else 1


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(prog="python -m evals.bakeoff.run", description=__doc__)
    parser.add_argument("--arms", default="", help="comma-separated arm keys (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae/test"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=0, help="smoke only: first N per set")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--rows-dir", type=Path, default=ROWS_DIR)
    parser.add_argument("--parse-cache", type=Path, default=PARSE_CACHE_DIR)
    parser.add_argument("--llama-swap-config", type=Path, default=Path("infra/llama-swap.yaml"))
    parser.add_argument("--models-lock", type=Path, default=Path("infra/models.lock.json"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--reuse-a-rows",
        type=Path,
        default=None,
        help="A test rows (evals.extraction_a, run alone at --concurrency 1) to import as arm a "
        "when their identity matches exactly (ADR-0016 amendment 1)",
    )
    parser.add_argument("--request-timeout-s", type=float, default=REQUEST_TIMEOUT_S)
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_live(args))
    except BakeoffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
