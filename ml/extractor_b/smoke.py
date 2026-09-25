"""llama-server smoke for an exported Extractor B (ADR-0009 §1.7 d): 5 UAE dev documents go
through llama-swap's non-default `extractor-b-challenger` entry with the serving prompt and
schema-constrained decoding, and each must return a JSON object. Host-side (needs `make up`);
writes `export/smoke.json` next to the verified export. The served `extractor-b` is untouched."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from apx.common.errors import ApxError
from apx.extract.interfaces import extraction_json_schema
from apx.extract.prompt_b import b_messages
from apx.llm.client import LlamaSwapClient, LlmSettings
from apx.llm.interfaces import LlmClient, StructuredRequest
from ml.extractor_b.checkpoints import atomic_write_json
from ml.extractor_b.config import DEFAULT_CONFIG_PATH, load_config
from ml.extractor_b.evaluator import parse_prediction
from ml.extractor_b.export import Q4_NAME, ExportError, latest_run_dir, verify_export
from ml.extractor_b.interfaces import SampleSource, TrainingSample

CHALLENGER_MODEL: Final[str] = "extractor-b-challenger"
"""The non-default llama-swap entry serving the exported GGUF (infra/llama-swap.yaml)."""
SMOKE_DOCS: Final[int] = 5
SMOKE_NAME: Final[str] = "smoke.json"


class SmokeError(ExportError):
    """The smoke could not run (no export, too few UAE dev rows) or a document was not JSON."""


class SmokeDoc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample_id: str
    pages: int
    valid_json: bool
    finish_reason: str
    latency_ms: float


class SmokeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str
    run_id: str
    step: int
    q4_sha256: str
    documents: int
    valid: int
    passed: bool
    docs: tuple[SmokeDoc, ...]
    created_at: datetime


def pick_rows(rows: Sequence[TrainingSample], n: int = SMOKE_DOCS) -> list[TrainingSample]:
    """The first `n` UAE dev rows by sample_id: a fixed, reproducible subset."""
    uae = sorted((r for r in rows if r.source is SampleSource.UAE), key=lambda r: r.sample_id)
    if len(uae) < n:
        raise SmokeError(f"need {n} UAE dev rows for the smoke, found {len(uae)}")
    return uae[:n]


def smoke_request(row: TrainingSample, data_dir: Path, model: str) -> StructuredRequest:
    """The serving request for B (image-only prompt, extraction schema, temperature 0). Dev pages
    are already resized by the shared `apx.extract` function, as at serving time."""
    jpegs = [(data_dir / p).read_bytes() for p in row.image_paths]
    return StructuredRequest(
        model=model,
        messages=b_messages(jpegs),
        json_schema=extraction_json_schema(),
        temperature=0.0,
        seed=0,
    )


async def run_smoke(
    client: LlmClient, rows: Sequence[TrainingSample], data_dir: Path, model: str
) -> list[SmokeDoc]:
    """Sequential calls; a JSON object that did not stop on the token limit counts as valid,
    the same rule as `LlmExtractor` and the dev metric."""
    docs = []
    for row in rows:
        result = await client.complete_structured(smoke_request(row, data_dir, model))
        valid = result.finish_reason != "length" and parse_prediction(result.content) is not None
        docs.append(
            SmokeDoc(
                sample_id=row.sample_id,
                pages=len(row.image_paths),
                valid_json=valid,
                finish_reason=result.finish_reason,
                latency_ms=round(result.latency_ms, 1),
            )
        )
        print(f"smoke: {row.sample_id} valid_json={valid} finish={result.finish_reason}")
    return docs


def load_rows(path: Path) -> list[TrainingSample]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SmokeError(f"cannot read {path}: {exc}") from exc
    return [TrainingSample.model_validate_json(x) for x in lines if x.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.smoke", description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-swap/gateway")
    parser.add_argument("--model", default=CHALLENGER_MODEL)
    parser.add_argument("--run-dir", type=Path, help="default: the newest models/extractor_b/b-*")
    return parser.parse_args(argv)


async def _main(args: argparse.Namespace) -> SmokeResult:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    run_dir: Path = args.run_dir or latest_run_dir(cfg.runs_dir)
    export_dir = run_dir / "export"
    manifest = verify_export(export_dir)
    if manifest is None:
        raise SmokeError(f"no verified export in {export_dir}; run `make export-b` first")
    rows = pick_rows(load_rows(cfg.data_dir / "dev.jsonl"))
    settings = LlmSettings(base_url=args.base_url, allowed_models=frozenset({args.model}))
    client = LlamaSwapClient(settings)
    try:
        docs = await run_smoke(client, rows, cfg.data_dir, args.model)
    finally:
        await client.aclose()
    valid = sum(d.valid_json for d in docs)
    result = SmokeResult(
        model=args.model,
        run_id=manifest.run_id,
        step=manifest.step,
        q4_sha256=manifest.artifact(Q4_NAME).sha256,
        documents=len(docs),
        valid=valid,
        passed=valid == len(docs),
        docs=tuple(docs),
        created_at=datetime.now(UTC),
    )
    atomic_write_json(export_dir / SMOKE_NAME, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a live stack
    try:
        result = asyncio.run(_main(parse_args(argv)))
    except ApxError as exc:
        print(f"smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    verdict = "PASS" if result.passed else "FAIL"
    print(f"smoke: {result.valid}/{result.documents} valid JSON -> {verdict}")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
