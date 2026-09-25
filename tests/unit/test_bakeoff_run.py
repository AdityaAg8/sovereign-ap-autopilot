"""Unit tests for the M3.4 bake-off runner and its ExtractBench arm (`evals.bakeoff.run`,
`evals.bakeoff.extractbench`, `evals.bakeoff.parsed`) with a fake llama-swap and scripted
nvidia-smi: identity-stamped resumable rows, per-document rejections vs run-ending failures, the
VRAM session log, the probe gate, the parse cache and the schema-constrained prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apx.common.errors import ModelContextError, ModelServingError
from apx.llm.interfaces import ImagePart, StructuredRequest, TextPart
from apx.parse.interfaces import ParsedDocument
from evals.bakeoff.arms import BakeoffError, arm_by_key
from evals.bakeoff.extractbench import (
    ExtractBenchPromptBuilder,
    ExtractBenchPromptSettings,
    extractbench_row,
    is_request_rejection,
    predicted_fields,
    schema_digest,
)
from evals.bakeoff.parsed import ParseCache
from evals.bakeoff.probe import ProbeResult
from evals.bakeoff.run import (
    ArmIdentities,
    arm_files,
    load_extractbench_rows,
    load_vram_sessions,
    require_probe_ok,
    run_arm,
)
from evals.extraction_a import load_rows
from evals.extraction_identity import RowIdentity, RowIdentityError
from tests.unit._bakeoff_fakes import (
    FakeLlm,
    Unloads,
    document,
    invoice_json,
    readings,
    write_uae_doc,
)

U1, U2 = "1" * 64, "2" * 64
E1, E2 = "e" * 64, "f" * 64
SCHEMA = {
    "type": "object",
    "title": "Receipt",
    "properties": {
        "total": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Total " * 40},
        "vendor": {"type": "string", "default": None, "description": "Vendor name as printed."},
    },
}


def _record(doc_id: str, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"short/{doc_id}",
        "pdf": f"docs/short/{doc_id}.pdf",
        "data_schema": json.dumps(SCHEMA),
        "expected_output": expected or {"total": 12.5, "vendor": "ACME"},
        "field_rules": {},
    }


def _identities(prompt: str = "p1") -> ArmIdentities:
    uae = RowIdentity(model="bakeoff-q38-27b", gguf_sha256="d" * 64, prompt_version=prompt)
    eb = RowIdentity(model="bakeoff-q38-27b", gguf_sha256="d" * 64, prompt_version=prompt + "eb")
    return ArmIdentities(uae=uae, extractbench=eb)


class _Cache(ParseCache):
    """ParseCache over in-memory documents (the real store is covered separately)."""

    def __init__(self, documents: dict[str, ParsedDocument]) -> None:
        self._documents = documents

    def get(self, sha: str) -> ParsedDocument:
        return self._documents[sha]


def _reply(request: StructuredRequest) -> str | Exception:
    text = "".join(p.text for p in request.messages[1].content if isinstance(p, TextPart))
    if "OVERFLOW" in text:
        return ModelContextError("llama-swap returned HTTP 400: context size has been exceeded")
    if "BADSCHEMA" in text:
        return ModelServingError("llama-swap /v1/chat/completions returned HTTP 400: bad grammar")
    if "CRASH" in text:
        return ModelServingError("llama-swap /v1/chat/completions returned HTTP 502: gone")
    if "<schema>" in text:
        return json.dumps({"total": 12.5, "vendor": "ACME"})
    return invoice_json(invoice_number="INV-1")


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Any]:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    write_uae_doc(data_dir, U1)
    write_uae_doc(data_dir, U2)
    documents = {
        U1: document(U1),
        U2: document(U2, texts=("OVERFLOW",)),
        E1: document(E1, texts=("Receipt", "ACME")),
        E2: document(E2, texts=("BADSCHEMA",)),
    }
    return {
        "data_dir": data_dir,
        "cache": _Cache(documents),
        "files": arm_files(tmp_path / "rows", arm_by_key("q38-27b")),
        "records": {"d1": _record("d1"), "d2": _record("d2")},
        "shas": {"d1": E1, "d2": E2},
    }


async def _run(
    setup: dict[str, Any], llm: FakeLlm, identities: ArmIdentities, logs: list[str]
) -> Unloads:
    unloads = Unloads()
    await run_arm(
        arm_by_key("q38-27b"),
        uae_shas=[U1, U2],
        uae_data_dir=setup["data_dir"],
        extractbench_records=setup["records"],
        extractbench_shas=setup["shas"],
        files=setup["files"],
        identities=identities,
        client=llm,
        unload=unloads,
        sample_fn=readings([1.5], tail=8.25),
        cache=setup["cache"],
        log=logs.append,
        interval_s=0.001,
    )
    return unloads


async def test_run_writes_identity_rows_rejections_and_a_vram_session(
    setup: dict[str, Any],
) -> None:
    logs: list[str] = []
    unloads = await _run(setup, FakeLlm(_reply), _identities(), logs)
    files = setup["files"]
    uae = load_rows(files.uae)
    assert uae[U1]["valid_json"] and uae[U1]["fields"]["invoice_number"]["correct"]
    assert uae[U1]["model"] == "bakeoff-q38-27b" and uae[U1]["prompt_version"] == "p1"
    assert not uae[U2]["valid_json"] and "context size" in uae[U2]["error"]
    eb = load_extractbench_rows(files.extractbench)
    assert (
        eb["d1"]["output"] == {"total": 12.5, "vendor": "ACME"} and eb["d1"]["latency_ms"] == 1000
    )
    assert not eb["d2"]["valid_json"] and "bad grammar" in eb["d2"]["error"]
    assert eb["d1"]["prompt_version"] == "p1eb"
    (session,) = load_vram_sessions(files.vram)
    assert session["idle_gb"] == 1.5 and session["peak_gb"] == 8.25 and session["misses"] == 0
    assert session["uae_done"] == 2 and session["extractbench_done"] == 2
    assert unloads.count == 2
    assert logs[0].startswith("[1/2] q38-27b uae 111111111111 valid=True")
    assert logs[3].startswith("[2/2] q38-27b extractbench d2 valid=False")


async def test_resume_skips_done_rows_and_refuses_foreign_ones(setup: dict[str, Any]) -> None:
    await _run(setup, FakeLlm(_reply), _identities(), [])
    again = FakeLlm(_reply)
    logs: list[str] = []
    unloads = await _run(setup, again, _identities(), logs)
    assert again.calls == [] and unloads.count == 0 and "all 2 UAE and 2 ExtractBench" in logs[0]
    with pytest.raises(RowIdentityError, match="another run"):
        await _run(setup, FakeLlm(_reply), _identities(prompt="p2"), [])


async def test_a_serving_failure_ends_the_run_but_keeps_rows_and_the_session(
    setup: dict[str, Any],
) -> None:
    setup["cache"]._documents[E2] = document(E2, texts=("CRASH",))
    with pytest.raises(ModelServingError, match="502"):
        await _run(setup, FakeLlm(_reply), _identities(), [])
    files = setup["files"]
    assert set(load_rows(files.uae)) == {U1, U2}
    assert set(load_extractbench_rows(files.extractbench)) == {"d1"}
    (session,) = load_vram_sessions(files.vram)
    assert session["extractbench_done"] == 1 and session["peak_gb"] == 8.25


def _probe(status: str) -> ProbeResult:
    return ProbeResult(
        arm="q38-27b",
        alias="bakeoff-q38-27b",
        status=status,  # type: ignore[arg-type]
        reason="whole-card peak 12.40 GB > 11.8 GB",
        idle_gb=1.0,
        peak_gb=12.4,
        model_gb=11.4,
        concurrency=1,
        requests=2,
        failed_requests=0,
        vram_samples=9,
        measured_at="2026-09-25T00:00:00+00:00",
    )


def test_no_eval_hours_without_an_in_budget_probe() -> None:
    arm = arm_by_key("q38-27b")
    with pytest.raises(BakeoffError, match="no VRAM probe"):
        require_probe_ok({}, arm)
    with pytest.raises(BakeoffError, match=r"disqualified.*12.40 GB"):
        require_probe_ok({"q38-27b": _probe("over_budget")}, arm)
    require_probe_ok({"q38-27b": _probe("ok")}, arm)


async def test_parse_cache_parses_once_and_survives_a_restart(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    parsed: list[str] = []

    async def parse(sha: str, data: bytes) -> ParsedDocument:
        parsed.append(sha)
        assert data == b"%PDF-1.4 fake"
        return document(sha, pages=2)

    cache = ParseCache(tmp_path / "cache")
    logs: list[str] = []
    assert await cache.fill([(U1, pdf), (U2, pdf)], parse, logs.append) == 2
    assert logs == [f"[1/2] parse {U1[:12]} pages=2", f"[2/2] parse {U2[:12]} pages=2"]
    reopened = ParseCache(tmp_path / "cache")
    assert U1 in reopened and await reopened.fill([(U1, pdf)], parse, logs.append) == 0
    assert reopened.get(U2) == document(U2, pages=2) and parsed == [U1, U2]
    with pytest.raises(BakeoffError, match="parse phase"):
        reopened.get(E1)


def test_schema_digest_drops_noise_then_cuts_descriptions_to_fit() -> None:
    full = schema_digest(SCHEMA, 10_000)
    assert full is not None and "title" not in full and "default" not in full
    assert "Vendor name as printed." in full
    short = schema_digest(SCHEMA, 60)
    assert short is not None and len(short) < len(full) and '"anyOf"' in short
    assert schema_digest(SCHEMA, 5) is None


def test_extractbench_prompt_is_constrained_by_the_record_schema() -> None:
    doc = document(E1, texts=("Receipt", "ACME"), pages=3)
    request = ExtractBenchPromptBuilder(images=True).build(doc, SCHEMA, "bakeoff-q38-27b")
    assert request.json_schema == SCHEMA and request.temperature == 0.0
    parts = request.messages[1].content
    assert sum(isinstance(p, ImagePart) for p in parts) == 2
    text = "".join(p.text for p in parts if isinstance(p, TextPart))
    assert "<schema>" in text and "ACME" in text
    text_only = ExtractBenchPromptBuilder(images=False).build(doc, SCHEMA, "m")
    assert not any(isinstance(p, ImagePart) for p in text_only.messages[1].content)
    versions = {
        ExtractBenchPromptBuilder(images=True).version,
        ExtractBenchPromptBuilder(images=False).version,
        ExtractBenchPromptBuilder(
            images=True, settings=ExtractBenchPromptSettings(max_schema_tokens=10)
        ).version,
    }
    assert len(versions) == 3


def test_extractbench_text_is_budgeted_but_never_below_the_floor() -> None:
    long_doc = document(E1, texts=("word " * 20_000,))  # ~28.6k estimated tokens of text
    settings = ExtractBenchPromptSettings(slot_tokens=5000, min_text_tokens=1000)
    request = ExtractBenchPromptBuilder(images=False, settings=settings).build(
        long_doc, SCHEMA, "m"
    )
    text = "".join(p.text for p in request.messages[1].content if isinstance(p, TextPart))
    assert "[... middle of document omitted ...]" in text
    document_part = text.split("</document_text>")[0]
    assert 3000 < len(document_part) < 4000  # the 1000-token floor, not the negative remainder


async def test_extractbench_row_flags_non_object_output() -> None:
    llm = FakeLlm(lambda _: "[1, 2]")
    row = await extractbench_row(
        _record("d9"), document(E1), llm, ExtractBenchPromptBuilder(images=True), "m"
    )
    assert row["doc_id"] == "d9" and not row["valid_json"] and row["output"] is None
    assert "finish_reason=stop" in row["error"]


def test_request_rejection_is_only_a_400_or_context_overflow() -> None:
    assert is_request_rejection(ModelContextError("context size has been exceeded"))
    assert is_request_rejection(ModelServingError("x returned HTTP 400: bad schema"))
    assert not is_request_rejection(ModelServingError("x returned HTTP 500: crash"))


def test_predictions_flatten_like_gold() -> None:
    output = {"vendor": "ACME", "total": 12.5, "items": [{"b": 1, "a": 2}], "empty": None}
    assert predicted_fields(output) == {
        "vendor": "ACME",
        "total": "12.5",
        "items": '[{"a": 2, "b": 1}]',
    }
    assert predicted_fields(None) == {}
