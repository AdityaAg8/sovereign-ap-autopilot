"""Unit tests for the M3.4 bake-off runner and its ExtractBench arm (`evals.bakeoff.run`,
`evals.bakeoff.extractbench`, `evals.bakeoff.parsed`) with a fake llama-swap and scripted
nvidia-smi: identity-stamped resumable rows, per-document rejections vs run-ending failures, the
VRAM session log, the probe gate, the parse cache and the schema-constrained prompt."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from apx.common.errors import ModelBusyError, ModelContextError, ModelServingError
from apx.llm.interfaces import CompletionResult, ImagePart, StructuredRequest, TextPart
from apx.parse.interfaces import ParsedDocument
from evals.bakeoff.arms import BakeoffError, arm_by_key
from evals.bakeoff.extractbench import (
    ExtractBenchPromptBuilder,
    ExtractBenchPromptSettings,
    extractbench_row,
    failure_kind,
    is_request_rejection,
    predicted_fields,
    schema_digest,
    score_extractbench_rows,
)
from evals.bakeoff.parsed import ParseCache
from evals.bakeoff.probe import ProbeResult
from evals.bakeoff.run import (
    ArmIdentities,
    arm_files,
    load_extractbench_rows,
    load_vram_sessions,
    require_probe_ok,
    reuse_a_rows,
    run_arm,
)
from evals.extraction_a import ROW_SCHEMA, load_rows
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
    if "DOWN" in text:
        return ModelServingError("llama-swap request to /v1/chat/completions failed: ConnectError")
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


async def test_an_unreachable_server_ends_the_run_but_keeps_rows_and_the_session(
    setup: dict[str, Any],
) -> None:
    setup["cache"]._documents[E2] = document(E2, texts=("DOWN",))
    with pytest.raises(ModelServingError, match="ConnectError"):
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


class _StallingLlm(FakeLlm):
    """A llama-server that never answers a request whose text holds STALL."""

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        text = "".join(p.text for p in request.messages[1].content if isinstance(p, TextPart))
        if "STALL" in text:
            self.calls.append(request)
            await asyncio.sleep(30)
        return await super().complete_structured(request)


async def test_timeouts_and_5xx_are_recorded_rows_counted_against_the_arm(
    setup: dict[str, Any],
) -> None:
    setup["cache"]._documents[E1] = document(E1, texts=("STALL",))
    setup["cache"]._documents[E2] = document(E2, texts=("CRASH",))
    setup["records"]["d3"] = _record("d3")
    setup["shas"]["d3"] = "c" * 64
    setup["cache"]._documents["c" * 64] = document("c" * 64, texts=("Receipt", "ACME"))
    logs: list[str] = []
    unloads = Unloads()
    await run_arm(
        arm_by_key("q38-27b"),
        uae_shas=[U1],
        uae_data_dir=setup["data_dir"],
        extractbench_records=setup["records"],
        extractbench_shas=setup["shas"],
        files=setup["files"],
        identities=_identities(),
        client=_StallingLlm(_reply),
        unload=unloads,
        sample_fn=readings([1.5], tail=8.25),
        cache=setup["cache"],
        request_timeout_s=0.05,
        log=logs.append,
        interval_s=0.001,
    )
    eb = load_extractbench_rows(setup["files"].extractbench)
    assert eb["d1"]["error_kind"] == "timeout" and not eb["d1"]["valid_json"]
    assert "no answer within" in eb["d1"]["error"] and eb["d1"]["latency_ms"] is None
    assert eb["d2"]["error_kind"] == "server_error" and "502" in eb["d2"]["error"]
    assert eb["d3"]["valid_json"] and eb["d3"]["error_kind"] is None
    assert unloads.count == 4  # before, after each of the two failed requests, after the arm
    assert any(line.endswith(" timeout") for line in logs)
    # Scored, not skipped: the failed documents are misses in the arm's value F1.
    score = score_extractbench_rows(setup["records"], eb, ["d1", "d2", "d3"], Path("absent"))
    assert score.value_f1.recall < 0.5


async def test_consecutive_server_failures_end_the_run(setup: dict[str, Any]) -> None:
    for doc_id, sha in (("d1", E1), ("d2", E2), ("d3", "c" * 64)):
        setup["records"][doc_id] = _record(doc_id)
        setup["shas"][doc_id] = sha
        setup["cache"]._documents[sha] = document(sha, texts=("CRASH",))
    setup["cache"]._documents[U2] = document(U2)
    with pytest.raises(BakeoffError, match=r"3 ExtractBench requests in a row.*d1, d2, d3"):
        await _run(setup, FakeLlm(_reply), _identities(), [])
    assert len(load_extractbench_rows(setup["files"].extractbench)) == 3


def test_failure_kinds_separate_documents_from_the_server() -> None:
    timeout = ModelServingError("llama-swap request to /x failed: ReadTimeout")
    timeout.__cause__ = httpx.ReadTimeout("slow")
    assert failure_kind(timeout) == "timeout" and failure_kind(TimeoutError()) == "timeout"
    assert failure_kind(ModelServingError("x returned HTTP 500: crash")) == "server_error"
    assert failure_kind(ModelServingError("x returned HTTP 400: bad schema")) == "rejected"
    assert failure_kind(ModelBusyError("x returned HTTP 503: loading")) is None
    assert failure_kind(ModelServingError("request to /x failed: ConnectError")) is None
    assert failure_kind(ValueError("x returned HTTP 500")) is None


BASE_ID = RowIdentity(model="extractor-a", gguf_sha256="a" * 64, prompt_version="pa")


def _a_row(sha: str, identity: RowIdentity = BASE_ID, schema: int = ROW_SCHEMA) -> dict[str, Any]:
    return {
        "sha": sha,
        "row_schema": schema,
        "valid_json": True,
        "latency_ms": 900.0,
        "fields": {"invoice_number": {"pred": "INV-1", "correct": True}},
        **identity.as_row_fields(),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_a_test_rows_are_reused_only_when_the_identity_matches(tmp_path: Path) -> None:
    source, target = tmp_path / "extraction_a.jsonl", tmp_path / "rows" / "a.uae.jsonl"
    _write_rows(source, [_a_row(U1), _a_row(U2), _a_row("9" * 64)])
    logs: list[str] = []
    assert reuse_a_rows(source, target, BASE_ID, [U1, U2], logs.append) == 2
    rows = load_rows(target)
    assert set(rows) == {U1, U2} and rows[U1]["reused_from"] == source.as_posix()
    assert rows[U1]["error"] is None and rows[U1]["latency_ms"] == 900.0
    assert reuse_a_rows(source, target, BASE_ID, [U1, U2], logs.append) == 0  # idempotent
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2
    other = RowIdentity(model="extractor-a", gguf_sha256="b" * 64, prompt_version="pa")
    _write_rows(source, [_a_row(U1), _a_row(U2, other)])
    fresh = tmp_path / "fresh.jsonl"
    assert reuse_a_rows(source, fresh, BASE_ID, [U1, U2], logs.append) == 0
    assert not fresh.exists() and "1 rows of another identity" in logs[-1]
    _write_rows(source, [_a_row(U1, schema=ROW_SCHEMA - 1)])
    assert reuse_a_rows(source, fresh, BASE_ID, [U1], logs.append) == 0
    assert "1 of an old row schema" in logs[-1]
    assert reuse_a_rows(tmp_path / "none.jsonl", fresh, BASE_ID, [U1], logs.append) == 0
    # The bake-off file itself must still hold arm a's identity.
    _write_rows(source, [_a_row(U1, other)])
    with pytest.raises(RowIdentityError):
        reuse_a_rows(source, target, other, [U1], logs.append)


def _uae_docs(tmp_path: Path) -> tuple[Path, list[str], dict[str, dict[str, Any]]]:
    data_dir = tmp_path / "uae"
    data_dir.mkdir()
    shas = [f"{i:064x}" for i in range(8)]
    for sha in shas:
        write_uae_doc(data_dir, sha)
    return data_dir, shas, {sha: {**_a_row(sha), "error": None} for sha in shas}


async def test_a_challenger_that_cannot_win_stops_at_the_first_look(tmp_path: Path) -> None:
    data_dir, shas, baseline = _uae_docs(tmp_path)
    files = arm_files(tmp_path / "rows", arm_by_key("q38-27b"))
    logs: list[str] = []

    async def run(llm: FakeLlm) -> Any:
        return await run_arm(
            arm_by_key("q38-27b"),
            uae_shas=shas,
            uae_data_dir=data_dir,
            extractbench_records={"d1": _record("d1")},
            extractbench_shas={"d1": E1},
            files=files,
            identities=_identities(),
            client=llm,
            unload=Unloads(),
            sample_fn=readings([1.5], tail=8.25),
            cache=_Cache({sha: document(sha) for sha in [*shas, E1]}),
            baseline_rows=baseline,
            log=logs.append,
            interval_s=0.001,
        )

    wrong = FakeLlm(lambda _: invoice_json(invoice_number="INV-9"))
    stopped = await run(wrong)
    assert stopped is not None and (stopped.look, stopped.n) == (1, 2)
    assert len(wrong.calls) == 2 and set(load_rows(files.uae)) == set(shas[:2])
    assert not files.extractbench.exists()  # a stopped arm spends nothing on ExtractBench
    (session,) = load_vram_sessions(files.vram)
    assert session["stopped_at_look"] == 1 and session["uae_done"] == 2
    again = FakeLlm(_reply)
    assert await run(again) == stopped and again.calls == []  # a resume re-derives the stop
    assert "stopped for futility at look 1" in logs[-1]


@pytest.mark.parametrize(
    ("key", "number"),
    [("q38-27b", "INV-1"), ("a", "INV-9"), ("q35-9b-mtp", "INV-9")],
)
async def test_a_possible_winner_the_baseline_and_the_latency_arm_run_in_full(
    tmp_path: Path, key: str, number: str
) -> None:
    data_dir, shas, baseline = _uae_docs(tmp_path)
    files = arm_files(tmp_path / "rows", arm_by_key(key))
    llm = FakeLlm(lambda _: invoice_json(invoice_number=number))
    result = await run_arm(
        arm_by_key(key),
        uae_shas=shas,
        uae_data_dir=data_dir,
        extractbench_records={},
        extractbench_shas={},
        files=files,
        identities=_identities(),
        client=llm,
        unload=Unloads(),
        sample_fn=readings([1.5], tail=8.25),
        cache=_Cache({sha: document(sha) for sha in shas}),
        baseline_rows=baseline,
        log=lambda _: None,
        interval_s=0.001,
    )
    assert result is None and len(load_rows(files.uae)) == 8 and len(llm.calls) == 8
