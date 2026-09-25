"""Unit tests for the Extractor B P1-P5 gates (`ml.extractor_b.gates`, ADR-0009 §1.8): the
fail-closed nvidia-smi sampler, the sequential warm latency run (fake extract calls and a fake
llama-swap HTTP endpoint), the P5 hygiene checks, the per-gate verdicts at their thresholds and
the evidence file's tamper check. CPU only, no GPU or live model."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from apx.llm.client import LlamaSwapClient, LlmSettings
from apx.parse.interfaces import ParsedDocument
from evals.extraction_identity import RowIdentity
from evals.splits import empty_splits
from ml.extractor_b.gates import (
    LATENCY_SUBSET_SIZE,
    GateEvidence,
    GateEvidenceError,
    StrictVramSampler,
    VramUnavailableError,
    check_hygiene,
    evaluate_gates,
    extract_group_swaps,
    latency_stats,
    latency_subset,
    llm_extract_fn,
    load_evidence,
    render_markdown,
    run_latency,
    write_evidence,
)
from ml.extractor_b.interfaces import RunRecord, TrainingSample
from ml.promotion.evidence import ExtractorGateReport
from tests.unit._payload_fixtures import parsed_document
from tests.unit.ml.conftest import RowFactory, make_fingerprint, sha

REPO_ROOT = Path(__file__).resolve().parents[3]

A_IDENTITY = RowIdentity(model="extractor-a", gguf_sha256=sha("a-gguf"), prompt_version="pa")
B_IDENTITY = RowIdentity(
    model="extractor-b-challenger", gguf_sha256=sha("b-gguf"), prompt_version="pb"
)


def gate_report(**overrides: Any) -> ExtractorGateReport:
    """A report that passes all five gates by a clear margin."""
    base: dict[str, Any] = {
        "session_id": "s-1",
        "uae_test_digest": sha("test"),
        "latency_subset_digest": sha("subset"),
        "uae_f1_overall": 0.90,
        "uae_f1_en": 0.91,
        "uae_f1_ar": 0.88,
        "uae_n": 750,
        "valid_json_rate": 0.998,
        "latency_p50_s": 10.0,
        "latency_p95_s": 20.0,
        "latency_n": 200,
        "reference_a_f1_overall": 0.91,
        "reference_a_p50_s": 30.0,
        "extract_peak_vram_gb": 9.5,
        "sequential_fallback_recorded": True,
        "leakage_green": True,
        "base_revision": "rev-1",
        "lock_base_revision": "rev-1",
    }
    base.update(overrides)
    return ExtractorGateReport(**base)


def evidence(report: ExtractorGateReport | None = None) -> GateEvidence:
    report = report or gate_report()
    return GateEvidence(
        registered_model="extractor_b",
        model_version="1",
        run_id="b-run",
        mlflow_run_id="mlf",
        gguf_sha256=sha("q4"),
        mmproj_sha256=sha("mmproj"),
        champion_at_measurement=None,
        challenger_alias="extractor-b-challenger",
        reference_alias="extractor-a",
        measured_at=datetime(2026, 9, 25, tzinfo=UTC),
        git_sha="abc",
        report=report,
        reference_a_f1_en=0.92,
        reference_a_f1_ar=0.89,
        reference_a_p95_s=45.0,
        uae_n_en=564,
        uae_n_ar=186,
        vram_samples=900,
        vram_missed_readings=0,
        a_rows_sha256=sha("a-rows"),
        b_rows_sha256=sha("b-rows"),
        a_rows_identity=A_IDENTITY,
        b_rows_identity=B_IDENTITY,
        leakage_detail="4252 rows, all on the train pool",
        results=evaluate_gates(report),
    )


def verdicts(report: ExtractorGateReport) -> dict[str, bool]:
    return {g.gate: g.passed for g in evaluate_gates(report)}


# --- verdicts --------------------------------------------------------------------------------


def test_all_gates_pass_on_a_good_report() -> None:
    assert verdicts(gate_report()) == dict.fromkeys(("P1", "P2", "P3", "P4", "P5"), True)
    assert evidence().all_passed


@pytest.mark.parametrize(
    ("overrides", "gate", "passed"),
    [
        ({"uae_f1_overall": 0.880}, "P1", True),  # exactly A - 3.0 points
        ({"uae_f1_overall": 0.879}, "P1", False),
        ({"latency_p50_s": 18.0}, "P2", True),  # exactly 0.60 x A
        ({"latency_p50_s": 18.01}, "P2", False),
        ({"valid_json_rate": 0.99}, "P3", True),
        ({"valid_json_rate": 0.989}, "P3", False),
        ({"extract_peak_vram_gb": 11.8}, "P4", True),
        ({"extract_peak_vram_gb": 11.81}, "P4", False),
        ({"extract_peak_vram_gb": 11.6, "sequential_fallback_recorded": False}, "P4", False),
        ({"extract_peak_vram_gb": 11.5, "sequential_fallback_recorded": False}, "P4", True),
        ({"leakage_green": False}, "P5", False),
        ({"lock_base_revision": "rev-2"}, "P5", False),
    ],
)
def test_each_gate_at_its_threshold(overrides: dict[str, Any], gate: str, passed: bool) -> None:
    result = verdicts(gate_report(**overrides))
    assert result[gate] is passed
    assert all(v for k, v in result.items() if k != gate)
    assert evidence(gate_report(**overrides)).all_passed is passed


def test_evidence_roundtrip_and_tamper_check(tmp_path: Path) -> None:
    path = tmp_path / "gates.json"
    write_evidence(path, evidence())
    assert load_evidence(path) == evidence()

    failing = evidence(gate_report(valid_json_rate=0.5))
    data = json.loads(failing.model_dump_json())
    data["results"] = [dict(r, passed=True) for r in data["results"]]  # a hand-edited PASS
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GateEvidenceError, match="differ from its measurements"):
        load_evidence(path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(GateEvidenceError):
        load_evidence(path)


def test_markdown_lists_every_gate_and_the_verdict() -> None:
    passing = render_markdown(evidence())
    assert "**Verdict: PASS: all five gates hold.**" in passing
    assert all(f"| {gate} |" in passing for gate in ("P1", "P2", "P3", "P4", "P5"))
    assert "| Field F1, AR (186 docs) | 0.8900 | 0.8800 |" in passing
    assert "9.50 GiB" in passing and "(0 missed)" in passing and "MiB / 1024" in passing
    assert A_IDENTITY.gguf_sha256 in passing and "prompt `pb`" in passing
    failing = render_markdown(evidence(gate_report(latency_p50_s=25.0)))
    assert "**Verdict: FAIL.**" in failing and "| FAIL |" in failing


# --- P2/P4: latency run and VRAM sampling ----------------------------------------------------


def test_sampler_reports_the_peak_and_fails_closed() -> None:
    readings = iter([3.0, 9.25, 4.0])
    with StrictVramSampler(lambda: next(readings, 4.0), interval_s=0.001) as sampler:
        while sampler.samples < 3:
            pass
    assert sampler.peak_gb == 9.25 and sampler.samples >= 1

    with StrictVramSampler(lambda: None, interval_s=0.001) as absent:
        pass
    with pytest.raises(VramUnavailableError):
        _ = absent.peak_gb

    flaky = iter([5.0, None])
    with StrictVramSampler(lambda: next(flaky, 5.0), interval_s=0.001) as missed:
        while missed.misses == 0:
            pass
    with pytest.raises(VramUnavailableError, match="missed 1"):
        _ = missed.peak_gb


async def test_latency_run_is_sequential_warm_a_then_b() -> None:
    docs = [parsed_document() for _ in range(3)]
    calls: list[str] = []
    in_flight = 0

    async def extract(model: str, document: ParsedDocument) -> float:
        nonlocal in_flight
        in_flight += 1
        assert in_flight == 1  # strictly sequential
        calls.append(model)
        in_flight -= 1
        return {"a": 30.0, "b": 12.0}[model] + len(calls) / 100

    sampler = StrictVramSampler(lambda: 7.5, interval_s=0.001)
    run = await run_latency(docs, extract, sampler, a_model="a", b_model="b")
    assert calls == ["a"] * 4 + ["b"] * 4  # one untimed warm-up each, then every document
    assert run.a.n == run.b.n == 3
    assert run.a.p50_s == pytest.approx(30.03) and run.b.p50_s == pytest.approx(12.07)
    assert run.peak_vram_gb == 7.5 and run.vram_samples >= 1 and run.vram_misses == 0


async def test_latency_run_without_documents_raises() -> None:
    async def extract(model: str, document: ParsedDocument) -> float:
        raise AssertionError("not called")

    with pytest.raises(GateEvidenceError):
        await run_latency([], extract, StrictVramSampler(lambda: 1.0))
    with pytest.raises(GateEvidenceError):
        latency_stats([])


async def test_extract_fn_calls_each_alias_over_http() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        body = {
            "model": models[-1],
            "choices": [
                {
                    "message": {"role": "assistant", "content": "not json"},
                    "finish_reason": "stop",
                    "logprobs": {"content": []},
                }
            ],
        }
        return httpx.Response(200, json=body)

    client = LlamaSwapClient(
        LlmSettings(base_url="http://llm.test", allowed_models=frozenset({"a", "b"})),
        transport=httpx.MockTransport(handler),
    )
    extract = llm_extract_fn(client, a_model="a", b_model="b")
    try:
        assert await extract("a", parsed_document()) >= 0.0
        assert await extract("b", parsed_document()) >= 0.0
    finally:
        await client.aclose()
    assert models == ["a", "b"]


def test_latency_subset_is_the_first_200_sorted_ids() -> None:
    ids = [sha(str(i)) for i in range(300)]
    subset = latency_subset(list(reversed(ids)))
    assert subset == sorted(ids)[:LATENCY_SUBSET_SIZE]
    with pytest.raises(GateEvidenceError):
        latency_subset(ids[:199])


def test_extract_group_swap_is_read_from_llama_swap_config(tmp_path: Path) -> None:
    config = REPO_ROOT / "infra" / "llama-swap.yaml"
    assert extract_group_swaps(config, ("extractor-a", "extractor-b-challenger"))
    assert not extract_group_swaps(config, ("extractor-a", "arbiter"))
    copy = tmp_path / "llama-swap.yaml"
    shutil.copy(config, copy)
    copy.write_text(copy.read_text(encoding="utf-8").replace("swap: true", "swap: false"), "utf-8")
    assert not extract_group_swaps(copy, ("extractor-a", "extractor-b-challenger"))
    (tmp_path / "broken.yaml").write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(GateEvidenceError):
        extract_group_swaps(tmp_path / "broken.yaml", ("extractor-a",))


# --- P5: hygiene -----------------------------------------------------------------------------


def _hygiene_fixture(
    tmp_path: Path, rows: list[TrainingSample], *, train: list[str]
) -> tuple[Path, Path, Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.jsonl"
    manifest.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    from ml.extractor_b.checkpoints import sha256_file

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = RunRecord(
        run_id="b-run",
        mlflow_run_id="mlf",
        fingerprint=make_fingerprint(manifest_sha256=sha256_file(manifest)),
        created_at=datetime(2026, 9, 23, tzinfo=UTC),
    )
    (run_dir / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    splits = empty_splits().model_copy(deep=True)
    splits.sets["uae"]["train"] = train
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(splits.model_dump_json(), encoding="utf-8")
    lock = {
        "models": [
            {
                "intent": "extractor_b_train_base",
                "repo_id": "Qwen/Qwen3.5-4B",
                "revision_sha": "rev-1",
            }
        ]
    }
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return run_dir, data_dir, lock_path, splits_path


def test_hygiene_green_on_a_train_only_manifest(tmp_path: Path, make_row: RowFactory) -> None:
    rows = [make_row("r1", doc="d1"), make_row("r2", doc="d2")]
    run_dir, data_dir, lock, splits = _hygiene_fixture(tmp_path, rows, train=[sha("d1"), sha("d2")])
    result = check_hygiene(run_dir, data_dir, lock_path=lock, splits_path=splits)
    assert result.leakage_green and result.base_revision == result.lock_base_revision == "rev-1"


def test_hygiene_red_when_a_row_is_off_the_train_pool(tmp_path: Path, make_row: RowFactory) -> None:
    rows = [make_row("r1", doc="d1"), make_row("r2", doc="test-doc")]
    run_dir, data_dir, lock, splits = _hygiene_fixture(tmp_path, rows, train=[sha("d1")])
    result = check_hygiene(run_dir, data_dir, lock_path=lock, splits_path=splits)
    assert not result.leakage_green and "outside the train split" in result.leakage_detail
    assert not verdicts(gate_report(leakage_green=result.leakage_green))["P5"]


def test_hygiene_refuses_a_manifest_the_run_did_not_train_on(
    tmp_path: Path, make_row: RowFactory
) -> None:
    run_dir, data_dir, lock, splits = _hygiene_fixture(
        tmp_path, [make_row("r1", doc="d1")], train=[sha("d1")]
    )
    with (data_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(make_row("r9", doc="d1").model_dump_json() + "\n")
    with pytest.raises(GateEvidenceError, match="not the manifest"):
        check_hygiene(run_dir, data_dir, lock_path=lock, splits_path=splits)
    lock.write_text(json.dumps({"models": []}), encoding="utf-8")
    (data_dir / "manifest.jsonl").unlink()
    with pytest.raises(GateEvidenceError, match="missing"):
        check_hygiene(run_dir, data_dir, lock_path=lock, splits_path=splits)


def test_hygiene_reports_the_models_lock_revision(tmp_path: Path, make_row: RowFactory) -> None:
    run_dir, data_dir, lock, splits = _hygiene_fixture(
        tmp_path, [make_row("r1", doc="d1")], train=[sha("d1")]
    )
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["models"][0]["revision_sha"] = "rev-9"
    lock.write_text(json.dumps(data), encoding="utf-8")
    result = check_hygiene(run_dir, data_dir, lock_path=lock, splits_path=splits)
    assert result.lock_base_revision == "rev-9"
    lock.write_text(json.dumps({"models": []}), encoding="utf-8")
    with pytest.raises(GateEvidenceError, match="no base pin"):
        check_hygiene(run_dir, data_dir, lock_path=lock, splits_path=splits)
