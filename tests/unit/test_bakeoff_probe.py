"""Unit tests for the M3.4 VRAM probe (`evals.bakeoff.probe`) with a scripted nvidia-smi and a fake
llama-swap: budget verdicts at the 11.8 / 9.5 GB edges, a model that never serves, a failed burst,
fail-closed sampling gaps, the text-only arm's prompt and the merged probe file."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apx.common.errors import ModelServingError
from apx.llm.interfaces import ImagePart, StructuredRequest
from evals.bakeoff.arms import BakeoffError, arm_by_key
from evals.bakeoff.probe import (
    ProbeResult,
    budget_status,
    load_probe,
    probe_arm,
    probe_document_ids,
    write_probe,
)
from ml.extractor_b.gates import VramUnavailableError
from tests.unit._bakeoff_fakes import FakeLlm, Unloads, document, invoice_json, readings

SHA = "c" * 64
FIXED = datetime(2026, 9, 25, tzinfo=UTC)


@pytest.mark.parametrize(
    ("idle", "peak", "status"),
    [
        (2.3, 11.8, "ok"),  # both edges inclusive: 11.8 whole card, 9.5 model
        (2.3, 11.81, "over_budget"),
        (1.0, 10.5, "ok"),
        (1.0, 10.51, "over_budget"),  # model 9.51 > 9.5 although the card is under 11.8
    ],
)
def test_budget_status_edges(idle: float, peak: float, status: str) -> None:
    assert budget_status(idle, peak)[0] == status


async def test_probe_measures_the_whole_card_window_and_unloads_around_it() -> None:
    llm = FakeLlm(lambda _: invoice_json(invoice_number="INV-1"))
    unloads = Unloads()
    result = await probe_arm(
        arm_by_key("q38-27b"),
        [document(SHA, pages=5)],
        call=llm.complete_structured,
        unload=unloads,
        sample_fn=readings([1.2], tail=9.9),
        concurrency=2,
        interval_s=0.001,
        now=lambda: FIXED,
    )
    assert result.status == "ok" and result.idle_gb == 1.2 and result.peak_gb == 9.9
    assert result.model_gb == pytest.approx(8.7)
    assert len(llm.calls) == 1 + 2 * 2  # load call + 2 rounds at concurrency 2
    assert result.requests == 4 and result.failed_requests == 0
    assert all(call.model == "bakeoff-q38-27b" for call in llm.calls)
    assert unloads.count == 2 and result.measured_at == FIXED.isoformat()


async def test_over_budget_arm_is_disqualified_with_the_reason() -> None:
    result = await probe_arm(
        arm_by_key("q38-27b"),
        [document(SHA)],
        call=FakeLlm(lambda _: invoice_json()).complete_structured,
        unload=Unloads(),
        sample_fn=readings([2.0], tail=12.1),
        concurrency=1,
        interval_s=0.001,
    )
    assert result.status == "over_budget"
    assert "12.10 GB > 11.8 GB" in result.reason and "10.10 GB > 9.5 GB" in result.reason


async def test_a_model_that_does_not_serve_is_recorded_failed_not_raised() -> None:
    refused = ModelServingError("llama-swap /v1/chat/completions returned HTTP 502: exited")
    llm = FakeLlm(lambda _: refused)
    result = await probe_arm(
        arm_by_key("q35-9b-mtp"),
        [document(SHA)],
        call=llm.complete_structured,
        unload=Unloads(),
        sample_fn=readings([1.0], tail=1.1),
        concurrency=1,
        interval_s=0.001,
    )
    assert result.status == "failed" and "did not serve" in result.reason
    assert result.peak_gb is None and result.model_gb is None and len(llm.calls) == 1


async def test_a_failed_burst_request_fails_the_probe() -> None:
    calls = iter([invoice_json(), invoice_json(), ModelServingError("HTTP 500: oom")])
    result = await probe_arm(
        arm_by_key("q38-9b-distill"),
        [document(SHA)],
        call=FakeLlm(lambda _: next(calls, invoice_json())).complete_structured,
        unload=Unloads(),
        sample_fn=readings([1.0], tail=7.0),
        concurrency=1,
        interval_s=0.001,
    )
    assert result.status == "failed" and result.failed_requests == 1
    assert "oom" in result.reason


async def test_a_missed_nvidia_smi_reading_fails_closed() -> None:
    with pytest.raises(VramUnavailableError):
        await probe_arm(
            arm_by_key("a"),
            [document(SHA)],
            call=FakeLlm(lambda _: invoice_json()).complete_structured,
            unload=Unloads(),
            sample_fn=readings([1.0, None], tail=5.0),  # the window's first reading is missed
            concurrency=1,
            interval_s=0.001,
        )
    with pytest.raises(VramUnavailableError, match="idle"):
        await probe_arm(
            arm_by_key("a"),
            [document(SHA)],
            call=FakeLlm(lambda _: invoice_json()).complete_structured,
            unload=Unloads(),
            sample_fn=readings([], tail=None),
            concurrency=1,
        )


async def test_text_only_arm_sends_no_page_images() -> None:
    llm = FakeLlm(lambda _: invoice_json())
    await probe_arm(
        arm_by_key("q38-9b-distill"),
        [document(SHA, pages=5)],
        call=llm.complete_structured,
        unload=Unloads(),
        sample_fn=readings([1.0], tail=6.0),
        concurrency=1,
        interval_s=0.001,
    )

    def images(request: StructuredRequest) -> int:
        content = request.messages[1].content
        return sum(isinstance(part, ImagePart) for part in content if not isinstance(part, str))

    assert images(llm.calls[0]) == 0
    with_images = FakeLlm(lambda _: invoice_json())
    await probe_arm(
        arm_by_key("q38-27b"),
        [document(SHA, pages=5)],
        call=with_images.complete_structured,
        unload=Unloads(),
        sample_fn=readings([1.0], tail=6.0),
        concurrency=1,
        interval_s=0.001,
    )
    assert images(with_images.calls[0]) == 2


async def test_probe_needs_documents() -> None:
    with pytest.raises(BakeoffError, match="5-page"):
        await probe_arm(
            arm_by_key("a"),
            [],
            call=FakeLlm(lambda _: invoice_json()).complete_structured,
            unload=Unloads(),
            sample_fn=readings([], tail=1.0),
            concurrency=1,
        )


def _result(arm: str, status: str = "ok") -> ProbeResult:
    return ProbeResult(
        arm=arm,
        alias=arm_by_key(arm).alias,
        status=status,  # type: ignore[arg-type]
        reason="r",
        idle_gb=1.0,
        peak_gb=8.0,
        model_gb=7.0,
        concurrency=1,
        requests=2,
        failed_requests=0,
        vram_samples=10,
        measured_at=FIXED.isoformat(),
    )


def test_probe_file_merges_by_arm_in_arm_order(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    write_probe(path, [_result("q38-27b", "over_budget")])
    write_probe(path, [_result("a")])
    body = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["arm"] for entry in body["arms"]] == ["a", "q38-27b"]
    assert body["budget"] == {"whole_card_gb": 11.8, "model_gb": 9.5}
    loaded = load_probe(path)
    assert loaded["q38-27b"].status == "over_budget" and loaded["a"] == _result("a")
    assert load_probe(tmp_path / "absent.json") == {}


def test_probe_documents_are_born_digital_five_pagers_in_sha_order(tmp_path: Path) -> None:
    for sha, pages, scan in (("3" * 64, 5, False), ("1" * 64, 5, True), ("2" * 64, 4, False)):
        truth = {"n_pages": pages, "has_scan": scan}
        (tmp_path / f"{sha}.truth.json").write_text(json.dumps(truth), "utf-8")
    ids = ["3" * 64, "1" * 64, "2" * 64, "4" * 64]
    assert probe_document_ids(tmp_path, ids) == ["3" * 64]
