"""Unit tests for scripts/smoke.py (PRD §10 M0.5): nvidia-smi output parsing, the VRAM
sampler thread, report rendering, and the llama-swap check logic run against a fake HTTP
transport (httpx.MockTransport) -- no GPU, no network, no real llama-swap server.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import httpx

_SPEC_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke.py"
_spec = importlib.util.spec_from_file_location("smoke", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
smoke: ModuleType = importlib.util.module_from_spec(_spec)
# Registered in sys.modules before exec: dataclasses (used by scripts/smoke.py with
# `from __future__ import annotations`) resolves annotation types via
# sys.modules[cls.__module__], which fails for an unregistered module_from_spec module.
sys.modules["smoke"] = smoke
_spec.loader.exec_module(smoke)


# --- nvidia-smi parsing / VRAM sampler ----------------------------------------------------


def test_parse_nvidia_smi_memory_used_single_gpu() -> None:
    assert smoke.parse_nvidia_smi_memory_used("4096\n") == [4.0]


def test_parse_nvidia_smi_memory_used_multi_gpu_and_blank_lines() -> None:
    output = "1024\n2048\n\n"
    assert smoke.parse_nvidia_smi_memory_used(output) == [1.0, 2.0]


def test_sample_vram_gb_returns_max_across_gpus() -> None:
    assert smoke.sample_vram_gb(runner=lambda: "1024\n8192\n") == 8.0


def test_sample_vram_gb_returns_none_when_nvidia_smi_missing() -> None:
    def raise_not_found() -> str:
        raise FileNotFoundError("nvidia-smi not found")

    assert smoke.sample_vram_gb(runner=raise_not_found) is None


def test_sample_vram_gb_returns_none_on_unparseable_output() -> None:
    assert smoke.sample_vram_gb(runner=lambda: "not-a-number\n") is None


def test_vram_sampler_tracks_peak_across_samples() -> None:
    values = iter([1.0, 5.0, 3.0, 5.0, 2.0])

    def fake_sample() -> float | None:
        return next(values, 2.0)

    with smoke.VramSampler(interval_s=0.02, sample_fn=fake_sample) as sampler:
        time.sleep(0.15)

    assert sampler.peak_gb == 5.0


def test_vram_sampler_peak_is_none_when_every_sample_is_none() -> None:
    with smoke.VramSampler(interval_s=0.02, sample_fn=lambda: None) as sampler:
        time.sleep(0.05)

    assert sampler.peak_gb is None


# --- pure check logic (dict fixtures) -----------------------------------------------------


def test_evaluate_structured_response_all_pass() -> None:
    data = {
        "choices": [
            {
                "message": {"content": json.dumps({"invoice_number": "INV-1", "total": 1.0})},
                "logprobs": {"content": [{"token": "a", "logprob": -0.1}]},
            }
        ]
    }
    checks = smoke.evaluate_structured_response(data)
    assert {c.name: c.passed for c in checks} == {
        "json_schema_output": True,
        "thinking_disabled": True,
        "logprobs": True,
    }


def test_evaluate_structured_response_flags_think_leakage_and_missing_logprobs() -> None:
    data = {
        "choices": [
            {
                "message": {"content": '<think>reasoning</think>{"invoice_number": "x"}'},
                "logprobs": None,
            }
        ]
    }
    checks = {c.name: c for c in smoke.evaluate_structured_response(data)}
    assert checks["thinking_disabled"].passed is False
    assert checks["logprobs"].passed is False
    assert checks["json_schema_output"].passed is False  # missing "total"


def test_evaluate_structured_response_handles_malformed_body() -> None:
    checks = smoke.evaluate_structured_response({"choices": []})
    assert all(not c.passed for c in checks)
    assert {c.name for c in checks} == {"json_schema_output", "logprobs", "thinking_disabled"}


def test_evaluate_tool_call_response_pass_and_fail() -> None:
    good = {"choices": [{"message": {"tool_calls": [{"function": {"name": "lookup_po"}}]}}]}
    assert smoke.evaluate_tool_call_response(good).passed is True

    bad = {"choices": [{"message": {"tool_calls": []}}]}
    assert smoke.evaluate_tool_call_response(bad).passed is False


# --- llama-swap end-to-end checks against a fake transport (mock the transport only) ------


def _client_with_handler(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_run_llama_swap_model_passes_against_a_healthy_fake_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"tool_calls": [{"function": {"name": "lookup_po"}}]}}]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"invoice_number": "INV-1", "total": 12.5})
                        },
                        "logprobs": {"content": [{"token": "a", "logprob": -0.2}]},
                    }
                ]
            },
        )

    with _client_with_handler(handler) as client:
        result = smoke.run_llama_swap_model(
            client,
            "http://fake",
            intent="extractor_a",
            served_name="extractor-a",
            stage="extract",
            include_tool_call=True,
            invoice_image_data_url="data:image/png;base64,AAAA",
        )

    assert result.reachable is True
    assert result.passed is True
    assert result.latency_ms is not None
    check_names = {c.name for c in result.checks}
    assert check_names == {
        "reachable",
        "json_schema_output",
        "thinking_disabled",
        "logprobs",
        "tool_call",
    }


def test_run_llama_swap_model_fails_when_server_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client_with_handler(handler) as client:
        result = smoke.run_llama_swap_model(
            client,
            "http://fake",
            intent="arbiter",
            served_name="arbiter",
            stage="arbitrate",
            include_tool_call=False,
            invoice_image_data_url="data:image/png;base64,AAAA",
        )

    assert result.reachable is False
    assert result.passed is False
    reachable_check = next(c for c in result.checks if c.name == "reachable")
    assert "ConnectError" in reachable_check.detail


def test_run_llama_swap_model_fails_when_thinking_leaks_even_if_reachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "<think>oops</think>{}"},
                        "logprobs": {"content": [{"token": "a", "logprob": -0.1}]},
                    }
                ]
            },
        )

    with _client_with_handler(handler) as client:
        result = smoke.run_llama_swap_model(
            client,
            "http://fake",
            intent="extractor_b_base",
            served_name="extractor-b",
            stage="extract",
            include_tool_call=False,
            invoice_image_data_url="data:image/png;base64,AAAA",
        )

    assert result.reachable is True
    assert result.passed is False


# --- transformers path: no network regardless of whether transformers is installed -------


def test_transformers_load_only_model_fails_without_local_files(tmp_path: Path) -> None:
    result = smoke.run_transformers_load_only_model(
        intent="injection_guard_base", stage="scan", repo_dir=tmp_path / "does-not-exist"
    )
    assert result.reachable is False
    assert result.passed is False
    load_check = next(c for c in result.checks if c.name == "load")
    assert load_check.detail


def test_transformers_embedding_model_fails_without_local_files(tmp_path: Path) -> None:
    result = smoke.run_transformers_embedding_model(
        intent="text_embeddings",
        stage="fraud",
        repo_dir=tmp_path / "does-not-exist",
        texts=["hello"],
    )
    assert result.reachable is False
    assert result.passed is False


# --- report rendering ----------------------------------------------------------------------


def test_render_report_marks_stage_over_budget() -> None:
    results = [
        smoke.ModelResult(
            intent="extractor_a",
            served_name="extractor-a",
            stage="extract",
            reachable=True,
            checks=(smoke.CheckResult("reachable", True),),
            peak_vram_gb=12.0,
            latency_ms=250.0,
        ),
        smoke.ModelResult(
            intent="arbiter",
            served_name="arbiter",
            stage="arbitrate",
            reachable=False,
            checks=(smoke.CheckResult("reachable", False, "ConnectError: refused"),),
            peak_vram_gb=None,
            latency_ms=None,
        ),
    ]

    report = smoke.render_report(results, budget_gb=11.5, generated_at="2026-01-01T00:00:00+00:00")

    assert "extractor_a" in report
    assert "PASS" in report
    assert "arbiter" in report
    assert "FAIL" in report
    assert "ConnectError: refused" in report
    assert "sequential groups" in report  # extract stage's 12.0 GB > 11.5 GB budget
    assert "n/a" in report  # arbiter's peak VRAM / latency are unmeasured


def test_render_report_with_no_vram_samples() -> None:
    results = [
        smoke.ModelResult(
            intent="parse",
            served_name="parse",
            stage="parse",
            reachable=False,
            checks=(smoke.CheckResult("load", False, "transformers not installed"),),
            peak_vram_gb=None,
            latency_ms=1.0,
        )
    ]

    report = smoke.render_report(results)

    assert "No VRAM samples collected" in report


# --- lock-file loading ----------------------------------------------------------------------


def test_load_runtime_models_filters_offline_only(tmp_path: Path) -> None:
    lock = {
        "models": [
            {"intent": "extractor_a", "runtime": True},
            {"intent": "bakeoff_challenger_unsloth", "runtime": False},
        ]
    }
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    models = smoke.load_runtime_models(lock_path)

    assert [m["intent"] for m in models] == ["extractor_a"]
