"""Unit tests for the runtime model allowlist (PRD §4): `LlamaSwapClient` refuses any alias
outside `RUNTIME_MODEL_ALIASES` with a typed error before sending anything, and neither apx/ nor
the runtime compose config ever names the offline red-team model."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from apx.common.errors import ModelNotAllowedError
from apx.llm.client import RUNTIME_MODEL_ALIASES, LlamaSwapClient, LlmSettings
from apx.llm.interfaces import ChatMessage, ToolRequest

ROOT = Path(__file__).resolve().parents[2]


def _request(model: str) -> ToolRequest:
    return ToolRequest(model=model, messages=(ChatMessage(role="user", content="hi"),), tools=())


async def test_default_client_refuses_offline_aliases_before_any_request() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = json.loads(request.content)
        message = {"role": "assistant", "content": "ok"}
        choice = {"message": message, "finish_reason": "stop"}
        return httpx.Response(200, json={"model": body["model"], "choices": [choice]})

    client = LlamaSwapClient(
        LlmSettings(base_url="http://llm.test"), transport=httpx.MockTransport(handler)
    )
    for alias in ("redteam-uncensored", "bakeoff-challenger", "anything"):
        with pytest.raises(ModelNotAllowedError):
            await client.complete_with_tools(_request(alias))
        with pytest.raises(ModelNotAllowedError):
            await client.embed(alias, ["x"])
    assert sent == []
    assert (await client.complete_with_tools(_request("ops-chat"))).content == "ok"
    await client.aclose()


def test_runtime_aliases_are_served_by_llama_swap_and_exclude_red_team() -> None:
    config = (ROOT / "infra" / "llama-swap.yaml").read_text(encoding="utf-8")
    for alias in RUNTIME_MODEL_ALIASES:
        assert alias in config
        assert "redteam" not in alias


def test_apx_and_runtime_config_never_reference_the_red_team_model() -> None:
    files = [*(ROOT / "apx").rglob("*.py"), ROOT / "infra" / "compose.yaml"]
    files += [p for p in (ROOT / ".env.example",) if p.exists()]
    offenders = [
        p.relative_to(ROOT).as_posix()
        for p in files
        if "redteam" in p.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []
