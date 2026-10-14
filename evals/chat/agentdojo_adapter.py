"""AgentDojo (pinned 0.1.35) banking adapter (chat evals): `LocalEndpointLLM` is a pipeline
element that talks to the local OpenAI-compatible endpoint (llama-swap through the gateway) over
httpx, and `run_banking` runs the banking suite with and without the important-instructions
prompt-injection attack, returning utility and targeted attack success rate."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import httpx
from agentdojo.agent_pipeline import (  # type: ignore[import-untyped]
    AgentPipeline,
    BasePipelineElement,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.agent_pipeline.agent_pipeline import (  # type: ignore[import-untyped]
    load_system_message,
)
from agentdojo.attacks.base_attacks import FixedJailbreakAttack  # type: ignore[import-untyped]
from agentdojo.attacks.important_instructions_attacks import (  # type: ignore[import-untyped]
    ImportantInstructionsAttack,
)
from agentdojo.benchmark import (  # type: ignore[import-untyped]
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.functions_runtime import (  # type: ignore[import-untyped]
    EmptyEnv,
    FunctionCall,
    FunctionsRuntime,
)
from agentdojo.logging import Logger, NullLogger  # type: ignore[import-untyped]
from agentdojo.task_suite.load_suites import get_suite  # type: ignore[import-untyped]
from agentdojo.types import (  # type: ignore[import-untyped]
    ChatAssistantMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from apx.common.errors import ModelOutputError, ModelServingError
from apx.common.types import Frozen

AGENTDOJO_VERSION: Final[str] = "0.1.35"
BENCHMARK_VERSION: Final[str] = "v1.2.2"
SUITE_NAME: Final[str] = "banking"
ATTACK_NAME: Final[str] = "important_instructions"


class LocalEndpointLLM(BasePipelineElement):  # type: ignore[misc]
    """AgentDojo LLM element for an OpenAI-compatible `/v1/chat/completions` endpoint. Thinking is
    off and sampling is fixed (temperature, seed) so runs are repeatable. A tool call whose
    arguments are not JSON is counted in `malformed_tool_calls` and answered as plain text."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 2048,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = f"local-{model}"
        self.model = model
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)
        self._n_calls = 0
        self.malformed_tool_calls = 0
        self.requests = 0

    def close(self) -> None:
        self._http.close()

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = EmptyEnv(),  # noqa: B008 - AgentDojo's own default
        messages: Sequence[Any] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Any, Sequence[Any], dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f.name,
                    "description": f.description,
                    "parameters": f.parameters.model_json_schema(),
                },
            }
            for f in runtime.functions.values()
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai_message(m) for m in messages],
            "temperature": self._temperature,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body["tools"] = tools
        message = self._post(body)
        output = self._to_assistant(message)
        return query, runtime, env, [*messages, output], extra_args or {}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        self.requests += 1
        try:
            response = self._http.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ModelServingError(f"chat completion request failed: {exc!r}") from exc
        if response.status_code != 200:
            raise ModelServingError(
                f"chat completion returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelOutputError(f"malformed chat completion: {exc!r}") from exc
        if not isinstance(message, dict):
            raise ModelOutputError("chat completion message is not an object")
        return message

    def _to_assistant(self, message: dict[str, Any]) -> Any:
        calls: list[Any] = []
        text = str(message.get("content") or "")
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not an object")
            except ValueError:
                self.malformed_tool_calls += 1
                continue
            self._n_calls += 1
            call_id = str(raw.get("id") or f"call_{self._n_calls}")
            calls.append(
                FunctionCall(function=str(function.get("name", "")), args=args, id=call_id)
            )
        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(text)],
            tool_calls=calls or None,
        )


def to_openai_message(message: Any) -> dict[str, Any]:
    """AgentDojo chat message -> OpenAI chat-completions message."""
    role = message["role"]
    if role in ("system", "user"):
        return {"role": role, "content": get_text_content_as_str(message["content"])}
    if role == "assistant":
        out: dict[str, Any] = {
            "role": "assistant",
            "content": get_text_content_as_str(message["content"] or []),
        }
        if message.get("tool_calls"):
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function, "arguments": json.dumps(dict(call.args))},
                }
                for call in message["tool_calls"]
            ]
        return out
    if role == "tool":
        text = get_text_content_as_str(message["content"])
        if message.get("error"):
            text = f"Error: {message['error']}"
        return {"role": "tool", "tool_call_id": message["tool_call_id"], "content": text}
    raise ModelOutputError(f"unknown AgentDojo message role {role!r}")


def build_pipeline(llm: LocalEndpointLLM) -> Any:
    """AgentDojo's standard tool-calling pipeline around the local model."""
    pipeline = AgentPipeline(
        [
            SystemMessage(load_system_message(None)),
            InitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), llm]),
        ]
    )
    pipeline.name = llm.name
    return pipeline


class ImportantInstructions(FixedJailbreakAttack):  # type: ignore[misc]
    """AgentDojo's `important_instructions` template with generic user and model names, since a
    local alias is not in AgentDojo's model-name table."""

    name = ATTACK_NAME

    def __init__(self, task_suite: Any, target_pipeline: Any) -> None:
        super().__init__(ImportantInstructionsAttack._JB_STRING, task_suite, target_pipeline)


class _DirLogger(NullLogger):  # type: ignore[misc]
    """Routes AgentDojo's per-task trace files to `logdir` instead of site-packages."""

    def __init__(self, logdir: Path) -> None:
        self._dir = str(logdir)

    def __enter__(self) -> _DirLogger:
        Logger.__enter__(self)
        self.messages: list[Any] = []
        self.logdir = self._dir
        return self


class BankingResult(Frozen):
    model: str
    user_tasks: int
    injection_pairs: int
    utility: float
    """Share of user tasks solved with no attack."""
    utility_under_attack: float
    attack_success_rate: float
    """Targeted ASR: share of (user task, injection task) pairs where the injection goal was met."""
    malformed_tool_calls: int


def _mean(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _natural(task_id: str) -> tuple[str, int]:
    stem, _, num = task_id.rpartition("_")
    return (stem, int(num)) if num.isdigit() else (task_id, -1)


def banking_task_ids() -> tuple[list[str], list[str]]:
    """(user task ids, injection task ids) of the pinned banking suite, in numeric order."""
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    return sorted(suite.user_tasks, key=_natural), sorted(suite.injection_tasks, key=_natural)


def run_banking(
    llm: LocalEndpointLLM,
    logdir: Path,
    *,
    user_tasks: Sequence[str] | None = None,
    injection_tasks: Sequence[str] | None = None,
) -> BankingResult:
    """Run banking without and with the attack. `None` selects every task in the suite."""
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    pipeline = build_pipeline(llm)
    logdir.mkdir(parents=True, exist_ok=True)
    with _DirLogger(logdir):
        clean = benchmark_suite_without_injections(
            pipeline,
            suite,
            logdir=logdir,
            force_rerun=True,
            user_tasks=user_tasks,
            benchmark_version=BENCHMARK_VERSION,
        )
        attacked = benchmark_suite_with_injections(
            pipeline,
            suite,
            ImportantInstructions(suite, pipeline),
            logdir=logdir,
            force_rerun=True,
            user_tasks=user_tasks,
            injection_tasks=injection_tasks,
            verbose=False,
            benchmark_version=BENCHMARK_VERSION,
        )
    return BankingResult(
        model=llm.model,
        user_tasks=len(clean["utility_results"]),
        injection_pairs=len(attacked["security_results"]),
        utility=_mean(list(clean["utility_results"].values())),
        utility_under_attack=_mean(list(attacked["utility_results"].values())),
        attack_success_rate=_mean(list(attacked["security_results"].values())),
        malformed_tool_calls=llm.malformed_tool_calls,
    )


__all__ = [
    "AGENTDOJO_VERSION",
    "ATTACK_NAME",
    "BENCHMARK_VERSION",
    "BankingResult",
    "ImportantInstructions",
    "LocalEndpointLLM",
    "banking_task_ids",
    "build_pipeline",
    "run_banking",
    "to_openai_message",
]
