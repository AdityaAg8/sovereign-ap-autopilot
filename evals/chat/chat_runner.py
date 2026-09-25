"""Runs the M9.5 scripted chat tasks through the production ops-chat stack: `LlamaSwapClient` ->
`ToolLoop` -> `PgToolDispatcher` -> `PgAgentToolbox`, with the fixture world as the SQL reader.
A recording wrapper around the real dispatcher captures what each tool returned so the scorers can
check citations against it. The model endpoint is configurable (base URL, alias, transport)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from apx.agent.dispatch import PgToolDispatcher
from apx.agent.interfaces import AgentConfig, ChatTurn, ToolCaller, ToolExecution
from apx.agent.prompts import DefaultAgentPromptBuilder
from apx.agent.runner import ToolLoop
from apx.agent.toolbox import PgAgentToolbox
from apx.common.errors import ModelOutputError, ModelServingError
from apx.common.master import User
from apx.common.types import Role, UserId
from apx.llm.client import LlamaSwapClient, LlmSettings
from apx.llm.interfaces import ToolCall, ToolSpec
from evals.chat.scorers import PredictedCall, TaskRun, ToolOutputRecord
from evals.chat.tasks import ChatTask
from evals.chat.world import RecordingSink, WorldReader


class RecordingDispatcher:
    """Delegates to the real dispatcher and keeps every execution it returned."""

    def __init__(self, inner: PgToolDispatcher) -> None:
        self._inner = inner
        self.executions: list[ToolExecution] = []
        self.calls: list[ToolCall] = []

    def specs(self, caller: ToolCaller) -> tuple[ToolSpec, ...]:
        return self._inner.specs(caller)

    async def dispatch(self, call: ToolCall, caller: ToolCaller) -> ToolExecution:
        execution = await self._inner.dispatch(call, caller)
        self.calls.append(call)
        self.executions.append(execution)
        return execution


def _record(execution: ToolExecution) -> ToolOutputRecord:
    return ToolOutputRecord(
        tool=str(execution.trace.tool),
        refs=execution.trace.refs,
        content_json=json.dumps(execution.result.content, sort_keys=True, ensure_ascii=False),
        untrusted_sources=tuple(block.source for block in execution.result.untrusted),
    )


async def run_task(
    task: ChatTask,
    world: Mapping[str, Any],
    llm: LlamaSwapClient,
    model: str,
    *,
    seed: int = 0,
) -> TaskRun:
    """One task, one fresh world. Model/serving errors become a scored `error`, never a crash."""
    dispatcher = RecordingDispatcher(
        PgToolDispatcher(PgAgentToolbox(WorldReader(world), RecordingSink()))
    )
    prompts = DefaultAgentPromptBuilder()
    loop = ToolLoop(llm, dispatcher, prompts, AgentConfig(model=model, seed=seed))
    user = User(
        id=UserId(f"eval-{task.role}"), display_name="Eval", roles=frozenset({Role(task.role)})
    )
    caller = ToolCaller(origin="chat", user=user, session_id=f"eval-{task.id}")
    request = prompts.build_chat(
        [ChatTurn(role="user", text=task.prompt)], dispatcher.specs(caller), model
    )
    try:
        reply = await loop.run(request, caller)
    except (ModelServingError, ModelOutputError) as exc:
        return TaskRun(
            task_id=task.id,
            calls=tuple(_calls(dispatcher)),
            outputs=tuple(_record(e) for e in dispatcher.executions),
            reply_text="",
            citations=(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return TaskRun(
        task_id=task.id,
        calls=tuple(_calls(dispatcher)),
        outputs=tuple(_record(e) for e in dispatcher.executions if e.trace.outcome == "ok"),
        reply_text=reply.text,
        citations=reply.citations,
        refusal=reply.refusal,
    )


def _calls(dispatcher: RecordingDispatcher) -> list[PredictedCall]:
    """What the model asked for (its raw name and args) and whether the tool ran."""
    return [
        PredictedCall(name=call.name, args=dict(call.arguments), outcome=e.trace.outcome)
        for call, e in zip(dispatcher.calls, dispatcher.executions, strict=True)
    ]


async def run_tasks(
    tasks: Sequence[ChatTask],
    world: Mapping[str, Any],
    *,
    base_url: str,
    model: str,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 600.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[TaskRun]:
    """Run every task sequentially against one endpoint and model alias. `on_progress(done, total)`
    is called after each task (the CLI prints it so the ops dashboard can show progress)."""
    llm = LlamaSwapClient(
        LlmSettings(base_url=base_url, timeout_s=timeout_s, allowed_models=frozenset({model})),
        transport=transport,
    )
    runs: list[TaskRun] = []
    try:
        for task in tasks:
            runs.append(await run_task(task, world, llm, model))
            if on_progress is not None:
                on_progress(len(runs), len(tasks))
        return runs
    finally:
        await llm.aclose()


__all__ = ["RecordingDispatcher", "run_task", "run_tasks"]
