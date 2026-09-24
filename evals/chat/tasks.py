"""Typed loader for the M9.5 scripted chat tasks (`evals/chat/data/tasks.v1.jsonl`). Every golden
call is checked against the real ADR-0010 tool registry (`TOOL_CONTRACTS`): the tool must exist
and its concrete args must validate against the tool's params model, so the goldens cannot
drift from the tools the agent is actually offered."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, JsonValue, ValidationError

from apx.agent.interfaces import TOOL_CONTRACTS
from apx.common.errors import ConfigError
from apx.common.types import Frozen

DATA_DIR: Final[Path] = Path(__file__).parent / "data"
DEFAULT_TASKS: Final[Path] = DATA_DIR / "tasks.v1.jsonl"
DEFAULT_WORLD: Final[Path] = DATA_DIR / "world.v1.json"

Category = Literal["lookup", "fraud_explain", "vendor_payment", "refusal"]

MATCHER_KEYS: Final[frozenset[str]] = frozenset({"$free", "$one_of", "$subset_of"})
"""Golden arg matchers: `$free` (any value present), `$one_of` (value in list, the default
counts when the key is omitted), `$subset_of` (non-empty subset of the list)."""


class GoldenCall(Frozen):
    name: str
    args: dict[str, JsonValue]


class ChatTask(Frozen):
    id: str
    category: Category
    lang: Literal["en", "ar"]
    role: Literal["reviewer", "approver", "admin"]
    prompt: str
    golden_calls: tuple[GoldenCall, ...]
    alternatives: tuple[tuple[GoldenCall, ...], ...] = ()
    expect_refusal: bool = False
    expect_citations: bool = True
    tags: tuple[str, ...] = Field(default=())

    def golden_options(self) -> tuple[tuple[GoldenCall, ...], ...]:
        """The primary golden call list followed by every accepted alternative."""
        return (self.golden_calls, *self.alternatives)


def is_matcher(value: JsonValue) -> bool:
    return isinstance(value, dict) and len(value) == 1 and next(iter(value)) in MATCHER_KEYS


def _concrete_example(value: JsonValue) -> JsonValue:
    """One concrete value a matcher accepts, used only to type-check the golden."""
    if not is_matcher(value):
        return value
    assert isinstance(value, dict)
    key, spec = next(iter(value.items()))
    if key == "$free":
        return "x"
    assert isinstance(spec, list) and spec, f"{key} needs a non-empty list"
    return spec[0] if key == "$one_of" else [spec[0]]


def validate_golden(call: GoldenCall) -> None:
    """Raise ConfigError unless `call` names a registered tool with args its params accept."""
    contract = TOOL_CONTRACTS.get(call.name)  # type: ignore[call-overload]
    if contract is None:
        raise ConfigError(f"golden call names unknown tool {call.name!r}")
    unknown = set(call.args) - set(contract.params.model_fields)
    if unknown:
        raise ConfigError(f"golden {call.name} has unknown args {sorted(unknown)}")
    example = {k: _concrete_example(v) for k, v in call.args.items()}
    try:
        contract.params.model_validate(example)
    except ValidationError as exc:
        raise ConfigError(f"golden {call.name} args do not validate: {exc}") from exc


def load_tasks(path: Path = DEFAULT_TASKS) -> tuple[ChatTask, ...]:
    """Load and validate every task. Raises ConfigError on a malformed or duplicate task."""
    tasks: list[ChatTask] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            task = ChatTask.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:
            raise ConfigError(f"{path.name}:{lineno}: invalid task: {exc}") from exc
        if task.id in seen:
            raise ConfigError(f"{path.name}:{lineno}: duplicate task id {task.id}")
        seen.add(task.id)
        if task.expect_refusal and task.golden_calls:
            raise ConfigError(f"{task.id}: a refusal task has no golden calls")
        for option in task.golden_options():
            for call in option:
                validate_golden(call)
        tasks.append(task)
    return tuple(tasks)


__all__ = [
    "DEFAULT_TASKS",
    "DEFAULT_WORLD",
    "ChatTask",
    "GoldenCall",
    "is_matcher",
    "load_tasks",
    "validate_golden",
]
