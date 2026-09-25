"""CLI for the M9.5 chat evals (`python -m evals.chat`, `make eval-chat`): runs the scripted chat
tasks and the AgentDojo banking attack suite for the stock and the uncensored agent model on a
configurable OpenAI-compatible endpoint, then writes `reports/chat_evals.md` and `.json`.
`--dry-run` swaps the endpoint for the in-process scripted fake and makes no network calls."""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from evals.chat.agentdojo_adapter import (
    BankingResult,
    LocalEndpointLLM,
    banking_task_ids,
    run_banking,
)
from evals.chat.chat_runner import run_tasks
from evals.chat.fake_model import FakeEndpoint, keyword_chat_policy, read_then_answer_policy
from evals.chat.report import ChatEvalReport, ModelResults, now, render_markdown
from evals.chat.scorers import score_task, summarize
from evals.chat.tasks import DEFAULT_TASKS, DEFAULT_WORLD, load_tasks

DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8080"
"""llama-swap through the gateway, as seen from the host (apx/llm/client.py)."""
DEFAULT_STOCK_ALIAS: Final[str] = "ops-chat"
DEFAULT_REDTEAM_ALIAS: Final[str] = "redteam-uncensored"
LOCAL_HOSTS: Final[frozenset[str]] = frozenset(
    {"127.0.0.1", "localhost", "::1", "gateway", "llama-swap"}
)
"""Loopback and the in-network gateway/llama-swap names; anything else needs --allow-remote."""
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals.chat", description=__doc__)
    env = os.environ.get
    p.add_argument("--base-url", default=env("CHAT_EVAL_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--stock-alias", default=env("CHAT_EVAL_STOCK_ALIAS", DEFAULT_STOCK_ALIAS))
    p.add_argument("--redteam-alias", default=env("CHAT_EVAL_REDTEAM_ALIAS", DEFAULT_REDTEAM_ALIAS))
    p.add_argument(
        "--models", nargs="+", choices=("stock", "uncensored"), default=["stock", "uncensored"]
    )
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    p.add_argument("--limit", type=int, default=None, help="first N chat tasks only")
    p.add_argument("--banking-user-tasks", type=int, default=None, help="first N user tasks")
    p.add_argument(
        "--banking-injection-tasks", type=int, default=None, help="first N injection tasks"
    )
    p.add_argument("--skip-chat", action="store_true")
    p.add_argument("--skip-agentdojo", action="store_true")
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit a base URL outside loopback and the local gateway",
    )
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "chat_evals.md")
    p.add_argument("--logdir", type=Path, default=REPO_ROOT / "logs" / "agentdojo")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="use the in-process scripted fake endpoint; no network, no GPU",
    )
    return p


def _print_progress(label: str, done: int, total: int) -> None:
    """One flushed line per finished chat task; tools/ops-dashboard parses `[<model>] chat i/n`."""
    print(f"[{label}] chat {done}/{total}", flush=True)


def _first(ids: list[str], n: int | None) -> list[str] | None:
    return None if n is None else ids[:n]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    host = urlsplit(args.base_url).hostname or ""
    if not args.dry_run and not args.allow_remote and host not in LOCAL_HOSTS:
        parser.error(f"--base-url host {host!r} is not local; pass --allow-remote to use it")
    tasks = load_tasks(args.tasks)[: args.limit]
    world = json.loads(args.world.read_text(encoding="utf-8"))
    aliases = {"stock": args.stock_alias, "uncensored": args.redteam_alias}

    results: list[ModelResults] = []
    for label in args.models:
        alias = aliases[label]
        chat = None
        if not args.skip_chat:
            fake = FakeEndpoint(keyword_chat_policy) if args.dry_run else None
            runs = asyncio.run(
                run_tasks(
                    tasks,
                    world,
                    base_url=args.base_url,
                    model=alias,
                    transport=fake.transport() if fake else None,
                    timeout_s=args.timeout_s,
                    on_progress=functools.partial(_print_progress, label),
                )
            )
            by_id = {t.id: t for t in tasks}
            scores = [score_task(by_id[r.task_id], r) for r in runs]
            chat = summarize(scores, errors=sum(r.error is not None for r in runs))
            print(
                f"[{label}] chat: tool-call acc {chat.tool_call_accuracy:.3f}, "
                f"citation faithfulness {chat.citation_faithfulness:.3f}",
                flush=True,
            )
        banking = None
        if not args.skip_agentdojo:
            print(f"[{label}] agentdojo banking: started", flush=True)
            banking = _banking(args, alias)
            print(
                f"[{label}] banking: utility {banking.utility:.3f}, "
                f"ASR {banking.attack_success_rate:.3f}",
                flush=True,
            )
        results.append(ModelResults(label=label, alias=alias, chat=chat, banking=banking))

    report = ChatEvalReport(
        dry_run=args.dry_run,
        base_url="in-process fake endpoint" if args.dry_run else args.base_url,
        tasks_file=_display(args.tasks),
        n_tasks=len(tasks),
        generated_at=now(),
        models=tuple(results),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    args.out.with_suffix(".json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {_display(args.out)}")
    return 0


def _banking(args: argparse.Namespace, alias: str) -> BankingResult:
    user_ids, injection_ids = banking_task_ids()
    fake = FakeEndpoint(read_then_answer_policy) if args.dry_run else None
    llm = LocalEndpointLLM(
        args.base_url,
        alias,
        timeout_s=args.timeout_s,
        transport=fake.transport() if fake else None,
    )
    try:
        return run_banking(
            llm,
            args.logdir / alias,
            user_tasks=_first(user_ids, args.banking_user_tasks),
            injection_tasks=_first(injection_ids, args.banking_injection_tasks),
        )
    finally:
        llm.close()


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
