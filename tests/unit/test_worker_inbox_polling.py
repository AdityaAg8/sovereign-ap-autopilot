"""Compose contract for the inbox watcher: `data/` is a host bind mount, and on Docker Desktop a
file dropped on the host raises no inotify event inside the container, so the worker's watchfiles
watcher must poll. Without it the M5.6 e2e run dropped 20 documents and none were ingested."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from watchfiles.main import _default_force_polling

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "infra" / "compose.yaml"


def _worker_environment() -> dict[str, Any]:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))  # SafeLoader resolves `<<:`
    env: dict[str, Any] = compose["services"]["worker"]["environment"]
    return env


def test_the_worker_inbox_is_a_bind_mount_and_its_watcher_polls() -> None:
    volumes = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["worker"]["volumes"]
    assert "../data:/app/data" in volumes
    assert str(_worker_environment()["WATCHFILES_FORCE_POLLING"]).lower() == "true"


def test_watchfiles_reads_the_polling_switch_from_the_environment(monkeypatch: Any) -> None:
    # `watch_inbox` calls `awatch(inbox_dir)` with force_polling=None, which defers to this
    # variable; if a watchfiles upgrade stops reading it, the compose setting silently does nothing.
    monkeypatch.setenv("WATCHFILES_FORCE_POLLING", "true")
    assert _default_force_polling(None) is True
