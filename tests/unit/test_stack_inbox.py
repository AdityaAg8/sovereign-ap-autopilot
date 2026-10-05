"""`evals.stack_inbox.host_inbox`: the scripts that drop documents into the running stack (e2e
pipeline, attack suite, demo) drop them into the folder the worker watches, so a run on a second
database (`INGEST_INBOX_DIR=data/inbox_attacks`) never leaves files in the live inbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from apx.common.errors import ConfigError
from evals.stack_inbox import host_inbox


def test_defaults_to_the_inbox_under_the_data_folder(tmp_path: Path) -> None:
    assert host_inbox(tmp_path / "data", tmp_path, environ={}) == tmp_path / "data" / "inbox"
    blank = {"INGEST_INBOX_DIR": "  "}
    assert host_inbox(tmp_path / "data", tmp_path, environ=blank) == tmp_path / "data" / "inbox"


def test_follows_the_workers_inbox_relative_to_the_repo_root(tmp_path: Path) -> None:
    # The worker resolves INGEST_INBOX_DIR against /app, the repo root inside the container.
    env = {"INGEST_INBOX_DIR": "data/inbox_attacks"}
    assert host_inbox(tmp_path / "x", tmp_path, environ=env) == tmp_path / "data" / "inbox_attacks"


def test_an_explicit_folder_wins_over_the_environment(tmp_path: Path) -> None:
    env = {"INGEST_INBOX_DIR": "data/inbox_attacks"}
    chosen = tmp_path / "elsewhere"
    assert host_inbox(tmp_path / "data", tmp_path, chosen, environ=env) == chosen


@pytest.mark.parametrize(
    "value",
    [
        "/app/data/inbox_demo",
        "../inbox",
        "data/../../inbox",
        "inbox_attacks",  # outside data/, the only folder the worker shares with the host
        "apx/inbox",
        r"data\inbox_attacks",  # one directory name in the Linux container
        "data",
    ],
)
def test_refuses_a_value_it_cannot_map_to_the_host(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="INGEST_INBOX_DIR"):
        host_inbox(tmp_path / "data", tmp_path, environ={"INGEST_INBOX_DIR": value})
