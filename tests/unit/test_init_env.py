"""`scripts/init_env.py`: first run copies `.env.example` and generates the empty secrets; later
runs never touch a value that is already set (PRD §10 M5.6)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "init_env.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("init_env", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_run_generates_the_secret_and_later_runs_keep_it(tmp_path: Path) -> None:
    init_env = _module().init_env
    (tmp_path / ".env.example").write_text("# comment\nAPI_JWT_SECRET=\nPOSTGRES_DB=\n")
    assert init_env(tmp_path) == ["API_JWT_SECRET"]
    first = (tmp_path / ".env").read_text()
    secret = next(line for line in first.splitlines() if line.startswith("API_JWT_SECRET="))
    assert len(secret) > len("API_JWT_SECRET=") + 40
    assert "# comment" in first and "POSTGRES_DB=" in first
    assert init_env(tmp_path) == []
    assert (tmp_path / ".env").read_text() == first


def test_missing_or_quoted_empty_secret_is_generated(tmp_path: Path) -> None:
    init_env = _module().init_env
    (tmp_path / ".env").write_text("POSTGRES_DB=apx\n")
    assert init_env(tmp_path) == ["API_JWT_SECRET"]
    assert (tmp_path / ".env").read_text().startswith("POSTGRES_DB=apx\nAPI_JWT_SECRET=")
    (tmp_path / ".env").write_text('API_JWT_SECRET=""\n')
    assert init_env(tmp_path) == ["API_JWT_SECRET"]
