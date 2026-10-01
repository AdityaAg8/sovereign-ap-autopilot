"""Database credentials without defaults (REVIEW C-02, L-01): `scripts/rotate_db_password.py`
sets both roles from `.env` without a secret on any command line or in its output, migration
0013 sets `apx_agent_ro`'s password from `AGENT_RO_PASSWORD`, and `apx.agent.sql` builds the
agent URL from the environment with no built-in password."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from apx.agent.sql import get_agent_database_url
from apx.common.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]
PG_SECRET = "pg-Secret_value-0123456789abcdef"
AGENT_SECRET = "agent-Secret_value-0123456789abcd"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rotate() -> ModuleType:
    return _load(REPO / "scripts" / "rotate_db_password.py", "rotate_db_password")


def _migration() -> ModuleType:
    return _load(REPO / "db" / "migrations" / "versions" / "0013_agent_ro_password.py", "m0013")


class _FakeDocker:
    def __init__(self, codes: Sequence[int] = (0, 0)) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self._codes = list(codes)

    def __call__(self, argv: Sequence[str], stdin: str | None) -> int:
        self.calls.append((list(argv), stdin))
        return self._codes.pop(0)


def _env_file(tmp_path: Path, extra: str = "") -> Path:
    env = tmp_path / ".env"
    env.write_text(
        f"POSTGRES_USER=apx\nPOSTGRES_PASSWORD={PG_SECRET}\n"
        f'export AGENT_RO_PASSWORD="{AGENT_SECRET}"\n{extra}'
    )
    return env


# --- scripts/rotate_db_password.py -------------------------------------------------------


def test_rotation_runs_one_transaction_over_stdin_and_never_puts_a_secret_in_argv(
    tmp_path: Path,
) -> None:
    module = _rotate()
    docker = _FakeDocker()
    assert module.rotate(_env_file(tmp_path), docker) == ["apx", "apx_agent_ro"]
    (up_argv, up_stdin), (psql_argv, script) = docker.calls
    assert up_argv[-4:] == ["up", "-d", "--wait", "postgres"] and up_stdin is None
    assert psql_argv[-10:] == [
        *("exec", "-T", "postgres", "psql", "-X", "-q", "-U", "apx", "-d", "apx")
    ]
    for argv, _ in docker.calls:
        assert not any(PG_SECRET in arg or AGENT_SECRET in arg for arg in argv)
    assert script is not None
    assert f"ALTER ROLE \"apx\" PASSWORD '{PG_SECRET}';" in script
    assert f"'ALTER ROLE apx_agent_ro PASSWORD %L', '{AGENT_SECRET}'" in script
    assert script.index("BEGIN;") < script.index("ALTER ROLE") < script.index("COMMIT;")
    assert "\\set ON_ERROR_STOP on" in script and "\\set SHOW_CONTEXT never" in script


def test_rotation_quotes_hand_written_passwords(tmp_path: Path) -> None:
    module = _rotate()
    script = module.rotation_sql(
        {"POSTGRES_USER": 'we"ird', "POSTGRES_PASSWORD": "it's", "AGENT_RO_PASSWORD": "a'b"}
    )
    assert "ALTER ROLE \"we\"\"ird\" PASSWORD 'it''s';" in script
    assert "%L', 'a''b')" in script


def test_rotation_refuses_missing_secrets_without_naming_values(tmp_path: Path) -> None:
    module = _rotate()
    with pytest.raises(module.RotationError, match="AGENT_RO_PASSWORD") as info:
        module.rotation_sql({"POSTGRES_PASSWORD": PG_SECRET})
    assert PG_SECRET not in str(info.value)


def test_a_failed_docker_step_is_a_typed_error(tmp_path: Path) -> None:
    module = _rotate()
    with pytest.raises(module.RotationError, match="psql failed"):
        module.rotate(_env_file(tmp_path), _FakeDocker(codes=(0, 3)))
    with pytest.raises(module.RotationError, match="up -d --wait postgres"):
        module.rotate(_env_file(tmp_path), _FakeDocker(codes=(1,)))


def test_main_prints_role_names_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _rotate()
    docker = _FakeDocker()
    monkeypatch.setattr(module, "_subprocess_runner", docker)
    assert module.main(["--env-file", str(_env_file(tmp_path))]) == 0
    printed = capsys.readouterr()
    assert "apx_agent_ro" in printed.out
    for secret in (PG_SECRET, AGENT_SECRET):
        assert secret not in printed.out and secret not in printed.err


def test_main_without_env_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _rotate().main(["--env-file", str(tmp_path / "absent.env")]) == 1
    assert "make env" in capsys.readouterr().err


# --- migration 0013 ------------------------------------------------------------------------


def test_migration_chains_after_0012() -> None:
    module = _migration()
    assert module.revision == "0013_agent_ro_password"
    assert module.down_revision == "0012_approval_digest"


def test_migration_sets_the_env_password_or_clears_it() -> None:
    statement = _migration().password_statement
    assert statement(AGENT_SECRET) == f"ALTER ROLE apx_agent_ro PASSWORD '{AGENT_SECRET}'"
    assert statement(None) == "ALTER ROLE apx_agent_ro PASSWORD NULL"
    assert statement("") == "ALTER ROLE apx_agent_ro PASSWORD NULL"


@pytest.mark.parametrize(
    "bad", ["short-pw", "has'quote-0123456789", "colon:bind-0123456789", "pct%s-0123456789"]
)
def test_migration_rejects_passwords_it_cannot_splice_safely(bad: str) -> None:
    with pytest.raises(ValueError, match="AGENT_RO_PASSWORD") as info:
        _migration().password_statement(bad)
    assert bad not in str(info.value)


def test_no_agent_password_literal_remains_in_migrations_or_agent_code() -> None:
    for path in [
        REPO / "db" / "migrations" / "versions" / "0004_agent_ro_role_and_views.py",
        REPO / "apx" / "agent" / "sql.py",
    ]:
        assert "apx-agent-ro-dev-only" not in path.read_text(encoding="utf-8"), path


# --- apx.agent.sql.get_agent_database_url -------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("AGENT_DATABASE_URL", "AGENT_RO_PASSWORD", "POSTGRES_HOST", "POSTGRES_DB"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_agent_url_is_built_from_the_env_password(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_RO_PASSWORD", "p@ss/word")
    clean_env.setenv("POSTGRES_HOST", "postgres")
    clean_env.setenv("POSTGRES_DB", "apx")
    assert get_agent_database_url() == (
        "postgresql+asyncpg://apx_agent_ro:p%40ss%2Fword@postgres:5432/apx"
    )


def test_an_explicit_agent_url_wins(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_RO_PASSWORD", "ignored")
    clean_env.setenv("AGENT_DATABASE_URL", "postgresql+asyncpg://apx_agent_ro:x@db:5432/apx")
    assert get_agent_database_url() == "postgresql+asyncpg://apx_agent_ro:x@db:5432/apx"


def test_no_agent_password_is_a_config_error(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="AGENT_RO_PASSWORD"):
        get_agent_database_url()


def test_wrong_driver_error_does_not_echo_the_password(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AGENT_DATABASE_URL", f"postgresql://apx_agent_ro:{AGENT_SECRET}@db/apx")
    with pytest.raises(ConfigError, match="asyncpg") as info:
        get_agent_database_url()
    assert AGENT_SECRET not in str(info.value)
