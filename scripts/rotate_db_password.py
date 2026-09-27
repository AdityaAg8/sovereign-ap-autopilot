"""Sets the Postgres passwords of the schema owner (`POSTGRES_USER`) and the read-only
`apx_agent_ro` role to the values in `.env`, for installs whose volume was initialised with an
old default (REVIEW C-02, L-01). Run as `make rotate-db-password` (after `make env`); idempotent,
and it never prints or passes a password on a command line."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("POSTGRES_PASSWORD", "AGENT_RO_PASSWORD")
AGENT_ROLE = "apx_agent_ro"

Runner = Callable[[Sequence[str], str | None], int]
"""Runs argv (cwd infra/) with optional stdin text and returns the exit code."""


class RotationError(RuntimeError):
    """A precondition or docker step failed; the message never contains a secret."""


def read_env(path: Path) -> dict[str, str]:
    """Parses `KEY=value` lines (optional `export `, surrounding quotes stripped); later
    assignments win, as in compose. Values stay in memory and are never logged."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def sql_literal(value: str) -> str:
    """A Postgres string literal (standard_conforming_strings, the default): quotes doubled."""
    if "\x00" in value:
        raise RotationError("a password contains NUL")
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def rotation_sql(env: Mapping[str, str]) -> str:
    """The script fed to psql on stdin: one transaction that sets both roles' passwords. The
    agent role is altered only if it exists (a volume not yet migrated past 0004 has none)."""
    missing = [key for key in REQUIRED if not env.get(key)]
    if missing:
        raise RotationError(f"missing in .env: {', '.join(missing)} (run `make env` first)")
    owner = env.get("POSTGRES_USER") or "apx"
    agent_literal = sql_literal(env["AGENT_RO_PASSWORD"])
    return (
        "\\set ON_ERROR_STOP on\n"
        # Terse errors and no CONTEXT, so a failing statement is never echoed with its password.
        "\\set VERBOSITY terse\n"
        "\\set SHOW_CONTEXT never\n"
        "BEGIN;\n"
        f"ALTER ROLE {sql_identifier(owner)} PASSWORD {sql_literal(env['POSTGRES_PASSWORD'])};\n"
        "DO $rotate$\n"
        "BEGIN\n"
        f"    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{AGENT_ROLE}') THEN\n"
        f"        EXECUTE format('ALTER ROLE {AGENT_ROLE} PASSWORD %L', {agent_literal});\n"
        "    END IF;\n"
        "END\n"
        "$rotate$;\n"
        "COMMIT;\n"
    )


def compose_base(env_file: Path) -> list[str]:
    return ["docker", "compose", "--env-file", str(env_file)]


def rotate(env_file: Path, run: Runner) -> list[str]:
    """Starts `postgres` if needed, then runs `rotation_sql` through `psql` over the container's
    local socket, which the official image trusts, so neither the old nor the new password is
    needed to connect. Returns the roles it updated."""
    env = read_env(env_file)
    script = rotation_sql(env)
    owner = env.get("POSTGRES_USER") or "apx"
    database = env.get("POSTGRES_DB") or "apx"
    base = compose_base(env_file)
    if run([*base, "up", "-d", "--wait", "postgres"], None) != 0:
        raise RotationError("`docker compose up -d --wait postgres` failed")
    psql = ["exec", "-T", "postgres", "psql", "-X", "-q", "-U", owner, "-d", database]
    if run([*base, *psql], script) != 0:
        raise RotationError("psql failed; no password was changed (single transaction)")
    return [owner, AGENT_ROLE]


def _subprocess_runner(argv: Sequence[str], stdin: str | None) -> int:
    completed = subprocess.run(  # fixed docker argv, no shell
        list(argv), cwd=ROOT / "infra", input=stdin, text=True, check=False
    )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args(argv)
    env_file: Path = args.env_file.resolve()
    if not env_file.exists():
        print(f"error: {env_file} not found; run `make env` first", file=sys.stderr)
        return 1
    try:
        roles = rotate(env_file, _subprocess_runner)
    except RotationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"passwords set from .env for roles: {', '.join(roles)}")
    print("next: `make up` recreates api/worker/migrate with the new credentials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
