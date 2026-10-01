"""Set the `apx_agent_ro` role's password from the `AGENT_RO_PASSWORD` env var instead of the
literal migration 0004 used to hardcode (REVIEW L-01). `make env` generates the value into
`.env` and compose hands it to `migrate`; without it the role's password is cleared (it cannot
log in) rather than left at the old public default. Later changes go through
`make rotate-db-password`.

Revision ID: 0013_agent_ro_password
Revises: 0012_approval_digest
Create Date: 2026-09-27
"""

from __future__ import annotations

import logging
import os
import re

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013_agent_ro_password"
down_revision: str | None = "0012_approval_digest"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

ENV_VAR = "AGENT_RO_PASSWORD"
PASSWORD_PATTERN = re.compile(r"[A-Za-z0-9_-]{16,256}")
_log = logging.getLogger("alembic.runtime.migration")


def password_statement(password: str | None) -> str:
    """The `ALTER ROLE` statement for `password`; None or empty clears the password.

    `ALTER ROLE ... PASSWORD` takes no bind parameters, so the secret is spliced into the SQL.
    It is therefore restricted to `PASSWORD_PATTERN` (what `make env` generates), which leaves
    nothing to quote and nothing SQLAlchemy or the driver would read as a placeholder (`:`,
    `%`). Raises ValueError otherwise; the message never contains the value."""
    if not password:
        return "ALTER ROLE apx_agent_ro PASSWORD NULL"
    if not PASSWORD_PATTERN.fullmatch(password):
        raise ValueError(
            f"{ENV_VAR} must be 16-256 characters of A-Z, a-z, 0-9, '-' or '_' "
            "(regenerate it with `make env`)"
        )
    return f"ALTER ROLE apx_agent_ro PASSWORD '{password}'"


def upgrade() -> None:
    password = os.environ.get(ENV_VAR)
    if not password:
        _log.warning(
            "%s is not set: clearing apx_agent_ro's password, so the agent cannot log in until "
            "`make env` + `make rotate-db-password` set one",
            ENV_VAR,
        )
    op.execute(password_statement(password))


def downgrade() -> None:
    # Nothing to restore: going back must not reinstate the old public default. The role keeps
    # whatever password it has; 0004's downgrade drops the role altogether.
    pass
