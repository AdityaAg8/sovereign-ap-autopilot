"""Unit tests for `apx.db.session`: `get_database_url`'s default/validation logic and
`get_session`'s commit/rollback/close contract, exercised against a fake `AsyncSession` so no
database connection is ever opened (PRD §6.4/§8, §10 M5.1). The real Postgres round trip lives
in `tests/integration/test_migrations.py`.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import apx.db.session as session_module
from apx.common.errors import ConfigError


class _FakeAsyncSession:
    """Stand-in for `sqlalchemy.ext.asyncio.AsyncSession` that records which of
    commit/rollback/close were awaited, without touching a real engine or database."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


def _patch_session_factory(
    monkeypatch: pytest.MonkeyPatch, fake_session: _FakeAsyncSession
) -> None:
    """Replaces the module-level `_session_factory` with one that always hands back
    `fake_session`, bypassing `get_engine()`/`create_async_engine` entirely."""

    def factory() -> Callable[[], _FakeAsyncSession]:
        return lambda: fake_session

    monkeypatch.setattr(session_module, "_session_factory", factory)


def test_get_database_url_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert session_module.get_database_url() == session_module.DEFAULT_DATABASE_URL


def test_get_database_url_honours_a_valid_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@otherhost:5432/otherdb")
    assert session_module.get_database_url() == "postgresql+asyncpg://u:p@otherhost:5432/otherdb"


def test_get_database_url_rejects_a_non_asyncpg_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/apx")
    with pytest.raises(ConfigError, match="asyncpg"):
        session_module.get_database_url()


async def test_get_session_commits_and_closes_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeAsyncSession()
    _patch_session_factory(monkeypatch, fake_session)

    async with session_module.get_session() as session:
        assert session is fake_session
        assert not fake_session.committed

    assert fake_session.committed
    assert not fake_session.rolled_back
    assert fake_session.closed


async def test_get_session_rolls_back_reraises_and_still_closes_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeAsyncSession()
    _patch_session_factory(monkeypatch, fake_session)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with session_module.get_session() as session:
            assert session is fake_session
            raise _Boom("failure inside the session block")

    assert fake_session.rolled_back
    assert not fake_session.committed
    assert fake_session.closed
