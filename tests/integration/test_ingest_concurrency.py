"""Integration tests for `apx.ingest.ingestor.Ingestor` against a real, throwaway Postgres
container: the sha256 unique constraint on `ingested_documents` is the atomic
gate proving the acceptance criterion verbatim -- the same file ingested twice, whether through
the watcher, the upload path, or both, including concurrently, starts exactly one
`InvoiceWorkflow`. Also covers the `WorkflowStartError` rollback path: a failed workflow start
frees the sha256 for a later retry rather than wedging it as a permanent phantom duplicate.
Marked `integration`, so it needs Docker (`pytest -m integration`)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import select

import apx.db.session as db_session
from apx.api.app import create_app
from apx.api.deps import get_workflow_starter
from apx.api.security import hash_password
from apx.api.settings import get_settings
from apx.db.models import IngestedDocument as IngestedDocumentRow
from apx.db.models import User
from apx.db.session import get_session
from apx.ingest.errors import WorkflowStartError
from apx.ingest.ingestor import Ingestor
from apx.ingest.interfaces import IngestedDocument
from apx.ingest.settings import get_ingest_settings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-ingest"
HOST_PORT = 55435
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

PDF_BYTES = b"%PDF-1.4 fake invoice bytes for ingest concurrency tests"


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    return cfg


async def _wait_for_postgres(dsn: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return
    raise TimeoutError(f"Postgres did not become ready within {timeout_s}s") from last_error


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """Starts a throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port (same
    pattern as `tests/integration/test_api_authz.py`), waits for it, and stops it after every
    test in this module has run."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{HOST_PORT}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        yield DATABASE_URL
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
def ingest_schema(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Points `apx.db.session` at the throwaway container, migrates to `head`, and forces fresh
    caches so every test starts from an identical, empty schema on a connection bound to its own
    event loop."""
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    db_session.get_engine.cache_clear()
    db_session._session_factory.cache_clear()
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        yield postgres_url
    finally:
        command.downgrade(cfg, "base")
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()


@pytest.fixture
def api_schema(
    ingest_schema: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[str]:
    """Extends `ingest_schema` with the API/ingest settings the FastAPI app needs (same pattern
    as `tests/integration/test_api_authz.py::api_schema`)."""
    monkeypatch.setenv("API_JWT_SECRET", "integration-test-secret-value-long-enough-for-hs256")
    monkeypatch.setenv("INGEST_STORE_DIR", str(tmp_path / "documents"))
    get_settings.cache_clear()
    get_ingest_settings.cache_clear()
    try:
        yield ingest_schema
    finally:
        get_settings.cache_clear()
        get_ingest_settings.cache_clear()


def _make_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 700, "hello")
    c.showPage()
    c.save()
    return buf.getvalue()


async def _create_uploader_user(*, user_id: str, password: str) -> None:
    """Seeds one active user with no roles: `POST /uploads` only requires authentication, not
    a specific capability (any authenticated session may upload)."""
    async with get_session() as session:
        session.add(
            User(
                id=user_id,
                email=f"{user_id}@example.test",
                display_name=user_id,
                password_hash=hash_password(password),
            )
        )


async def _login(client: AsyncClient, user_id: str, password: str) -> str:
    response = await client.post(
        "/auth/login", data={"username": f"{user_id}@example.test", "password": password}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["access_token"])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeStore:
    """`DocumentStore` fake: validation and disk storage are `apx.ingest.store`'s job (covered
    by `tests/unit/test_ingest_store.py`); these tests are only about the DB-level dedupe gate,
    so `put` just wraps the bytes into an `IngestedDocument` keyed by a real sha256."""

    def put(
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestedDocument:
        return IngestedDocument(
            sha256=_sha256(data),
            filename=filename,
            media_type="application/pdf",
            size_bytes=len(data),
            received_at=datetime.now(UTC),
            source=source,
        )

    def get(self, sha256: str) -> bytes:  # pragma: no cover -- unused by these tests
        raise NotImplementedError

    def exists(self, sha256: str) -> bool:  # pragma: no cover -- unused by these tests
        return False


class _CountingWorkflowStarter:
    """Records every call and returns a deterministic id derived from the sha256 -- the
    idempotency key M5.5 requires (`f"invoice-{document.sha256}"`)."""

    def __init__(self) -> None:
        self.started: list[IngestedDocument] = []

    async def start_invoice_workflow(self, document: IngestedDocument) -> str:
        self.started.append(document)
        return f"invoice-{document.sha256}"


class _FailingWorkflowStarter:
    def __init__(self) -> None:
        self.attempts = 0

    async def start_invoice_workflow(self, document: IngestedDocument) -> str:
        self.attempts += 1
        raise RuntimeError("temporal is down")


async def _count_rows(sha256: str) -> int:
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(IngestedDocumentRow).where(IngestedDocumentRow.sha256 == sha256)
                )
            )
            .scalars()
            .all()
        )
        return len(rows)


# ------------------------------------------------------------------------------------------
# Sequential: upload then watch, and watch then upload
# ------------------------------------------------------------------------------------------


async def test_same_bytes_via_upload_then_watch_starts_one_workflow(ingest_schema: str) -> None:
    starter = _CountingWorkflowStarter()
    ingestor = Ingestor(_FakeStore(), starter)

    first = await ingestor.ingest(PDF_BYTES, "a.pdf", "upload")
    second = await ingestor.ingest(PDF_BYTES, "b.pdf", "watch")

    assert first.duplicate is False
    assert first.workflow_id == f"invoice-{first.document.sha256}"
    assert second.duplicate is True
    assert second.workflow_id is None
    assert len(starter.started) == 1
    assert starter.started[0].sha256 == first.document.sha256
    assert await _count_rows(first.document.sha256) == 1


async def test_same_bytes_via_watch_then_upload_starts_one_workflow(ingest_schema: str) -> None:
    starter = _CountingWorkflowStarter()
    ingestor = Ingestor(_FakeStore(), starter)

    first = await ingestor.ingest(PDF_BYTES, "a.pdf", "watch")
    second = await ingestor.ingest(PDF_BYTES, "b.pdf", "upload")

    assert first.duplicate is False
    assert second.duplicate is True
    assert len(starter.started) == 1


async def test_different_bytes_each_start_their_own_workflow(ingest_schema: str) -> None:
    starter = _CountingWorkflowStarter()
    ingestor = Ingestor(_FakeStore(), starter)

    first = await ingestor.ingest(PDF_BYTES, "a.pdf", "upload")
    second = await ingestor.ingest(PDF_BYTES + b" different", "b.pdf", "upload")

    assert first.duplicate is False
    assert second.duplicate is False
    assert first.document.sha256 != second.document.sha256
    assert len(starter.started) == 2


# ------------------------------------------------------------------------------------------
# Concurrent: the acceptance criterion verbatim
# ------------------------------------------------------------------------------------------


async def test_concurrent_ingest_of_the_same_bytes_starts_exactly_one_workflow(
    ingest_schema: str,
) -> None:
    starter = _CountingWorkflowStarter()
    ingestor = Ingestor(_FakeStore(), starter)

    sources: list[Literal["watch", "upload"]] = ["upload", "watch", "upload", "watch", "upload"]
    results = await asyncio.gather(
        *(ingestor.ingest(PDF_BYTES, f"file-{i}.pdf", source) for i, source in enumerate(sources))
    )

    duplicates = [r.duplicate for r in results]
    assert duplicates.count(False) == 1, "exactly one caller must win the race"
    assert duplicates.count(True) == len(sources) - 1
    assert len(starter.started) == 1
    sha256 = results[0].document.sha256
    assert await _count_rows(sha256) == 1


# ------------------------------------------------------------------------------------------
# WorkflowStartError rolls back the dedupe row so a retry is possible
# ------------------------------------------------------------------------------------------


async def test_a_failed_workflow_start_frees_the_sha256_for_retry(ingest_schema: str) -> None:
    failing_starter = _FailingWorkflowStarter()
    ingestor = Ingestor(_FakeStore(), failing_starter)

    with pytest.raises(WorkflowStartError):
        await ingestor.ingest(PDF_BYTES, "a.pdf", "upload")

    assert failing_starter.attempts == 1
    assert await _count_rows(_sha256(PDF_BYTES)) == 0

    working_starter = _CountingWorkflowStarter()
    retry_ingestor = Ingestor(_FakeStore(), working_starter)
    result = await retry_ingestor.ingest(PDF_BYTES, "a.pdf", "upload")

    assert result.duplicate is False
    assert len(working_starter.started) == 1


# ------------------------------------------------------------------------------------------
# The real `POST /uploads` route, wired through `apx.api.deps.get_ingestor`
# ------------------------------------------------------------------------------------------


async def test_upload_route_starts_one_workflow_for_the_same_file_uploaded_twice(
    api_schema: str,
) -> None:
    await _create_uploader_user(user_id="uploader1", password="pw")
    starter = _CountingWorkflowStarter()
    app = create_app()
    app.dependency_overrides[get_workflow_starter] = lambda: starter
    data = _make_pdf()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "uploader1", "pw")
        headers = {"Authorization": f"Bearer {token}"}

        first = await client.post(
            "/uploads", files={"file": ("a.pdf", data, "application/pdf")}, headers=headers
        )
        second = await client.post(
            "/uploads", files={"file": ("b.pdf", data, "application/pdf")}, headers=headers
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body["duplicate"] is False
    assert first_body["workflow_id"] is not None
    assert second_body["duplicate"] is True
    assert second_body["workflow_id"] is None
    assert first_body["sha256"] == second_body["sha256"]
    assert len(starter.started) == 1
