"""FastAPI dependencies shared by every router: an async DB session per request, the current
user decoded from a bearer JWT (no per-request database round trip, see apx.api.security), and
a capability-requiring dependency factory used by every route that changes state (PRD §10
M5.3).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from apx.api.security import JwtTokenVerifier
from apx.api.settings import ApiSettings, get_settings
from apx.api.signals import ReviewSignaller, TemporalReviewSignaller, UnconfiguredReviewSignaller
from apx.common.errors import ConfigError, PolicyDeniedError
from apx.common.master import User
from apx.db.session import get_session
from apx.ingest.ingestor import Ingestor
from apx.ingest.interfaces import IngestedDocument, WorkflowStarter
from apx.ingest.settings import IngestSettings, get_ingest_settings
from apx.ingest.store import FileDocumentStore
from apx.policy.interfaces import Capability, capabilities_for
from apx.workflows.starter import LazyTemporalClient, TemporalWorkflowStarter

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=True)

SettingsDep = Annotated[ApiSettings, Depends(get_settings)]


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """One `AsyncSession` per request, committed on success and rolled back on error (see
    `apx.db.session.get_session`)."""
    async with get_session() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_current_user(token: Annotated[str, Depends(_oauth2_scheme)], settings: SettingsDep) -> User:
    """Decodes and verifies the bearer token, returning the authenticated `User`. `display_name`
    is set to the user id: no database lookup happens on the request path, only at login."""
    try:
        claims = JwtTokenVerifier(settings).verify(token)
    except PolicyDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return User(id=claims.sub, display_name=claims.sub, roles=claims.roles)


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_capability(capability: Capability) -> Callable[[CurrentUser], User]:
    """Dependency factory: 403s unless the current user's roles grant `capability` (ADR-0008
    K1). Used as `Depends(require_capability(Capability.EDIT_FIELD))` on routes that change
    state; read routes accept any authenticated user."""

    def _check(user: CurrentUser) -> User:
        if capability not in capabilities_for(user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role does not grant the {capability.value} capability",
            )
        return user

    return _check


IngestSettingsDep = Annotated[IngestSettings, Depends(get_ingest_settings)]


def get_document_store(settings: IngestSettingsDep) -> FileDocumentStore:
    """The shared ingest path's content-addressed store (PRD §10 M5.5)."""
    return FileDocumentStore(
        settings.store_dir, max_bytes=settings.max_bytes, max_pages=settings.max_pages
    )


DocumentStoreDep = Annotated[FileDocumentStore, Depends(get_document_store)]


class _UnconfiguredWorkflowStarter:
    """`WorkflowStarter` used while `API_TEMPORAL_ADDRESS` is unset (PRD §10 M5.2). Raises
    rather than silently no-opping, per PRD §8: a misconfigured deployment must fail
    loudly, not pretend a workflow started."""

    async def start_invoice_workflow(self, document: IngestedDocument) -> str:
        raise ConfigError(
            "API_TEMPORAL_ADDRESS is not set; cannot start the invoice workflow (PRD §10 M5.2)"
        )


@lru_cache(maxsize=4)
def _temporal_client(target_host: str) -> LazyTemporalClient:
    """One reused Temporal connection per address, shared by the starter and the signaller."""
    return LazyTemporalClient(target_host)


def get_workflow_starter(settings: SettingsDep) -> WorkflowStarter:
    """The Temporal-backed starter when `API_TEMPORAL_ADDRESS` is set, else one that fails
    loudly. Overridden in tests via `app.dependency_overrides`."""
    if settings.temporal_address is None:
        return _UnconfiguredWorkflowStarter()
    return TemporalWorkflowStarter(_temporal_client(settings.temporal_address))


WorkflowStarterDep = Annotated[WorkflowStarter, Depends(get_workflow_starter)]


def get_ingestor(store: DocumentStoreDep, workflow_starter: WorkflowStarterDep) -> Ingestor:
    """The single `Ingestor.ingest` function shared by the API upload path and the `data/inbox/`
    watcher (PRD §10 M5.5)."""
    return Ingestor(store, workflow_starter)


IngestorDep = Annotated[Ingestor, Depends(get_ingestor)]


def get_review_signaller(settings: SettingsDep) -> ReviewSignaller:
    """The Temporal-backed signaller when `API_TEMPORAL_ADDRESS` is set, else one that fails
    loudly. Overridden in tests via `app.dependency_overrides`."""
    if settings.temporal_address is None:
        return UnconfiguredReviewSignaller()
    return TemporalReviewSignaller(_temporal_client(settings.temporal_address))


ReviewSignallerDep = Annotated[ReviewSignaller, Depends(get_review_signaller)]
