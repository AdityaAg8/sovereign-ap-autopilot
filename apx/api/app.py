"""FastAPI application factory: wires the auth/invoices/proposals/uploads/vendors routers, CORS
locked to the single web origin, security response headers and the typed-error exception
handlers (PRD §10 M5.3, M5.4, docs/security.md B8/B9). Typed OpenAPI comes free from the pydantic
models on every route.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from apx.api.errors import install_exception_handlers
from apx.api.routers import auth, chat, invoices, ops, proposals, uploads, vendors
from apx.api.settings import get_settings

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'",
    "Cache-Control": "no-store",
}


async def _security_headers_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


def create_app() -> FastAPI:
    """Builds the API app. Settings are only read here, inside the factory, so importing this
    module never fails on a missing `API_JWT_SECRET`; only calling `create_app()` (or a route's
    own `Depends(get_settings)`) does."""
    app = FastAPI(
        title="Sovereign AP Autopilot API",
        description="JWT-authenticated REST + SSE API for invoice review and approval.",
        version="0.1.0",
    )

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(_security_headers_middleware)

    install_exception_handlers(app)

    app.include_router(auth.router)
    app.include_router(invoices.router)
    app.include_router(proposals.router)
    app.include_router(uploads.router)
    app.include_router(vendors.router)
    app.include_router(ops.router)
    app.include_router(chat.router)

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Unauthenticated liveness probe for the Compose healthcheck; touches no data."""
        return {"status": "ok"}

    return app
