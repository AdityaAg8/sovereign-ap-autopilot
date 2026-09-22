"""Root pytest fixtures for Sovereign AP Autopilot (PRD §8): blocks every real outbound HTTP
call by default so a unit test can never silently grow a network dependency, exempting only
tests marked gpu/integration/artifacts/external, which declare the access they need up front.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

_NETWORK_MARKERS = ("gpu", "integration", "artifacts", "external")


class NetworkDisabledError(RuntimeError):
    """Raised when a test not marked gpu/integration/artifacts/external attempts a real
    outbound HTTP request; unit tests must never touch the network (PRD §8)."""


def _blocked_handle_request(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
    raise NetworkDisabledError(
        f"unit test attempted a real network call to {request.url}; mark the test "
        "gpu/integration/artifacts/external if it genuinely needs network access"
    )


async def _blocked_handle_async_request(
    self: httpx.AsyncHTTPTransport, request: httpx.Request
) -> httpx.Response:
    raise NetworkDisabledError(
        f"unit test attempted a real network call to {request.url}; mark the test "
        "gpu/integration/artifacts/external if it genuinely needs network access"
    )


@pytest.fixture(autouse=True)
def _block_network_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if any(request.node.get_closest_marker(name) for name in _NETWORK_MARKERS):
        yield
        return
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked_handle_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _blocked_handle_async_request
    )
    yield
