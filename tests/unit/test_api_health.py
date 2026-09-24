"""The API's unauthenticated liveness probe (PRD §10 M5.6): the Compose healthcheck for the
`api` service calls it, so it must answer without a token and without touching the database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apx.api import settings as settings_module
from apx.api.app import create_app


def test_health_answers_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_JWT_SECRET", "x" * 48)
    settings_module.get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            response = client.get("/health")
    finally:
        settings_module.get_settings.cache_clear()
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
