"""Start-up guards: the API and worker refuse to start without a readable
learning splits file and the image ships it, and both read one approval limit that a
stale per-process setting cannot silently override. No database or network is used."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

from apx.api import settings as settings_module
from apx.api.app import create_app
from apx.common.errors import ConfigError
from apx.common.raster import MAX_IMAGE_PIXELS
from apx.learning.splits import DEFAULT_SPLITS_PATH, SplitsUnavailableError, require_splits
from apx.policy.approval_limit import DEFAULT_APPROVAL_LIMIT, load_approval_limit
from apx.workflows.main import startup_checks

REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------------------
# The splits file
# ------------------------------------------------------------------------------------------


def test_require_splits_accepts_the_committed_file() -> None:
    assert require_splits(REPO_ROOT / DEFAULT_SPLITS_PATH) == REPO_ROOT / DEFAULT_SPLITS_PATH


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"no_sets": {}})])
def test_require_splits_refuses_a_missing_or_malformed_file(
    tmp_path: Path, content: str | None
) -> None:
    path = tmp_path / "splits.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(SplitsUnavailableError, match="LEARNING_SPLITS_PATH"):
        require_splits(path)


def test_splits_unavailable_is_a_config_error() -> None:
    assert issubclass(SplitsUnavailableError, ConfigError)


def test_the_api_refuses_to_start_without_the_splits_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_JWT_SECRET", "x" * 48)
    monkeypatch.setenv("LEARNING_SPLITS_PATH", str(tmp_path / "missing.json"))
    settings_module.get_settings.cache_clear()
    try:
        with pytest.raises(SplitsUnavailableError):
            create_app()
    finally:
        settings_module.get_settings.cache_clear()


def test_the_worker_refuses_to_start_without_the_splits_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNING_SPLITS_PATH", str(tmp_path / "missing.json"))
    with pytest.raises(SplitsUnavailableError):
        startup_checks()


def test_worker_startup_checks_cap_pil_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEARNING_SPLITS_PATH", str(REPO_ROOT / DEFAULT_SPLITS_PATH))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)  # restored after
    startup_checks()
    assert Image.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS


def test_the_app_image_ships_the_splits_file() -> None:
    dockerfile = (REPO_ROOT / "infra" / "docker" / "apx" / "Dockerfile").read_text("utf-8")
    assert "COPY evals/splits.json ./evals/splits.json" in dockerfile
    ignore = (REPO_ROOT / ".dockerignore").read_text("utf-8").splitlines()
    # The build context must not drop it: `evals` is excluded except for the splits file.
    assert "evals" not in ignore
    assert "!evals/splits.json" in ignore
    assert ignore.index("!evals/splits.json") > ignore.index("evals/*")


# ------------------------------------------------------------------------------------------
# One approval limit
# ------------------------------------------------------------------------------------------


def test_default_approval_limit() -> None:
    limit = load_approval_limit({})
    assert limit.amount == DEFAULT_APPROVAL_LIMIT
    assert limit.currency == "AED"


def test_approval_limit_comes_from_the_shared_variable() -> None:
    assert load_approval_limit({"APPROVAL_LIMIT": "2500.50"}).amount == Decimal("2500.50")


@pytest.mark.parametrize("legacy", ["API_APPROVAL_LIMIT", "WORKER_APPROVAL_LIMIT"])
def test_a_legacy_limit_equal_to_the_shared_one_is_accepted(legacy: str) -> None:
    env = {"APPROVAL_LIMIT": "5000", legacy: "5000.00"}
    assert load_approval_limit(env).amount == Decimal("5000")


@pytest.mark.parametrize("legacy", ["API_APPROVAL_LIMIT", "WORKER_APPROVAL_LIMIT"])
def test_a_legacy_limit_that_drifts_is_refused(legacy: str) -> None:
    with pytest.raises(ConfigError, match=legacy):
        load_approval_limit({legacy: "20000"})


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "NaN", "Infinity"])
def test_a_malformed_limit_is_refused(raw: str) -> None:
    with pytest.raises(ConfigError):
        load_approval_limit({"APPROVAL_LIMIT": raw})


def test_api_and_worker_read_the_same_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_JWT_SECRET", "x" * 48)
    monkeypatch.setenv("APPROVAL_LIMIT", "7777.00")
    settings_module.get_settings.cache_clear()
    try:
        api_limit = settings_module.get_settings().approval_limit
    finally:
        settings_module.get_settings.cache_clear()
    assert api_limit == load_approval_limit().amount == Decimal("7777.00")


def test_api_settings_refuse_a_drifting_legacy_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_JWT_SECRET", "x" * 48)
    monkeypatch.setenv("API_APPROVAL_LIMIT", "1.00")
    settings_module.get_settings.cache_clear()
    try:
        with pytest.raises(ConfigError):
            settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()
