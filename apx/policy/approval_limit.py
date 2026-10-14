"""The single source of the P3 approval limit: the worker's routing and the API's authorization
both read `APPROVAL_LIMIT` through `load_approval_limit`, so the two can never disagree. The
retired per-process `API_APPROVAL_LIMIT`/`WORKER_APPROVAL_LIMIT` are still honoured only when they
equal it; a different value is a `ConfigError`, never a silent drift."""

from __future__ import annotations

import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Final

from apx.common.errors import ConfigError
from apx.common.types import Money

APPROVAL_LIMIT_ENV: Final[str] = "APPROVAL_LIMIT"
DEFAULT_APPROVAL_LIMIT: Final[Decimal] = Decimal("10000.00")
APPROVAL_LIMIT_CURRENCY: Final[str] = "AED"
LEGACY_APPROVAL_LIMIT_ENVS: Final[tuple[str, ...]] = ("API_APPROVAL_LIMIT", "WORKER_APPROVAL_LIMIT")


def _parse(name: str, raw: str) -> Decimal:
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ConfigError(f"{name}={raw!r} is not a decimal amount") from exc
    if not value.is_finite() or value <= 0:
        raise ConfigError(f"{name}={raw!r} must be a positive amount")
    return value


def load_approval_limit(environ: Mapping[str, str] | None = None) -> Money:
    """The approval limit in AED: `APPROVAL_LIMIT`, else `DEFAULT_APPROVAL_LIMIT`. Raises
    `ConfigError` for a malformed value or a legacy variable that disagrees with it."""
    env = os.environ if environ is None else environ
    raw = env.get(APPROVAL_LIMIT_ENV)
    limit = _parse(APPROVAL_LIMIT_ENV, raw) if raw else DEFAULT_APPROVAL_LIMIT
    for legacy in LEGACY_APPROVAL_LIMIT_ENVS:
        legacy_raw = env.get(legacy)
        if legacy_raw and _parse(legacy, legacy_raw) != limit:
            raise ConfigError(
                f"{legacy}={legacy_raw} differs from the approval limit {limit}; set only "
                f"{APPROVAL_LIMIT_ENV}, which both the API and the worker read"
            )
    return Money(amount=limit, currency=APPROVAL_LIMIT_CURRENCY)
