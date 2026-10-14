"""Settings for the parse service HTTP client.
Every field defaults to the value that matches infra/compose.yaml's `parse` service on the
internal `app` network, except the service token, which `make env` generates into `.env`; set the
`PARSE_` prefixed environment variables below to override in another deployment."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ParseClientConfig(BaseSettings):
    """Connection settings for `apx.parse.client.ParseServiceClient`."""

    model_config = SettingsConfigDict(env_prefix="PARSE_", extra="forbid")

    base_url: str = Field(
        default="http://parse:8000",
        description=(
            "Parse service base URL. Reachable only on the internal `app` network; "
            "never a host or public address."
        ),
    )
    connect_timeout_s: float = Field(
        default=5.0,
        gt=0.0,
        description="TCP connect timeout for a parse service request.",
    )
    request_timeout_s: float = Field(
        default=240.0,
        gt=0.0,
        description=(
            "Total per-page request timeout, covering the service's model inference time. "
            "Measured (ADR-0013): a cold page (lazy model load ~12 s) 24.7 s, warm pages "
            "13-15 s at 1024 max new tokens; the worker sends one page at a time. At the "
            "amendment-1 budget of 3072 tokens a dense page that finishes without looping took "
            "125 s (~21 tok/s, 2026-09-27 parse-cache run), so the unbounded first pass can reach "
            "~150 s plus the ~12 s cold load; strips only start inside the service's 90 s page "
            "deadline. 240 s covers that worst case with margin."
        ),
    )
    service_token: SecretStr | None = Field(
        default=None,
        description=(
            "Shared secret sent as `Authorization: Bearer <token>` on every call; "
            "the service refuses /parse and /unload without it. Generated into `.env` as "
            "PARSE_SERVICE_TOKEN by `make env`."
        ),
    )
