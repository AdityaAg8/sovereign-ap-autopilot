"""The running stack's inbox as a host path, for the scripts that drop documents into it (e2e
pipeline, attack suite, demo). The worker watches `INGEST_INBOX_DIR` (compose passthrough,
resolved against /app, the repo root inside the container), so a run on a second database drops
its files where that run's worker looks, never into the live inbox."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from apx.common.errors import ConfigError


def host_inbox(
    data: Path,
    root: Path,
    explicit: Path | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
) -> Path:
    """`explicit` if given; else `root/INGEST_INBOX_DIR` when that is set; else `data/inbox`.

    Raises `ConfigError` for an absolute or `..` value, which has no fixed place under `root`.
    """
    if explicit is not None:
        return explicit
    value = environ.get("INGEST_INBOX_DIR", "").strip()
    if not value:
        return data / "inbox"
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(
            f"INGEST_INBOX_DIR={value!r} must be a path inside the repo, such as "
            "data/inbox_attacks (the worker resolves it against /app, the repo root)"
        )
    return root.joinpath(*relative.parts)
