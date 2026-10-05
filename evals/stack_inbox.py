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

    Raises `ConfigError` unless the value is a relative POSIX path below `data/`: the worker
    shares only `data/` with the host (`../data:/app/data`), so any other folder is one the
    worker watches inside its container and the host can never write to.
    """
    if explicit is not None:
        return explicit
    value = environ.get("INGEST_INBOX_DIR", "").strip()
    if not value:
        return data / "inbox"
    relative = PurePosixPath(value)
    parts = relative.parts
    if (
        relative.is_absolute()
        or any(c in value for c in ("\\", ":"))  # Windows separators and drive letters
        or ".." in parts
        or parts[:1] != ("data",)
    ):
        raise ConfigError(
            f"INGEST_INBOX_DIR={value!r} must be a relative path below data/, such as "
            "data/inbox_attacks (the only folder the worker shares with the host)"
        )
    if len(parts) < 2:
        raise ConfigError(f"INGEST_INBOX_DIR={value!r} must name a folder below data/")
    return root.joinpath(*relative.parts)
