"""Identity of an extraction eval row (M7.1 review H2): the llama-swap alias, the sha256 of the
GGUF that alias serves and a hash of the prompt contract (ADR-0012), defined in
`apx.extract.identity` so the runtime trust gate checks the same thing; eval runs also stamp a
serving identity (the alias's llama-swap args, the parser build). Resumed runs and the promotion
gates refuse rows whose identity differs from the run they are counted in."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from apx.common.errors import ApxError
from apx.common.types import Frozen
from apx.extract.identity import (
    EXPORT_MANIFEST_NAME,
    IDENTITY_KEYS,
    ExtractorIdentity,
    ServedModelError,
    prompt_version,
    served_gguf_sha256,
    served_paths,
)

__all__ = [
    "EXPORT_MANIFEST_NAME",
    "IDENTITY_KEYS",
    "SERVING_KEYS",
    "RowIdentity",
    "RowIdentityError",
    "ServedModelError",
    "ServingIdentity",
    "ServingIdentityError",
    "prompt_version",
    "require_identity",
    "require_serving",
    "served_gguf_sha256",
    "served_paths",
    "serving_args_sha256",
]

RowIdentity = ExtractorIdentity
"""The identity stamped on every eval row (alias of `apx.extract.identity.ExtractorIdentity`)."""


class RowIdentityError(ApxError):
    """Rows in an eval file were produced by another model build or prompt contract."""


def require_identity(
    rows: Mapping[str, Mapping[str, Any]], expected: RowIdentity, path: Path
) -> None:
    """Every row must carry exactly `expected`; rows without identity fields (written before
    they existed) are foreign too."""
    want = expected.as_row_fields()
    foreign = [sha for sha, row in rows.items() if {k: row.get(k) for k in IDENTITY_KEYS} != want]
    if foreign:
        first = rows[foreign[0]]
        found = {k: first.get(k) for k in IDENTITY_KEYS}
        raise RowIdentityError(
            f"{path}: {len(foreign)} of {len(rows)} rows are from another run (e.g. "
            f"{foreign[0][:12]}: {found}, expected {want}); use a new --rows file or delete it"
        )


SERVING_KEYS: Final[tuple[str, ...]] = ("serving_args_sha256", "parser_identity")
_MACRO: Final = re.compile(r"\$\{([^}]+)\}")


class ServingIdentityError(RowIdentityError):
    """Rows in an eval file were produced under other serving args or another parser build."""


class ServingIdentity(Frozen):
    """How a run was served, beyond the build: a hash of the alias's llama-swap entry (context,
    slots, cache and sampling flags; ADR-0013 amdt 1 found `--parallel 2` corrupts Extractor A)
    and the parser identity (`evals.parse_cache.parser_identity`)."""

    serving_args_sha256: str
    parser_identity: str

    def as_row_fields(self) -> dict[str, str]:
        return {
            "serving_args_sha256": self.serving_args_sha256,
            "parser_identity": self.parser_identity,
        }


def serving_args_sha256(alias: str, *, llama_swap_config: Path) -> str:
    """16-hex hash of how llama-swap serves `alias`: its `cmd` with macros expanded (except
    `${PORT}`, which llama-swap assigns per start) and tokenised, so whitespace and line breaks
    do not count, plus its `env`. Every failure is a ServedModelError."""
    try:
        config = yaml.safe_load(llama_swap_config.read_text(encoding="utf-8"))
        entry = config["models"][alias]
        macros = {str(k): str(v) for k, v in (config.get("macros") or {}).items()}
        cmd = str(entry["cmd"])
        env = sorted(str(item) for item in entry.get("env") or [])
    except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError) as exc:
        raise ServedModelError(f"cannot read `{alias}` from {llama_swap_config}: {exc}") from exc
    for _ in range(len(macros) + 1):
        expanded = _MACRO.sub(lambda m: macros.get(m.group(1), m.group(0)), cmd)
        if expanded == cmd:
            break
        cmd = expanded
    else:
        raise ServedModelError(f"`{alias}` in {llama_swap_config} has recursive macros")
    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        raise ServedModelError(
            f"`{alias}` cmd in {llama_swap_config} does not parse: {exc}"
        ) from exc
    body = json.dumps({"cmd": tokens, "env": env}, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def require_serving(
    rows: Mapping[str, Mapping[str, Any]], expected: ServingIdentity, path: Path
) -> None:
    """Every row must carry exactly `expected`; rows without the serving fields (written before
    they existed, on an unknown server config) are foreign too."""
    want = expected.as_row_fields()
    foreign = [sha for sha, row in rows.items() if {k: row.get(k) for k in SERVING_KEYS} != want]
    if foreign:
        found = {k: rows[foreign[0]].get(k) for k in SERVING_KEYS}
        raise ServingIdentityError(
            f"{path}: {len(foreign)} of {len(rows)} rows were served or parsed differently (e.g. "
            f"{foreign[0][:12]}: {found}, expected {want}); archive the file or use a new --rows"
        )
