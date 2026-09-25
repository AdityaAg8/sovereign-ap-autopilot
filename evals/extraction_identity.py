"""Identity of an extraction eval row (M7.1 review H2): the llama-swap alias, the sha256 of the
GGUF that alias serves and a hash of the prompt contract (ADR-0012), defined in
`apx.extract.identity` so the runtime trust gate checks the same thing. Resumed runs and the
promotion gates refuse rows whose identity differs from the run they are counted in."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from apx.common.errors import ApxError
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
    "RowIdentity",
    "RowIdentityError",
    "ServedModelError",
    "prompt_version",
    "require_identity",
    "served_gguf_sha256",
    "served_paths",
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
