"""UntrustedText: the only wrapper in which document-derived text may reach the agent or any
tool-enabled prompt, quoted and length-capped (PRD §6.2 I2, ADR-0003)."""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field

from apx.common.types import Frozen

UNTRUSTED_MAX_CHARS: Final[int] = 2000
"""Hard cap on characters of document text per UntrustedText (ADR-0003)."""


class UntrustedText(Frozen):
    """Document-derived text with provenance. Never interpolated raw into an instruction; it is
    rendered only through `quoted()`, which fences and escapes it as data."""

    text: Annotated[str, Field(max_length=UNTRUSTED_MAX_CHARS)]
    source: str
    """Provenance, for example `doc:<sha256>:page:2` or `scan:metadata`."""

    def quoted(self) -> str:
        """Render as an inert quoted block (fenced, fence sequences escaped, source labelled)."""
        raise NotImplementedError


def to_untrusted(text: str, source: str) -> UntrustedText:
    """Wrap arbitrary text, stripping zero-width/bidi controls and truncating to the cap."""
    raise NotImplementedError
