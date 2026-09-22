"""Builds scan-flag evidence: a short, escaped excerpt of attacker-controlled document text
wrapped as `UntrustedText`. Escaping is implemented independently of
`apx.common.untrusted.to_untrusted`, which strips suspicious codepoints outright: scan evidence
must instead keep the exact codepoint visible, as a `\\uXXXX` escape, so a human reviewer can see
precisely what was hidden in the document, while still being safe to print or log (PRD §6.2 I2,
ADR-0008, M6.2)."""

from __future__ import annotations

from apx.common.untrusted import UntrustedText
from apx.scan.interfaces import ScanFlagKind

EXCERPT_MAX_CHARS = 200
"""Evidence excerpts are short pointers for a reviewer, not a full document dump."""

_ALWAYS_ESCAPE = frozenset(range(0x00, 0x09)) | frozenset(range(0x0B, 0x20)) | frozenset({0x7F})
"""C0 controls other than tab/newline; these are never meaningful in extracted PDF text."""


def escape_excerpt(text: str, *, limit: int = EXCERPT_MAX_CHARS) -> str:
    """Replace zero-width, bidi and C0/C1 control characters with visible `\\uXXXX` escapes and
    truncate to `limit` characters, so the excerpt is always safe to print or log."""
    from apx.scan.controls import SUSPICIOUS_CODEPOINTS  # local import: avoids a cycle

    out: list[str] = []
    for ch in text[:limit]:
        cp = ord(ch)
        if cp in SUSPICIOUS_CODEPOINTS or cp in _ALWAYS_ESCAPE:
            out.append(f"\\u{cp:04x}")
        else:
            out.append(ch)
    escaped = "".join(out)
    if len(text) > limit:
        escaped += "..."
    return escaped


def build_evidence(kind: ScanFlagKind, excerpt: str) -> UntrustedText:
    """Wrap an escaped excerpt as `UntrustedText` with provenance `scan:<kind>`."""
    return UntrustedText(text=escape_excerpt(excerpt), source=f"scan:{kind.value}", truncated=False)
