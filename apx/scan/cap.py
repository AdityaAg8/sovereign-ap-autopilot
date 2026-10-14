"""Caps a scan's flags per kind so an adversarial PDF (thousands of zero-width or hidden
characters) cannot inflate `ScanReport` past Temporal's payload limit, while keeping every
kind's highest-scoring flags so P4 routing is unchanged (P4)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Final

from apx.common.errors import ScanError
from apx.scan.interfaces import ScanFlag, ScanFlagKind

MAX_FLAGS_PER_KIND: Final[int] = 20
"""Flags kept per kind: plenty for a reviewer to see where the content is, and ~8 KB per kind
at typical evidence sizes."""


def cap_flags(
    flags: Sequence[ScanFlag], per_kind: int = MAX_FLAGS_PER_KIND
) -> tuple[tuple[ScanFlag, ...], dict[ScanFlagKind, int]]:
    """Keep, per kind, the `per_kind` highest-scoring flags (earliest first on ties), in their
    original order, and count the rest per kind. The maximum score of every kind survives."""
    if per_kind < 1:
        raise ScanError(f"per_kind must be at least 1, got {per_kind}")
    by_kind: defaultdict[ScanFlagKind, list[int]] = defaultdict(list)
    for position, flag in enumerate(flags):
        by_kind[flag.kind].append(position)
    keep: set[int] = set()
    suppressed: dict[ScanFlagKind, int] = {}
    for kind, positions in by_kind.items():
        ranked = sorted(positions, key=lambda i: (-flags[i].score, i))
        keep.update(ranked[:per_kind])
        if len(positions) > per_kind:
            suppressed[kind] = len(positions) - per_kind
    return tuple(flag for i, flag in enumerate(flags) if i in keep), suppressed
