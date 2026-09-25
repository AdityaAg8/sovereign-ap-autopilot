"""The fixed M3.4 bake-off document subset (PRD §10 M3.4): the first 200 UAE test sha256s in sorted
order, i.e. a hash-random, deterministic sample of evals/splits.json. The M7.1 P2 latency gate
(`ml.extractor_b.gates.latency_subset`) uses this same subset, so both runs time the same work."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from apx.common.errors import ApxError

BAKEOFF_SUBSET_SIZE: Final[int] = 200


class SubsetError(ApxError):
    """The UAE test split is too small to draw the fixed bake-off subset from."""


def uae_subset(test_ids: Sequence[str], size: int = BAKEOFF_SUBSET_SIZE) -> list[str]:
    """The first `size` distinct UAE test sha256s in sorted order; never fewer."""
    ordered = sorted(set(test_ids))
    if len(ordered) < size:
        raise SubsetError(f"UAE test has {len(ordered)} documents, the bake-off needs {size}")
    return ordered[:size]
