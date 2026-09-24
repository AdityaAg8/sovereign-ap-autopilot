"""Unit tests for `apx.scan.cap.cap_flags`: at most N flags per kind, each kind's highest scores
always kept, original order preserved and the rest counted per kind (PRD §6.3 P4, M6.2)."""

from __future__ import annotations

import pytest

from apx.common.errors import ScanError
from apx.scan.cap import cap_flags
from apx.scan.interfaces import ScanFlag, ScanFlagKind
from apx.scan.text import build_evidence


def _flag(kind: ScanFlagKind, score: float, page: int = 0) -> ScanFlag:
    return ScanFlag(kind=kind, page=page, score=score, evidence=build_evidence(kind, f"p{page}"))


def test_under_the_cap_nothing_changes() -> None:
    flags = [_flag(ScanFlagKind.ZERO_WIDTH, 1.0), _flag(ScanFlagKind.HIDDEN_TEXT, 0.4)]

    assert cap_flags(flags, per_kind=2) == (tuple(flags), {})


def test_caps_each_kind_separately_and_counts_the_rest() -> None:
    zero_width = [_flag(ScanFlagKind.ZERO_WIDTH, 1.0, page=i) for i in range(10)]
    hidden = [_flag(ScanFlagKind.HIDDEN_TEXT, 1.0, page=i) for i in range(2)]

    kept, suppressed = cap_flags([*zero_width, *hidden], per_kind=3)

    assert kept == (*zero_width[:3], *hidden)
    assert suppressed == {ScanFlagKind.ZERO_WIDTH: 7}


def test_a_late_higher_score_is_kept_in_its_original_position() -> None:
    flags = [_flag(ScanFlagKind.INVISIBLE_TEXT, 0.3, page=i) for i in range(5)]
    flags.append(_flag(ScanFlagKind.INVISIBLE_TEXT, 0.95, page=5))

    kept, suppressed = cap_flags(flags, per_kind=2)

    assert kept == (flags[0], flags[5])
    assert max(f.score for f in kept) == 0.95
    assert suppressed == {ScanFlagKind.INVISIBLE_TEXT: 4}


def test_non_positive_cap_is_a_typed_error() -> None:
    with pytest.raises(ScanError):
        cap_flags([], per_kind=0)
