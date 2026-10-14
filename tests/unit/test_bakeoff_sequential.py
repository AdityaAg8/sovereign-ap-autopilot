"""Unit tests for the bake-off futility rule (`evals.bakeoff.sequential`, ADR-0016 amendment
1): look sizes and O'Brien-Fleming bounds, each stopping criterion, determinism, per-document
counts from real rows, and a simulated property that an arm whose true gain meets the win
criteria is stopped no more often than the stated bound."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from evals.bakeoff.sequential import (
    FALSE_STOP_BOUND,
    OBF_C,
    Counts,
    doc_counts,
    evaluate_look,
    futility,
    look_sizes,
    look_z,
    max_invalid,
    micro_f1,
)
from evals.extraction_f1 import score_rows
from tests.unit._bakeoff_fakes import write_uae_doc

FIELDS = 14


def _counts(correct: np.ndarray) -> Counts:
    """Every field predicted; a wrong value is one false positive and one false negative."""
    wrong = FIELDS - correct
    return np.stack([correct, wrong, wrong], axis=1).astype(np.int64)


def _simulate(
    rng: np.random.Generator, delta: float, ratio: float, total: int = 200
) -> tuple[Counts, Counts, np.ndarray, np.ndarray]:
    """Paired documents: a per-document difficulty shared by both arms, the challenger's field
    accuracy `delta` points higher (so its true pooled gain is `delta`), and latencies sharing a
    per-document cost with a true median ratio of `ratio`."""
    base_p = rng.uniform(0.6, 0.95, total)
    base = _counts(rng.binomial(FIELDS, base_p))
    arm = _counts(rng.binomial(FIELDS, np.clip(base_p + delta / 100, 0, 1)))
    cost = rng.lognormal(3.0, 0.4, total)
    base_lat = cost * rng.lognormal(0.0, 0.2, total)
    arm_lat = cost * ratio * rng.lognormal(0.0, 0.2, total)
    return base, arm, base_lat, arm_lat


def _stops(
    base: Counts, arm: Counts, base_lat: np.ndarray, arm_lat: np.ndarray, replicates: int
) -> int | None:
    total = len(base)
    for look, n in enumerate(look_sizes(total), start=1):
        result = evaluate_look(
            arm_key="sim",
            look=look,
            total=total,
            base_counts=base[:n],
            arm_counts=arm[:n],
            base_latency=base_lat[:n],
            arm_latency=arm_lat[:n],
            arm_invalid=0,
            replicates=replicates,
        )
        if result.stop:
            return look
    return None


def test_looks_and_bounds_are_pre_registered() -> None:
    assert look_sizes(200) == (50, 100, 150)
    assert look_sizes(8) == (2, 4, 6) and look_sizes(3) == (1, 2)
    zs = [look_z(n, 200) for n in look_sizes(200)]
    assert [round(z, 2) for z in zs] == [4.05, 2.86, 2.34]
    assert look_z(200, 200) == OBF_C
    per_look = [0.5 * math.erfc(z / math.sqrt(2)) for z in zs]  # Phi(-z_k)
    assert 2 * sum(per_look) <= FALSE_STOP_BOUND  # F1 and latency criteria together
    assert max_invalid(200) == 2 and max_invalid(100) == 1


def test_micro_f1_matches_the_pooled_definition() -> None:
    counts = np.array([[3, 1, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int64)
    assert float(micro_f1(counts)) == 2 * 3 / (2 * 3 + 1 + 2)
    assert float(micro_f1(np.zeros((2, 3), dtype=np.int64))) == 1.0


def _look(**changes: Any) -> Any:
    rng = np.random.default_rng(7)
    base, arm, base_lat, arm_lat = _simulate(rng, 0.0, 1.0, total=200)
    kwargs: dict[str, Any] = {
        "arm_key": "x",
        "look": 1,
        "total": 200,
        "base_counts": base[:50],
        "arm_counts": arm[:50],
        "base_latency": base_lat[:50],
        "arm_latency": arm_lat[:50],
        "arm_invalid": 0,
    }
    kwargs.update(changes)
    return evaluate_look(**kwargs)


def test_each_criterion_stops_on_its_own_and_names_itself() -> None:
    fine = _look()
    assert not fine.stop and fine.n == 50 and fine.ratio is not None
    worse = _look(arm_counts=_counts(np.zeros(50, dtype=np.int64)))
    assert worse.stop and "ΔF1 upper bound" in worse.reasons[0]
    slow = _look(arm_latency=_slow())
    assert slow.stop and any("p50 ratio lower bound" in r for r in slow.reasons)
    invalid = _look(arm_invalid=3)
    assert invalid.reasons == ("3 invalid-JSON rows > 2 allowed in 200",)
    assert not _look(arm_invalid=2).stop


def _slow() -> np.ndarray:
    rng = np.random.default_rng(7)
    _, _, base_lat, _ = _simulate(rng, 0.0, 1.0, total=200)
    return np.asarray(base_lat[:50] * 4.0)


def test_rows_without_latency_never_stop_on_latency() -> None:
    result = _look(arm_latency=np.full(50, np.nan))
    assert result.ratio is None and result.ratio_lower is None and not result.stop


def test_the_same_rows_always_give_the_same_look() -> None:
    assert _look() == _look()
    assert _look(arm_key="y").delta_se != _look().delta_se  # seeded per arm and look


def test_clearly_losing_arms_stop_early() -> None:
    rng = np.random.default_rng(11)
    stopped = [_stops(*_simulate(rng, -6.0, 1.0), replicates=500) for _ in range(40)]
    assert sum(look is not None for look in stopped) >= 36
    slow = [_stops(*_simulate(rng, 3.0, 2.5), replicates=500) for _ in range(20)]
    assert all(look == 1 for look in slow)  # the latency bound fires at the first look


@settings(max_examples=5, deadline=None, derandomize=True)
@given(
    delta=st.floats(min_value=1.0, max_value=4.0),
    ratio=st.floats(min_value=0.5, max_value=1.5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_arms_that_truly_meet_the_criteria_are_rarely_stopped(
    delta: float, ratio: float, seed: int
) -> None:
    """The stated bound holds in simulation: at a true gain ≥ +1.0 pt and a true p50 ratio
    ≤ 1.5x, the stop rate stays within FALSE_STOP_BOUND plus three Monte Carlo SEs. 500
    bootstrap replicates (not the pre-registered 2000) keep the simulation fast; the rule and
    its bounds are otherwise the production ones."""
    rng = np.random.default_rng(seed)
    sims = 150
    stopped = sum(
        _stops(*_simulate(rng, delta, ratio), replicates=500) is not None for _ in range(sims)
    )
    slack = 3 * math.sqrt(FALSE_STOP_BOUND * (1 - FALSE_STOP_BOUND) / sims)
    assert stopped / sims <= FALSE_STOP_BOUND + slack


def _row(sha: str, number: str | None, latency: float | None, valid: bool = True) -> dict[str, Any]:
    fields = {"invoice_number": {"pred": number, "correct": number == "INV-1"}}
    return {"sha": sha, "valid_json": valid, "latency_ms": latency, "fields": fields}


def test_futility_reads_real_rows_and_takes_only_looks_both_arms_have(tmp_path: Path) -> None:
    shas = [f"{i:064x}" for i in range(8)]
    for sha in shas:
        write_uae_doc(tmp_path, sha)
    base = {sha: _row(sha, "INV-1", 1000.0) for sha in shas}
    wrong = {sha: _row(sha, "INV-9", 1000.0) for sha in shas}
    stop = futility("q38-27b", wrong, base, tmp_path, shas)
    assert stop.stopped is not None and stop.stopped.look == 1 and stop.stopped.n == 2
    assert stop.stopped.delta_points == -100.0
    # The per-document counts sum to exactly what the pooled scorer reports.
    counts = doc_counts(wrong, tmp_path, shas).sum(axis=0)
    pooled = score_rows(wrong, tmp_path, shas, "x").overall
    assert tuple(counts) == (pooled.true_positives, pooled.false_positives, pooled.false_negatives)
    same = futility("q38-27b", base, base, tmp_path, shas)
    assert same.stopped is None and [look.n for look in same.looks] == [2, 4, 6]
    partial = {sha: base[sha] for sha in shas[:5]}
    assert [look.n for look in futility("q38-27b", partial, base, tmp_path, shas).looks] == [2, 4]
    gap = {sha: base[sha] for sha in shas if sha != shas[0]}
    assert futility("q38-27b", gap, base, tmp_path, shas).looks == ()
