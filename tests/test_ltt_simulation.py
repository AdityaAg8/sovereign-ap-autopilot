"""M7.5 gate (ADR-0009 §7 as amended, PRD §6.1): the 500-split known-ground-truth simulation of the
whole calibrate + Learn-then-Test chain. Pass iff coverage >= (1 - delta) - 0.02 = 0.93, plus two
non-degeneracy checks derived from the oracle ceiling: lambda_hat found in >= 450/500 splits and a
median reference accept rate >= 0.75 x the oracle ceiling computed in code. Seeded, deterministic.

The gated run is shared module-wide (`gated`) and spread over a small process pool: results do
not depend on the worker count, because each split depends only on its own spawned seed.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import pytest

from apx.trust.interfaces import SimulationConfig, SimulationResult
from apx.trust.simulation import oracle_ceiling, simulate_ltt_coverage

# The gated case (ADR-0009 §7): alpha=1%, delta=5%, icc=0 (exchangeable, no document clustering).
_GATED_CONFIG = SimulationConfig(alpha=0.01, delta=0.05, icc=0.0)
_WORKERS = max(1, min(8, os.cpu_count() or 1))

# Non-degeneracy thresholds (ADR-0009 §7 amendment): measured 496/500 and 0.92 x oracle.
_MIN_SPLITS_WITH_LAMBDA_HAT = 450
_MIN_MEDIAN_ACCEPT_FRACTION_OF_ORACLE = 0.75
# The generator must itself admit a majority accept rate at alpha (the pre-amendment one gave 0.15).
_MIN_ORACLE_CEILING = 0.5
_DETERMINISM_PREFIX = 10


@dataclass(frozen=True)
class _TimedResult:
    result: SimulationResult
    elapsed_seconds: float


@pytest.fixture(scope="module")
def gated() -> _TimedResult:
    start = time.perf_counter()
    result = simulate_ltt_coverage(_GATED_CONFIG, workers=_WORKERS)
    elapsed = time.perf_counter() - start
    return _TimedResult(result=result, elapsed_seconds=elapsed)


@pytest.fixture(scope="module")
def ceiling() -> float:
    return oracle_ceiling(_GATED_CONFIG)


def test_simulation_reports_elapsed_seconds(
    gated: _TimedResult, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-0009 §7.B: the wall-clock time is reported, not asserted (see
    test_simulation_stays_within_a_perf_budget for the asserted, `perf`-marked form). This test
    only makes sure the number is actually surfaced somewhere, since a value nothing ever prints
    or checks is not "reported" in any meaningful sense."""
    print(f"LTT gated simulation elapsed_seconds={gated.elapsed_seconds:.2f}")
    captured = capsys.readouterr()
    assert "elapsed_seconds=" in captured.out


@pytest.mark.perf
def test_simulation_stays_within_a_perf_budget(gated: _TimedResult) -> None:
    """ADR-0009 §7.B: the <60s wall-clock bound is real but flaky under host load (measured
    32-95s across otherwise-identical runs, see §7.B), so it is asserted only under the `perf`
    marker (`pytest -m perf`, run on an idle host) and excluded from the default CI suite."""
    assert gated.elapsed_seconds < 60.0, (
        f"simulation took {gated.elapsed_seconds:.1f}s, must stay under 60s on an idle host"
    )


def test_simulation_is_deterministic_and_independent_of_workers(gated: _TimedResult) -> None:
    """A serial rerun of the first splits reproduces the pooled gated run exactly: split seeds are
    spawned by index, so a prefix of the run is itself a complete, comparable run."""
    prefix_config = _GATED_CONFIG.model_copy(update={"n_splits": _DETERMINISM_PREFIX})
    prefix = simulate_ltt_coverage(prefix_config, workers=1)
    n = _DETERMINISM_PREFIX
    assert prefix.lambda_hats == gated.result.lambda_hats[:n]
    assert prefix.true_risks == gated.result.true_risks[:n]
    assert prefix.reference_accept_rates == gated.result.reference_accept_rates[:n]
    assert prefix.test_errors == gated.result.test_errors[:n]


def test_simulation_coverage_meets_the_ltt_guarantee(gated: _TimedResult) -> None:
    """ADR-0009 §7: coverage = mean_s 1[R_s <= alpha] must be >= (1 - delta) - 0.02 = 0.93."""
    result = gated.result
    assert len(result.true_risks) == _GATED_CONFIG.n_splits

    recomputed_coverage = statistics.mean(
        1.0 if risk <= _GATED_CONFIG.alpha else 0.0 for risk in result.true_risks
    )
    assert recomputed_coverage == result.coverage
    assert result.coverage >= 0.93


def test_generator_admits_a_meaningful_accept_rate(ceiling: float) -> None:
    """The oracle ceiling (exact p*, best possible accept rule) must leave room to accept most
    fields at alpha; otherwise the non-degeneracy checks below would be unsatisfiable."""
    assert ceiling >= _MIN_ORACLE_CEILING, f"oracle ceiling {ceiling:.4f} < {_MIN_ORACLE_CEILING}"


def test_simulation_is_not_degenerate(gated: _TimedResult, ceiling: float) -> None:
    """Stops an accept-nothing (or accept-almost-nothing) implementation from passing vacuously
    (ADR-0009 §7 amendment): lambda_hat is found in >= 450/500 splits, and the median reference
    accept rate is >= 0.75 x the oracle ceiling."""
    result = gated.result

    n_found = sum(1 for lam in result.lambda_hats if lam is not None)
    assert n_found >= _MIN_SPLITS_WITH_LAMBDA_HAT, (
        f"lambda_hat found in only {n_found}/{_GATED_CONFIG.n_splits} splits"
    )

    median_accept_rate = statistics.median(result.reference_accept_rates)
    floor = _MIN_MEDIAN_ACCEPT_FRACTION_OF_ORACLE * ceiling
    assert median_accept_rate >= floor, (
        f"median reference accept rate {median_accept_rate:.4f} < 0.75 x oracle {ceiling:.4f}"
    )


def test_simulation_result_shapes_match_the_configured_split_count(gated: _TimedResult) -> None:
    result = gated.result
    n = _GATED_CONFIG.n_splits
    assert len(result.lambda_hats) == n
    assert len(result.true_risks) == n
    assert len(result.reference_accept_rates) == n
    assert len(result.test_errors) == n


def test_simulation_test_errors_are_none_only_when_lambda_hat_is_none(gated: _TimedResult) -> None:
    result = gated.result
    for lam, test_error in zip(result.lambda_hats, result.test_errors, strict=True):
        if lam is None:
            assert test_error is None
