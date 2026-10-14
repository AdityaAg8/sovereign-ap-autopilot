"""Predictors the eval harness can score: a small name-keyed registry so
evals/run.py's `--predictor` flag can grow beyond `dummy` (e.g. a real extractor in) without
changing its CLI shape or call sites."""

from __future__ import annotations

from collections.abc import Callable

from evals.predictors.base import Predictor
from evals.predictors.dummy import DummyPredictor

PREDICTORS: dict[str, Callable[[], Predictor]] = {
    "dummy": DummyPredictor,
}


def get_predictor(name: str) -> Predictor:
    """Instantiates a registered predictor by name. Raises `KeyError` listing the valid choices
    if `name` is not registered; the CLI additionally blocks this earlier via `argparse`
    `choices=`, so this is the guard for direct (non-CLI) callers."""
    try:
        factory = PREDICTORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown predictor {name!r}; expected one of {sorted(PREDICTORS)}") from exc
    return factory()


__all__ = ["PREDICTORS", "Predictor", "get_predictor"]
