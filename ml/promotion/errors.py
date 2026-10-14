"""Typed errors for the M10.2 promotion pipeline: any shadow eval that cannot prove "no gate
regressed" raises one of these instead of promoting (no silent fallbacks)."""

from __future__ import annotations

from apx.common.errors import ApxError


class PromotionError(ApxError):
    """Base of every promotion-pipeline error. The `champion` alias never moves when one is
    raised."""


class MissingEvalSetError(PromotionError):
    """A required eval set (ADR-0015 §3 `SET_POLICY`) is absent from, duplicated in, or
    deferred (not scored) in the champion's or the challenger's shadow eval."""


class MissingMetricError(PromotionError):
    """A gated metric is present on one side but not the other, has no measured value, or a
    scored set has no comparable metric at all."""


class InvalidMetricError(PromotionError):
    """A gated metric is not a finite real number (NaN, infinity, bool or non-numeric)."""


class EvalMismatchError(PromotionError):
    """The two shadow evals are not comparable (ADR-0015 §4): different eval code, scorer,
    session, test digests, split, document count, sample limit, gate kind or threshold, or a
    report from the wrong model (gates never move, I6)."""


class NoNewLabelsError(PromotionError):
    """The corrections export produced no training rows, so there is nothing to retrain on."""


class ChampionChangedError(PromotionError):
    """The `champion` alias no longer points at the version that was shadow-evaluated (another
    promotion or a manual alias move ran meanwhile), so the decision is stale. Raised before any
    promotion side effect (ADR-0015 §5 compare-and-set)."""

    def __init__(self, *, evaluated: str, current: str) -> None:
        super().__init__(
            f"champion alias moved from evaluated version {evaluated} to {current} during "
            "retrain; the decision is stale and no side effect ran"
        )
        self.evaluated = evaluated
        self.current = current


class PromotionApplyError(PromotionError):
    """A PROMOTE decision was audited but a side effect (calibrator invalidation, models.lock
    entry or the `champion` alias move) failed. `applied` names the steps that completed, in
    order, and `failed_stage` the one that raised; the alias has moved only if that step
    succeeded. Recovery: ADR-0015 §5 "Partial apply"."""

    def __init__(
        self,
        *,
        champion: str,
        challenger: str,
        failed_stage: str,
        applied: tuple[str, ...],
        cause: BaseException,
    ) -> None:
        done = ", ".join(applied) if applied else "none"
        super().__init__(
            f"promotion of {challenger} (champion {champion}) failed at {failed_stage}: "
            f"{cause}; steps already applied: {done}"
        )
        self.champion = champion
        self.challenger = challenger
        self.failed_stage = failed_stage
        self.applied = applied


class RetrainPrerequisiteError(PromotionError):
    """`make retrain` cannot run because production adapters are missing; the message names
    each one (ADR-0015 §5). A typed stop, not a fallback."""
