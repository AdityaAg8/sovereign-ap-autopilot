"""`make retrain` orchestration (ADR-0015 §5): corrections export, LoRA training, shadow eval of
champion and challenger, the decision, then on PROMOTE the ordered side effects (audit,
calibrator invalidation, models.lock entry, alias move, audit) through injected protocols.
`main()` stops with exit 2 and `RetrainPrerequisiteError` until the adapters exist."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NewType, Protocol

from pydantic import JsonValue

from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.types import Frozen, Sha256Hex
from evals.datasets.manifest import Manifest
from ml.promotion.decide import PromotionDecision, Verdict, decide
from ml.promotion.errors import (
    ChampionChangedError,
    EvalMismatchError,
    NoNewLabelsError,
    PromotionApplyError,
    RetrainPrerequisiteError,
)
from ml.promotion.evidence import ShadowEval

ModelVersion = NewType("ModelVersion", str)
"""A registered model version (the MLflow model-version number for `extractor_b`)."""

RegistryAlias = Literal["champion", "challenger"]

AUDIT_ACTOR: Final[str] = "retrain"
AUDIT_ENTITY: Final[str] = "model:extractor_b"
ACTION_DECIDED: Final[str] = "promotion.decided"
ACTION_APPLIED: Final[str] = "promotion.applied"
ACTION_FAILED: Final[str] = "promotion.failed"
EXIT_PREREQUISITE: Final[int] = 2
STAGE_COMPARE_AND_SET: Final[str] = "compare_and_set"
STAGE_INVALIDATE: Final[str] = "invalidate_calibrator"
STAGE_MODELS_LOCK: Final[str] = "models_lock"
STAGE_ALIAS: Final[str] = "alias"


def predictor_name(version: ModelVersion) -> str:
    """The `RunReport.predictor` a shadow eval of `version` must carry (ADR-0015 §4)."""
    return f"extractor_b@{version}"


class TrainedChallenger(Frozen):
    """What the trainer registered: the version plus what a models.lock entry needs."""

    version: ModelVersion
    mlflow_run_id: str
    gguf_sha256: Sha256Hex
    mmproj_sha256: Sha256Hex


class ModelsLockEntry(Frozen):
    """One idempotent `infra/models.lock.json` entry for a promoted local build (ADR-0009
    §1.8), keyed by version; inert until the alias points at it."""

    version: ModelVersion
    source: Literal["local"] = "local"
    mlflow_run_id: str
    base_revision: str
    gguf_sha256: Sha256Hex
    mmproj_sha256: Sha256Hex
    license: Literal["Apache-2.0"] = "Apache-2.0"


class CorrectionsExporter(Protocol):
    """Writes the training manifest (wrapping `apx.learning.export.export_training_manifest`)
    and returns the number of rows written."""

    async def export(self, out_path: Path) -> int: ...


class ChallengerTrainer(Protocol):
    """Trains a LoRA challenger on the manifest (the ADR-0011 job) and registers it."""

    async def train(self, manifest_path: Path) -> TrainedChallenger: ...


class ShadowEvaluator(Protocol):
    """Measures one version in the current shadow-eval session (ADR-0015 §1)."""

    async def evaluate(self, version: ModelVersion) -> ShadowEval: ...


class ModelAliasRegistry(Protocol):
    """The registered-model aliases (MLflow `champion` / `challenger`)."""

    def version_for(self, alias: RegistryAlias) -> ModelVersion:
        """Return the version under `alias`; raise if the alias is unset."""
        ...

    def set_alias(self, alias: RegistryAlias, version: ModelVersion) -> None: ...


class CalibratorInvalidator(Protocol):
    """Marks the active calibrator and λ̂ stale; the trust gate accepts nothing until the
    trust cal job refits on the new champion."""

    def invalidate(self, new_champion: ModelVersion, reason: str) -> None: ...


class ModelsLockWriter(Protocol):
    def record(self, entry: ModelsLockEntry) -> None: ...


class RetrainOutcome(Frozen):
    """What one `run_retrain` call did."""

    manifest_rows: int
    champion_before: ModelVersion
    challenger: ModelVersion
    decision: PromotionDecision
    champion_after: ModelVersion


async def _audit(audit: AuditLog, action: str, payload: dict[str, JsonValue]) -> None:
    await audit.append(
        AuditEvent(actor=AUDIT_ACTOR, action=action, entity=AUDIT_ENTITY, payload=payload)
    )


async def _audit_failure(
    audit: AuditLog, versions: dict[str, JsonValue], stage: str, exc: BaseException
) -> None:
    await _audit(
        audit,
        ACTION_FAILED,
        {**versions, "stage": stage, "error": type(exc).__name__, "detail": str(exc)},
    )


async def run_retrain(
    *,
    exporter: CorrectionsExporter,
    trainer: ChallengerTrainer,
    evaluator: ShadowEvaluator,
    registry: ModelAliasRegistry,
    audit: AuditLog,
    invalidator: CalibratorInvalidator,
    lock_writer: ModelsLockWriter,
    manifest: Manifest,
    manifest_path: Path,
) -> RetrainOutcome:
    """Export → train → shadow eval (champion, challenger) → decide → audit, then on PROMOTE:
    re-read the `champion` alias (compare-and-set: it must still be the evaluated version, else
    `ChampionChangedError`, audited, no side effect) → invalidate calibrator → models.lock →
    alias → audit applied. Evaluation or decision errors are audited as `promotion.failed` and
    re-raised; a side-effect failure is audited with the steps already applied and raised as
    `PromotionApplyError` carrying them (recovery: ADR-0015 §5). On REJECT no side effect runs."""
    rows = await exporter.export(manifest_path)
    if rows <= 0:
        raise NoNewLabelsError(f"corrections export wrote {rows} rows to {manifest_path}")
    champion = registry.version_for("champion")
    trained = await trainer.train(manifest_path)
    challenger = trained.version
    if challenger == champion:
        raise EvalMismatchError(f"trainer returned the champion version {champion} as challenger")
    registry.set_alias("challenger", challenger)
    versions: dict[str, JsonValue] = {"champion": champion, "challenger": challenger}

    try:
        champion_eval = await evaluator.evaluate(champion)
        challenger_eval = await evaluator.evaluate(challenger)
        for version, shadow in ((champion, champion_eval), (challenger, challenger_eval)):
            if shadow.report.predictor != predictor_name(version):
                raise EvalMismatchError(
                    f"shadow eval for {version} is from predictor {shadow.report.predictor!r}, "
                    f"expected {predictor_name(version)!r}"
                )
        decision = decide(champion_eval, challenger_eval, manifest=manifest)
    except Exception as exc:
        await _audit_failure(audit, versions, "evaluate", exc)
        raise

    # Audit first: if the append fails nothing below runs, so no promotion goes unrecorded.
    await _audit(
        audit, ACTION_DECIDED, {**versions, "manifest_rows": rows, **decision.to_payload()}
    )
    if decision.verdict is Verdict.PROMOTE:
        current = registry.version_for("champion")
        if current != champion:
            stale = ChampionChangedError(evaluated=champion, current=current)
            await _audit_failure(audit, versions, STAGE_COMPARE_AND_SET, stale)
            raise stale
        stage = STAGE_INVALIDATE
        applied: list[str] = []
        try:
            invalidator.invalidate(challenger, f"champion {champion} -> {challenger} (retrain)")
            applied.append(stage)
            stage = STAGE_MODELS_LOCK
            lock_writer.record(
                ModelsLockEntry(
                    version=challenger,
                    mlflow_run_id=trained.mlflow_run_id,
                    base_revision=challenger_eval.gates.base_revision,
                    gguf_sha256=trained.gguf_sha256,
                    mmproj_sha256=trained.mmproj_sha256,
                )
            )
            applied.append(stage)
            stage = STAGE_ALIAS
            registry.set_alias("champion", challenger)
        except Exception as exc:
            await _audit_failure(audit, {**versions, "applied": list(applied)}, stage, exc)
            raise PromotionApplyError(
                champion=champion,
                challenger=challenger,
                failed_stage=stage,
                applied=tuple(applied),
                cause=exc,
            ) from exc
        await _audit(audit, ACTION_APPLIED, {"old_champion": champion, "new_champion": challenger})
    return RetrainOutcome(
        manifest_rows=rows,
        champion_before=champion,
        challenger=challenger,
        decision=decision,
        champion_after=registry.version_for("champion"),
    )


# --- `make retrain` entry point -------------------------------------------------------------

REQUIRED_ADAPTERS: Final[tuple[str, ...]] = (
    "CorrectionsExporter (apx.learning.export.export_training_manifest over the labels DB)",
    "ChallengerTrainer (ADR-0011 detached LoRA job)",
    "ShadowEvaluator (ADR-0009 build item 10, ml/extractor_b/promote.py P1-P5 run)",
    "ModelAliasRegistry (MLflow extractor_b aliases)",
    "CalibratorInvalidator (trust store)",
    "ModelsLockWriter (infra/models.lock.json)",
    "AuditLog (apx.audit.chain.PgAuditLog)",
)


def compose_adapters() -> None:
    """Build the production adapters. None exist yet (ADR-0015 §5), so this always raises
    `RetrainPrerequisiteError` naming every missing one."""
    raise RetrainPrerequisiteError(
        "make retrain needs production adapters that do not exist yet: "
        + "; ".join(REQUIRED_ADAPTERS)
    )


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    try:
        compose_adapters()
    except RetrainPrerequisiteError as exc:
        print(f"retrain: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_PREREQUISITE
    return 0  # pragma: no cover - unreachable until the adapters land


if __name__ == "__main__":
    sys.exit(main())
