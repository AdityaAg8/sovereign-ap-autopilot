"""Unit tests for the M10.2 retrain orchestration (`ml.promotion.retrain`, ADR-0015 §5) with
in-memory fakes for export, training, shadow eval, registry, calibrator, models.lock and audit:
a degraded challenger is auto-rejected with no side effects, a promotion runs its side effects
in order, failures are audited, and `make retrain` stops with exit 2 until adapters exist."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apx.audit.interfaces import GENESIS_HASH, AuditEntry, AuditEvent
from apx.common.errors import AuditChainError
from ml.promotion.decide import Verdict
from ml.promotion.errors import (
    ChampionChangedError,
    EvalMismatchError,
    MissingEvalSetError,
    NoNewLabelsError,
    PromotionApplyError,
    RetrainPrerequisiteError,
)
from ml.promotion.evidence import ShadowEval
from ml.promotion.retrain import (
    ACTION_APPLIED,
    ACTION_DECIDED,
    ACTION_FAILED,
    EXIT_PREREQUISITE,
    ModelsLockEntry,
    ModelVersion,
    RegistryAlias,
    RetrainOutcome,
    TrainedChallenger,
    compose_adapters,
    main,
    run_retrain,
)
from tests.unit.ml._promotion_fixtures import (
    CHAMPION,
    Predictor,
    corrupt,
    manifest,
    perfect,
    run_report,
    shadow,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAMPION_V = ModelVersion("3")
CHALLENGER_V = ModelVersion("4")


class FakeExporter:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    async def export(self, out_path: Path) -> int:
        out_path.write_text("{}\n" * self.rows, encoding="utf-8")
        return self.rows


class FakeTrainer:
    def __init__(self, version: ModelVersion = CHALLENGER_V) -> None:
        self.version = version
        self.trained_on: list[Path] = []

    async def train(self, manifest_path: Path) -> TrainedChallenger:
        self.trained_on.append(manifest_path)
        return TrainedChallenger(
            version=self.version,
            mlflow_run_id="mlflow-run-4",
            gguf_sha256="e" * 64,
            mmproj_sha256="f" * 64,
        )


class FakeEvaluator:
    """Returns each version's `ShadowEval`, built with the real scorers (fixtures)."""

    def __init__(self, evals: dict[str, ShadowEval | Exception]) -> None:
        self.evals = evals
        self.evaluated: list[str] = []

    async def evaluate(self, version: ModelVersion) -> ShadowEval:
        self.evaluated.append(version)
        result = self.evals[version]
        if isinstance(result, Exception):
            raise result
        return result


class FakeRegistry:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.aliases: dict[RegistryAlias, ModelVersion] = {"champion": CHAMPION_V}
        self.fail_champion = False
        self.champion_reads = 0
        self.moved_by_other: ModelVersion | None = None  # set on the 2nd champion read

    def version_for(self, alias: RegistryAlias) -> ModelVersion:
        if alias == "champion":
            self.champion_reads += 1
            if self.champion_reads == 2 and self.moved_by_other is not None:
                self.aliases["champion"] = self.moved_by_other  # a concurrent promotion
        return self.aliases[alias]

    def set_alias(self, alias: RegistryAlias, version: ModelVersion) -> None:
        if alias == "champion" and self.fail_champion:
            raise RuntimeError("mlflow unreachable")
        self.calls.append(f"alias:{alias}")
        self.aliases[alias] = version


class FakeInvalidator:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.invalidated: list[ModelVersion] = []

    def invalidate(self, new_champion: ModelVersion, reason: str) -> None:
        self.calls.append("invalidate")
        self.invalidated.append(new_champion)


class FakeLockWriter:
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail
        self.entries: list[ModelsLockEntry] = []

    def record(self, entry: ModelsLockEntry) -> None:
        if self.fail:
            raise OSError("models.lock not writable")
        self.calls.append("lock")
        self.entries.append(entry)


class FakeAudit:
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        if self.fail:
            raise AuditChainError("audit store down")
        self.calls.append(event.action)
        self.events.append(event)
        return AuditEntry(
            id=len(self.events),
            ts=datetime(2026, 9, 24, tzinfo=UTC),
            actor=event.actor,
            action=event.action,
            entity=event.entity,
            payload=event.payload,
            prev_hash=GENESIS_HASH,
            hash="1" * 64,
        )


def _evals(challenger: Predictor) -> dict[str, ShadowEval | Exception]:
    return {
        CHAMPION_V: shadow(CHAMPION, version=CHAMPION_V),
        CHALLENGER_V: shadow(challenger, version=CHALLENGER_V),
    }


@dataclass
class Deps:
    challenger: Predictor
    calls: list[str] = field(default_factory=list)
    exporter: FakeExporter = field(default_factory=lambda: FakeExporter(rows=12))
    trainer: FakeTrainer = field(default_factory=FakeTrainer)
    evaluator: FakeEvaluator | None = None
    lock_fail: bool = False
    audit_fail: bool = False

    def __post_init__(self) -> None:
        self.evaluator = self.evaluator or FakeEvaluator(_evals(self.challenger))
        self.registry = FakeRegistry(self.calls)
        self.invalidator = FakeInvalidator(self.calls)
        self.lock_writer = FakeLockWriter(self.calls, fail=self.lock_fail)
        self.audit = FakeAudit(self.calls, fail=self.audit_fail)

    async def run(self, manifest_path: Path) -> RetrainOutcome:
        assert self.evaluator is not None
        return await run_retrain(
            exporter=self.exporter,
            trainer=self.trainer,
            evaluator=self.evaluator,
            registry=self.registry,
            audit=self.audit,
            invalidator=self.invalidator,
            lock_writer=self.lock_writer,
            manifest=manifest(),
            manifest_path=manifest_path,
        )


async def test_deliberately_degraded_challenger_is_auto_rejected(tmp_path: Path) -> None:
    deps = Deps(corrupt(5))

    outcome = await deps.run(tmp_path / "manifest.jsonl")

    assert outcome.decision.verdict is Verdict.REJECT
    assert {c.gate for c in outcome.decision.regressions} >= {"G1"}
    assert outcome.champion_after == CHAMPION_V
    assert deps.registry.aliases == {"champion": CHAMPION_V, "challenger": CHALLENGER_V}
    assert deps.invalidator.invalidated == []
    assert deps.lock_writer.entries == []
    assert deps.calls == ["alias:challenger", ACTION_DECIDED]
    [event] = deps.audit.events
    assert event.payload["verdict"] == "reject"
    assert event.payload["manifest_rows"] == 12


async def test_promotion_runs_side_effects_in_order(tmp_path: Path) -> None:
    deps = Deps(perfect)
    manifest_path = tmp_path / "manifest.jsonl"

    outcome = await deps.run(manifest_path)

    assert outcome.decision.verdict is Verdict.PROMOTE
    assert outcome.champion_after == CHALLENGER_V
    assert deps.calls == [
        "alias:challenger",
        ACTION_DECIDED,
        "invalidate",
        "lock",
        "alias:champion",
        ACTION_APPLIED,
    ]
    [entry] = deps.lock_writer.entries
    assert entry.version == CHALLENGER_V
    assert entry.source == "local"
    assert entry.license == "Apache-2.0"
    assert entry.base_revision == "rev-base"
    assert entry.mlflow_run_id == "mlflow-run-4"
    applied = deps.audit.events[-1]
    assert applied.payload == {"old_champion": CHAMPION_V, "new_champion": CHALLENGER_V}
    assert deps.trainer.trained_on == [manifest_path]


async def test_lock_writer_failure_leaves_alias_unmoved_and_audits(tmp_path: Path) -> None:
    deps = Deps(perfect, lock_fail=True)

    with pytest.raises(PromotionApplyError, match="models_lock") as info:
        await deps.run(tmp_path / "m.jsonl")

    assert info.value.failed_stage == "models_lock"
    assert info.value.applied == ("invalidate_calibrator",)
    assert deps.registry.aliases["champion"] == CHAMPION_V
    assert deps.calls[-1] == ACTION_FAILED
    assert deps.audit.events[-1].payload["stage"] == "models_lock"
    assert deps.audit.events[-1].payload["applied"] == ["invalidate_calibrator"]
    assert ACTION_APPLIED not in deps.calls


async def test_alias_failure_reports_exactly_what_was_applied(tmp_path: Path) -> None:
    deps = Deps(perfect)
    deps.registry.fail_champion = True

    with pytest.raises(PromotionApplyError, match="alias") as info:
        await deps.run(tmp_path / "m.jsonl")

    err = info.value
    assert (err.champion, err.challenger) == (CHAMPION_V, CHALLENGER_V)
    assert err.failed_stage == "alias"
    assert err.applied == ("invalidate_calibrator", "models_lock")
    assert "invalidate_calibrator, models_lock" in str(err)
    assert deps.registry.aliases["champion"] == CHAMPION_V
    assert deps.invalidator.invalidated == [CHALLENGER_V]
    assert [e.version for e in deps.lock_writer.entries] == [CHALLENGER_V]
    assert deps.calls[-1] == ACTION_FAILED
    failed = deps.audit.events[-1].payload
    assert failed["stage"] == "alias"
    assert failed["applied"] == ["invalidate_calibrator", "models_lock"]


async def test_champion_moved_during_retrain_blocks_every_side_effect(tmp_path: Path) -> None:
    deps = Deps(perfect)
    deps.registry.moved_by_other = ModelVersion("5")

    with pytest.raises(ChampionChangedError) as info:
        await deps.run(tmp_path / "m.jsonl")

    assert (info.value.evaluated, info.value.current) == (CHAMPION_V, "5")
    assert deps.registry.aliases["champion"] == "5"  # the other promotion stands
    assert deps.invalidator.invalidated == []
    assert deps.lock_writer.entries == []
    assert deps.calls == ["alias:challenger", ACTION_DECIDED, ACTION_FAILED]
    failed = deps.audit.events[-1].payload
    assert failed["stage"] == "compare_and_set"
    assert failed["error"] == "ChampionChangedError"


async def test_unchanged_champion_passes_the_compare_and_set(tmp_path: Path) -> None:
    deps = Deps(perfect)

    outcome = await deps.run(tmp_path / "m.jsonl")

    assert deps.registry.champion_reads >= 2  # read at start and again before side effects
    assert outcome.champion_after == CHALLENGER_V


async def test_audit_failure_blocks_every_side_effect(tmp_path: Path) -> None:
    deps = Deps(perfect, audit_fail=True)

    with pytest.raises(AuditChainError):
        await deps.run(tmp_path / "m.jsonl")

    assert deps.registry.aliases["champion"] == CHAMPION_V
    assert deps.invalidator.invalidated == []
    assert deps.lock_writer.entries == []


async def test_decider_error_is_audited_and_raised(tmp_path: Path) -> None:
    evals = _evals(perfect)
    evals[CHALLENGER_V] = shadow(
        perfect, version=CHALLENGER_V, report=run_report(perfect, version="4", drop=("fatura",))
    )
    deps = Deps(perfect, evaluator=FakeEvaluator(evals))

    with pytest.raises(MissingEvalSetError):
        await deps.run(tmp_path / "m.jsonl")

    assert deps.registry.aliases["champion"] == CHAMPION_V
    [event] = deps.audit.events
    assert event.action == ACTION_FAILED
    assert event.payload["error"] == "MissingEvalSetError"


async def test_evaluator_failure_is_audited_and_raised(tmp_path: Path) -> None:
    evals = _evals(perfect)
    evals[CHALLENGER_V] = RuntimeError("llama-swap extract group down")
    deps = Deps(perfect, evaluator=FakeEvaluator(evals))

    with pytest.raises(RuntimeError, match="llama-swap"):
        await deps.run(tmp_path / "m.jsonl")

    [event] = deps.audit.events
    assert event.action == ACTION_FAILED
    assert event.payload["stage"] == "evaluate"
    assert deps.registry.aliases["champion"] == CHAMPION_V


async def test_report_from_the_wrong_model_is_a_typed_error(tmp_path: Path) -> None:
    evals = _evals(perfect)
    evals[CHALLENGER_V] = shadow(perfect, version="99")  # predictor extractor_b@99
    deps = Deps(perfect, evaluator=FakeEvaluator(evals))

    with pytest.raises(EvalMismatchError, match="extractor_b@4"):
        await deps.run(tmp_path / "m.jsonl")

    assert deps.audit.events[-1].action == ACTION_FAILED
    assert deps.registry.aliases["champion"] == CHAMPION_V


async def test_no_new_labels_is_a_typed_error_before_training(tmp_path: Path) -> None:
    deps = Deps(perfect, exporter=FakeExporter(rows=0))

    with pytest.raises(NoNewLabelsError):
        await deps.run(tmp_path / "m.jsonl")

    assert deps.trainer.trained_on == []


async def test_trainer_returning_the_champion_version_is_a_typed_error(tmp_path: Path) -> None:
    deps = Deps(perfect, trainer=FakeTrainer(version=CHAMPION_V))

    with pytest.raises(EvalMismatchError):
        await deps.run(tmp_path / "m.jsonl")


def test_compose_adapters_names_every_missing_adapter() -> None:
    with pytest.raises(RetrainPrerequisiteError) as info:
        compose_adapters()
    for name in ("CorrectionsExporter", "ShadowEvaluator", "ModelsLockWriter", "Calibrator"):
        assert name in str(info.value)


def test_main_exits_2_with_prerequisite_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_PREREQUISITE == 2
    assert "RetrainPrerequisiteError" in capsys.readouterr().err


def test_make_retrain_entry_point_exits_2() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\nretrain:\n", 1)[1].split("\n\n", 1)[0]
    assert "python -m ml.promotion.retrain" in recipe

    result = subprocess.run(
        [sys.executable, "-m", "ml.promotion.retrain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 2
    assert "RetrainPrerequisiteError" in result.stderr
