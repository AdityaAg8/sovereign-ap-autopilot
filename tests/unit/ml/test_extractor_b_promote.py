"""Unit tests for Extractor B registration and promotion (`ml.extractor_b.promote`, ADR-0009
§1.8, ADR-0015 §5) against an in-memory MLflow REST fake (httpx MockTransport), a real small
export on disk, a temp models.lock and a copy of infra/llama-swap.yaml: idempotent register, gates
evidence binding, the ordered side effects, compare-and-set, and fail-closed refusals, including
a challenger without its ADR-0019 §9.4 real hold-out reports."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
import yaml

from apx.extract.identity import served_paths
from evals.extraction_identity import RowIdentity
from evals.scorers.perf import LatencyStats
from ml.extractor_b.export import (
    MANIFEST_NAME,
    MERGED_DIR_NAME,
    MMPROJ_NAME,
    Q4_NAME,
    ExportManifest,
    digest_file,
    digest_tree,
)
from ml.extractor_b.gates import GateEvidenceError, Hygiene, LatencyRun, RealHoldoutsMissingError
from ml.extractor_b.promote import (
    STAGE_LLAMA_SWAP,
    ChampionMovedDuringApplyError,
    EvidenceMismatchError,
    ExportVerificationError,
    FileCalibratorInvalidator,
    GatesNotPassedError,
    JsonModelsLockWriter,
    LlamaSwapConfigError,
    LlamaSwapRepointer,
    LockConflictError,
    MlflowRegistry,
    RegistryError,
    UaeScores,
    assemble_evidence,
    container_path,
    docile_present,
    eval_b_argv,
    expected_identities,
    parse_args,
    promote,
    publish_evidence,
    register,
    require_challenger_serves,
    score_row_files,
)
from ml.promotion.errors import ChampionChangedError, PromotionApplyError
from ml.promotion.retrain import (
    STAGE_ALIAS,
    STAGE_INVALIDATE,
    STAGE_MODELS_LOCK,
    ModelsLockEntry,
    ModelVersion,
)
from tests.unit.ml.conftest import sha
from tests.unit.ml.test_real_holdouts import load_fake_reports, write_real_report

REPO_ROOT = Path(__file__).resolve().parents[3]


def _challenger_run_id() -> str:
    """The run the committed `extractor-b-challenger` entry serves (`/models/extractor_b/<run>/
    export/...`): the tests copy that config, so they follow it when a new round is exported."""
    model, _ = served_paths(REPO_ROOT / "infra" / "llama-swap.yaml", "extractor-b-challenger")
    return model.split("/")[3]


RUN_ID = _challenger_run_id()
MLFLOW_RUN = "62d8e2ab796c45d482a1063876eb69d5"
BASE_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


# --- fakes -----------------------------------------------------------------------------------


@dataclass
class FakeMlflow:
    """The slice of the MLflow 3 REST API the registry adapter uses, in memory."""

    events: list[str] = field(default_factory=list)
    models: dict[str, dict[str, str]] = field(default_factory=dict)  # name -> alias -> version
    versions: dict[str, dict[str, Any]] = field(default_factory=dict)
    fail_alias: bool = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/2.0/mlflow/")
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        body = json.loads(request.content) if request.content else {}
        if path == "registered-models/get":
            if query["name"] not in self.models:
                return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})
            aliases = [{"alias": a, "version": v} for a, v in self.models[query["name"]].items()]
            return httpx.Response(200, json={"registered_model": {"aliases": aliases}})
        if path == "registered-models/create":
            self.models[body["name"]] = {}
            return httpx.Response(200, json={"registered_model": {"name": body["name"]}})
        if path == "model-versions/search":
            run_id = re.search(r"run_id='([^']+)'", query["filter"])
            assert run_id is not None
            found = [v for v in self.versions.values() if v["run_id"] == run_id.group(1)]
            return httpx.Response(200, json={"model_versions": found} if found else {})
        if path == "model-versions/create":
            version = str(len(self.versions) + 1)
            self.versions[version] = {
                "version": version,
                "run_id": body["run_id"],
                "source": body["source"],
                "tags": list(body["tags"]),
            }
            return httpx.Response(200, json={"model_version": self.versions[version]})
        if path == "model-versions/get":
            return httpx.Response(200, json={"model_version": self.versions[query["version"]]})
        if path == "model-versions/set-tag":
            tags = [t for t in self.versions[body["version"]]["tags"] if t["key"] != body["key"]]
            self.versions[body["version"]]["tags"] = [
                *tags,
                {"key": body["key"], "value": body["value"]},
            ]
            return httpx.Response(200, json={})
        if path == "registered-models/alias":
            if self.fail_alias:
                return httpx.Response(503, text="unavailable")
            self.events.append(f"alias:{body['alias']}={body['version']}")
            self.models[body["name"]][body["alias"]] = body["version"]
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error_code": "ENDPOINT_NOT_FOUND"})

    def tags(self, version: str) -> dict[str, str]:
        return {t["key"]: t["value"] for t in self.versions[version]["tags"]}


class RecordingInvalidator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def invalidate(self, new_champion: ModelVersion, reason: str) -> None:
        self.events.append(f"invalidate:{new_champion}")


class RecordingLock:
    def __init__(self, inner: JsonModelsLockWriter, events: list[str], fail: bool = False) -> None:
        self.inner, self.events, self.fail = inner, events, fail

    def record(self, entry: ModelsLockEntry) -> None:
        if self.fail:
            raise OSError("models.lock is read-only")
        self.events.append(f"lock:{entry.version}")
        self.inner.record(entry)


class RecordingRepointer(LlamaSwapRepointer):
    def __init__(self, path: Path, events: list[str]) -> None:
        super().__init__(path)
        self.events = events

    def repoint(self, *, model_path: str, mmproj_path: str, description: str) -> None:
        self.events.append("llama_swap")
        super().repoint(model_path=model_path, mmproj_path=mmproj_path, description=description)


# --- fixtures --------------------------------------------------------------------------------


@dataclass
class Env:
    root: Path
    export_dir: Path
    server: FakeMlflow
    registry: MlflowRegistry
    lock: Path
    swap: Path
    evidence_path: Path


def make_export(root: Path, q4: bytes = b"q4-weights") -> Path:
    export_dir = root / "models" / "extractor_b" / RUN_ID / "export"
    (export_dir / MERGED_DIR_NAME).mkdir(parents=True)
    (export_dir / Q4_NAME).write_bytes(q4)
    (export_dir / MMPROJ_NAME).write_bytes(b"mmproj")
    (export_dir / MERGED_DIR_NAME / "model.safetensors").write_bytes(b"merged")
    manifest = ExportManifest(
        run_id=RUN_ID,
        mlflow_run_id=MLFLOW_RUN,
        step=383,
        adapter_sha256=sha("adapter"),
        base_path="models/Qwen/Qwen3.5-4B",
        base_revision=BASE_REV,
        llama_cpp_commit="c550d2f",
        quant="Q4_K_M",
        artifacts=(
            digest_file(export_dir, Q4_NAME),
            digest_file(export_dir, MMPROJ_NAME),
            digest_tree(export_dir, MERGED_DIR_NAME),
        ),
        mmproj_upstream_path="models/unsloth/Qwen3.5-4B-GGUF/mmproj-F16.gguf",
        mmproj_upstream_sha256=sha("upstream"),
        mmproj_tensors_compared=298,
        created_at=datetime(2026, 9, 25, tzinfo=UTC),
    )
    (export_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), "utf-8")
    return export_dir


def manifest_of(export_dir: Path) -> ExportManifest:
    return ExportManifest.model_validate_json((export_dir / MANIFEST_NAME).read_bytes())


@pytest.fixture
def env(tmp_path: Path) -> Env:
    server = FakeMlflow()
    client = httpx.Client(base_url="http://mlflow.test", transport=httpx.MockTransport(server))
    lock = tmp_path / "models.lock.json"
    shutil.copy(REPO_ROOT / "infra" / "models.lock.json", lock)
    swap = tmp_path / "llama-swap.yaml"
    shutil.copy(REPO_ROOT / "infra" / "llama-swap.yaml", swap)
    return Env(
        root=tmp_path,
        export_dir=make_export(tmp_path),
        server=server,
        registry=MlflowRegistry("http://mlflow.test", client=client),
        lock=lock,
        swap=swap,
        evidence_path=tmp_path / "reports" / "extractor_b_gates.json",
    )


def scores(b_overall: float = 0.90) -> UaeScores:
    return UaeScores(
        a_overall=0.91,
        a_en=0.92,
        a_ar=0.89,
        b_overall=b_overall,
        b_en=0.91,
        b_ar=0.87,
        b_valid_json=0.997,
        n=750,
        n_en=564,
        n_ar=186,
        test_digest=sha("test"),
        a_rows_sha256=sha("a-rows"),
        b_rows_sha256=sha("b-rows"),
        a_identity=RowIdentity(model="extractor-a", gguf_sha256=sha("a"), prompt_version="pa"),
        b_identity=RowIdentity(
            model="extractor-b-challenger", gguf_sha256=sha("b"), prompt_version="pb"
        ),
    )


def latency(b_p50: float = 10.0) -> LatencyRun:
    return LatencyRun(
        a=LatencyStats(p50_s=30.0, p95_s=60.0, n=200),
        b=LatencyStats(p50_s=b_p50, p95_s=25.0, n=200),
        peak_vram_gb=10.9,
        vram_samples=1200,
    )


def gated(env: Env, *, b_overall: float = 0.90, b_p50: float = 10.0) -> ModelVersion:
    """register → measured evidence → publish (JSON, markdown, version tags)."""
    version = register(env.export_dir, env.registry)
    evidence = assemble_evidence(
        version=version,
        manifest=manifest_of(env.export_dir),
        champion=env.registry.alias("champion"),
        scores=scores(b_overall),
        latency=latency(b_p50),
        latency_digest=sha("subset"),
        hygiene=Hygiene(True, "all on train", BASE_REV, BASE_REV),
        swap_recorded=True,
        session_id="m7.1-test",
        measured_at=datetime(2026, 9, 25, tzinfo=UTC),
        git_sha="abc123",
    )
    publish_evidence(
        evidence,
        env.registry,
        evidence_path=env.evidence_path,
        report_path=env.evidence_path.with_suffix(".md"),
    )
    return version


def challenger_gguf(env: Env) -> str:
    return manifest_of(env.export_dir).artifact(Q4_NAME).sha256


def run_promote(
    env: Env,
    *,
    lock_fail: bool = False,
    repointer: LlamaSwapRepointer | None = None,
    real_reports: bool = True,
    docile: bool = False,
) -> Any:
    events = env.server.events
    reports_dir = env.root / "real_reports"
    if real_reports:
        write_real_report(reports_dir, "cord", "test", gguf=challenger_gguf(env))
    return promote(
        evidence_path=env.evidence_path,
        export_dir=env.export_dir,
        registry=env.registry,
        invalidator=RecordingInvalidator(events),
        lock_writer=RecordingLock(JsonModelsLockWriter(env.lock), events, fail=lock_fail),
        repointer=repointer or RecordingRepointer(env.swap, events),
        models_root=env.root / "models",
        llama_swap_config=env.swap,
        real_reports_dir=reports_dir,
        docile_present=docile,
        load_real=load_fake_reports,
    )


# --- register --------------------------------------------------------------------------------


def test_register_creates_one_version_with_sha_tags_and_is_idempotent(env: Env) -> None:
    version = register(env.export_dir, env.registry)
    again = register(env.export_dir, env.registry)
    assert version == again == "1" and len(env.server.versions) == 1
    tags = env.server.tags("1")
    manifest = manifest_of(env.export_dir)
    assert tags["export.gguf_sha256"] == manifest.artifact(Q4_NAME).sha256
    assert tags[f"artifact.{MMPROJ_NAME}.size_bytes"] == "6"
    assert tags["export.base_revision"] == BASE_REV and tags["export.run_id"] == RUN_ID
    assert env.server.versions["1"]["run_id"] == MLFLOW_RUN
    assert env.server.models["extractor_b"] == {"challenger": "1"}


def test_register_new_build_of_same_run_gets_a_new_version(env: Env, tmp_path: Path) -> None:
    register(env.export_dir, env.registry)
    other = make_export(tmp_path / "second", q4=b"other-weights")
    assert register(other, env.registry) == "2"
    assert env.server.models["extractor_b"]["challenger"] == "2"


def test_register_refuses_a_tampered_export(env: Env) -> None:
    (env.export_dir / Q4_NAME).write_bytes(b"tampered!!")
    with pytest.raises(ExportVerificationError):
        register(env.export_dir, env.registry)
    assert env.server.versions == {}


def test_registry_errors_are_typed(env: Env) -> None:
    with pytest.raises(RegistryError, match="does not exist"):
        env.registry.alias("champion")
    env.registry.ensure_model()
    with pytest.raises(RegistryError, match="not set"):
        env.registry.version_for("champion")
    down = MlflowRegistry(
        "http://mlflow.test",
        client=httpx.Client(
            base_url="http://mlflow.test",
            transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
        ),
    )
    with pytest.raises(RegistryError, match="500"):
        down.ensure_model()


# --- promote ---------------------------------------------------------------------------------


def test_promote_runs_side_effects_in_order_and_moves_champion_last(env: Env) -> None:
    version = gated(env)
    env.server.events.clear()
    result = run_promote(env)

    assert env.server.events == [
        f"invalidate:{version}",
        f"lock:{version}",
        "llama_swap",
        f"alias:champion={version}",
    ]
    assert result.applied == (STAGE_INVALIDATE, STAGE_MODELS_LOCK, STAGE_LLAMA_SWAP, STAGE_ALIAS)
    assert result.champion_before is None and result.champion_after == version

    lock = json.loads(env.lock.read_text(encoding="utf-8"))
    manifest = manifest_of(env.export_dir)
    assert lock["local_models"] == [
        {
            "registered_model": "extractor_b",
            "version": version,
            "source": "local",
            "mlflow_run_id": MLFLOW_RUN,
            "base_revision": BASE_REV,
            "gguf_sha256": manifest.artifact(Q4_NAME).sha256,
            "mmproj_sha256": manifest.artifact(MMPROJ_NAME).sha256,
            "license": "Apache-2.0",
        }
    ]
    original = yaml.safe_load((REPO_ROOT / "infra" / "llama-swap.yaml").read_text("utf-8"))
    served = yaml.safe_load(env.swap.read_text(encoding="utf-8"))
    cmd = served["models"]["extractor-b"]["cmd"]
    assert f"/models/extractor_b/{RUN_ID}/export/{Q4_NAME}" in cmd
    assert f"/models/extractor_b/{RUN_ID}/export/{MMPROJ_NAME}" in cmd
    description = served["models"]["extractor-b"]["description"]
    assert "--parallel 2" in cmd and f"v{version}" in description
    assert lock["local_models"][0]["gguf_sha256"] in description
    for name, entry in original["models"].items():  # nothing else changed
        if name != "extractor-b":
            assert served["models"][name] == entry
    assert served["routing"] == original["routing"]

    assert run_promote(env).applied == ()  # rerun: already champion, nothing redone


def test_failed_gate_blocks_every_side_effect(env: Env) -> None:
    gated(env, b_p50=19.0)  # P2: 19 s > 0.60 x 30 s
    lock_before, swap_before = env.lock.read_bytes(), env.swap.read_bytes()
    env.server.events.clear()
    with pytest.raises(GatesNotPassedError, match="P2"):
        run_promote(env)
    assert env.server.events == []
    assert env.lock.read_bytes() == lock_before and env.swap.read_bytes() == swap_before
    assert "champion" not in env.server.models["extractor_b"]


def test_edited_evidence_no_longer_matches_the_registry(env: Env) -> None:
    gated(env)
    text = env.evidence_path.read_text(encoding="utf-8").replace("abc123", "def456")
    env.evidence_path.write_text(text, encoding="utf-8")
    with pytest.raises(EvidenceMismatchError, match="rerun `gates`"):
        run_promote(env)


def test_evidence_for_another_version_is_refused(env: Env, tmp_path: Path) -> None:
    gated(env)
    register(make_export(tmp_path / "second", q4=b"newer"), env.registry)  # challenger -> v2
    with pytest.raises(EvidenceMismatchError):
        run_promote(env)


def test_champion_moved_since_measurement_is_stale(env: Env) -> None:
    version = gated(env)
    env.registry.set_alias("champion", ModelVersion("7"))
    env.server.events.clear()
    with pytest.raises(ChampionChangedError):
        run_promote(env)
    assert env.server.events == [] and version != "7"


def test_lock_failure_leaves_the_alias_unmoved(env: Env) -> None:
    gated(env)
    swap_before = env.swap.read_bytes()
    with pytest.raises(PromotionApplyError) as info:
        run_promote(env, lock_fail=True)
    assert info.value.failed_stage == STAGE_MODELS_LOCK
    assert info.value.applied == (STAGE_INVALIDATE,)
    assert "champion" not in env.server.models["extractor_b"]
    assert env.swap.read_bytes() == swap_before


def test_alias_failure_reports_every_applied_step(env: Env) -> None:
    gated(env)
    env.server.fail_alias = True
    with pytest.raises(PromotionApplyError) as info:
        run_promote(env)
    assert info.value.failed_stage == STAGE_ALIAS
    assert info.value.applied == (STAGE_INVALIDATE, STAGE_MODELS_LOCK, STAGE_LLAMA_SWAP)


def test_evidence_with_mismatched_latency_counts_is_refused(env: Env) -> None:
    run = LatencyRun(
        a=LatencyStats(30.0, 60.0, 199),
        b=LatencyStats(10.0, 20.0, 200),
        peak_vram_gb=9.0,
        vram_samples=5,
    )
    with pytest.raises(GateEvidenceError):
        assemble_evidence(
            version=ModelVersion("1"),
            manifest=manifest_of(env.export_dir),
            champion=None,
            scores=scores(),
            latency=run,
            latency_digest=sha("subset"),
            hygiene=Hygiene(True, "ok", BASE_REV, BASE_REV),
            swap_recorded=True,
            session_id="s",
            measured_at=datetime(2026, 9, 25, tzinfo=UTC),
            git_sha="abc",
        )


# --- adapters --------------------------------------------------------------------------------


def entry(version: str = "1", gguf: str = "q4") -> ModelsLockEntry:
    return ModelsLockEntry(
        version=ModelVersion(version),
        mlflow_run_id=MLFLOW_RUN,
        base_revision=BASE_REV,
        gguf_sha256=sha(gguf),
        mmproj_sha256=sha("mmproj"),
    )


def test_lock_writer_is_idempotent_and_refuses_conflicts(tmp_path: Path) -> None:
    lock = tmp_path / "models.lock.json"
    shutil.copy(REPO_ROOT / "infra" / "models.lock.json", lock)
    hub_models = json.loads(lock.read_text(encoding="utf-8"))["models"]
    writer = JsonModelsLockWriter(lock)
    writer.record(entry())
    once = lock.read_bytes()
    writer.record(entry())
    assert lock.read_bytes() == once
    with pytest.raises(LockConflictError):
        writer.record(entry(gguf="other"))
    writer.record(entry("2", "q4-v2"))
    data = json.loads(lock.read_text(encoding="utf-8"))
    assert [e["version"] for e in data["local_models"]] == ["1", "2"]
    assert data["models"] == hub_models  # Hub entries untouched
    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    with pytest.raises(LockConflictError):
        JsonModelsLockWriter(tmp_path / "bad.json").record(entry())


def test_calibrator_marker_is_written(tmp_path: Path) -> None:
    path = tmp_path / "trust" / "calibrator_invalidation.json"
    clock = lambda: datetime(2026, 9, 25, tzinfo=UTC)  # noqa: E731
    FileCalibratorInvalidator(path, clock=clock).invalidate(ModelVersion("3"), "why")
    marker = json.loads(path.read_text(encoding="utf-8"))
    assert marker == {
        "stale": True,
        "extractor_b_champion": "3",
        "reason": "why",
        "invalidated_at": "2026-09-25T00:00:00+00:00",
    }


def test_repointer_is_idempotent_and_refuses_an_unexpected_config(tmp_path: Path) -> None:
    swap = tmp_path / "llama-swap.yaml"
    shutil.copy(REPO_ROOT / "infra" / "llama-swap.yaml", swap)
    repointer = LlamaSwapRepointer(swap)
    kwargs = {
        "model_path": "/models/x/q.gguf",
        "mmproj_path": "/models/x/m.gguf",
        "description": "d",
    }
    repointer.repoint(**kwargs)
    once = swap.read_bytes()
    repointer.repoint(**kwargs)
    assert swap.read_bytes() == once
    challenger = yaml.safe_load(once)["models"]["extractor-b-challenger"]["cmd"]
    assert "/models/x/q.gguf" not in challenger
    with pytest.raises(LlamaSwapConfigError, match="no `missing:`"):
        LlamaSwapRepointer(swap, alias="missing").repoint(**kwargs)
    swap.write_text("models:\n  extractor-b:\n    cmd: llama-server\n", encoding="utf-8")
    with pytest.raises(LlamaSwapConfigError, match="expected one line"):
        repointer.repoint(**kwargs)


def test_container_path_maps_models_dir(tmp_path: Path) -> None:
    root = tmp_path / "models"
    assert (
        container_path(root / "extractor_b" / "r" / "q.gguf", root)
        == "/models/extractor_b/r/q.gguf"
    )
    with pytest.raises(LlamaSwapConfigError):
        container_path(tmp_path / "elsewhere.gguf", root)


def test_eval_b_reuses_the_extraction_harness_on_the_challenger_alias() -> None:
    args = parse_args(["eval-b", "--export-dir", "x", "--limit", "5"])
    argv = eval_b_argv(args)
    assert argv[:4] == ["--extractor", "b", "--model", "extractor-b-challenger"]
    assert argv[argv.index("--rows") + 1] == str(Path("logs/extractor_b_test.jsonl"))
    assert argv[argv.index("--report") + 1] == str(Path("reports/extractor_b_test.md"))
    assert argv[-2:] == ["--limit", "5"]
    assert "--parser-id" not in argv


def test_eval_b_passes_a_pinned_parse_snapshot_through() -> None:
    args = parse_args(["eval-b", "--export-dir", "x", "--parser-id", "a8832889e65c7fd5"])
    argv = eval_b_argv(args)
    assert argv[argv.index("--parser-id") + 1] == "a8832889e65c7fd5"


# --- review fixes: served build, row identity, second compare-and-set ------------------------


class MovingRepointer(RecordingRepointer):
    """Moves `champion` behind promote's back while the llama-swap step runs."""

    def __init__(self, path: Path, events: list[str], registry: MlflowRegistry) -> None:
        super().__init__(path, events)
        self.registry = registry

    def repoint(self, *, model_path: str, mmproj_path: str, description: str) -> None:
        super().repoint(model_path=model_path, mmproj_path=mmproj_path, description=description)
        self.registry.set_alias("champion", ModelVersion("9"))


def test_champion_moved_during_apply_is_not_overwritten(env: Env) -> None:
    gated(env)
    moving = MovingRepointer(env.swap, env.server.events, env.registry)
    with pytest.raises(ChampionMovedDuringApplyError) as info:
        run_promote(env, repointer=moving)
    assert isinstance(info.value, ChampionChangedError)
    assert info.value.applied == (STAGE_INVALIDATE, STAGE_MODELS_LOCK, STAGE_LLAMA_SWAP)
    assert env.server.models["extractor_b"]["champion"] == "9"


def test_challenger_entry_must_serve_the_export(env: Env, tmp_path: Path) -> None:
    root = env.root / "models"
    require_challenger_serves(env.export_dir, llama_swap_config=env.swap, models_root=root)
    other = tmp_path / "models" / "extractor_b" / "b-other" / "export"
    with pytest.raises(GateEvidenceError, match="not the export"):
        require_challenger_serves(
            other, llama_swap_config=env.swap, models_root=tmp_path / "models"
        )
    with pytest.raises(GateEvidenceError):
        require_challenger_serves(env.export_dir, alias="absent", llama_swap_config=env.swap)


def test_promote_refuses_when_the_challenger_serves_another_build(env: Env) -> None:
    gated(env)
    text = env.swap.read_text(encoding="utf-8").replace(RUN_ID, "b-other-run")
    env.swap.write_text(text, encoding="utf-8")
    env.server.events.clear()
    with pytest.raises(GateEvidenceError, match="not the export"):
        run_promote(env)
    assert env.server.events == []


def _rows_file(path: Path, shas: list[str], identity: RowIdentity | None) -> Path:
    fields = {"invoice_number": {"pred": "INV-1"}}
    with path.open("w", encoding="utf-8") as fh:
        for doc in shas:
            row: dict[str, Any] = {"sha": doc, "valid_json": True, "fields": fields}
            if identity is not None:
                row.update(identity.as_row_fields())
            fh.write(json.dumps(row) + "\n")
    return path


def _gold_dir(tmp_path: Path, shas: list[str]) -> Path:
    data = tmp_path / "test"
    data.mkdir()
    for doc in shas:
        labels = {"labels": [{"field": "invoice_number", "value": "INV-1"}]}
        (data / f"{doc}.labels.json").write_text(json.dumps(labels), encoding="utf-8")
        (data / f"{doc}.truth.json").write_text(json.dumps({"arabic_first": False}), "utf-8")
    return data


def test_row_files_must_carry_the_expected_identity(tmp_path: Path) -> None:
    shas = [sha("d1"), sha("d2")]
    data = _gold_dir(tmp_path, shas)
    a_id = RowIdentity(model="extractor-a", gguf_sha256=sha("a"), prompt_version="pa")
    b_id = RowIdentity(model="extractor-b-challenger", gguf_sha256=sha("b"), prompt_version="pb")
    a_rows = _rows_file(tmp_path / "a.jsonl", shas, a_id)
    b_rows = _rows_file(tmp_path / "b.jsonl", shas, b_id)
    kwargs = {"data_dir": data, "test_ids": shas, "a_identity": a_id, "b_identity": b_id}

    result = score_row_files(a_rows=a_rows, b_rows=b_rows, **kwargs)  # type: ignore[arg-type]
    assert result.a_overall == result.b_overall == 1.0 and result.b_valid_json == 1.0
    assert result.b_identity == b_id and len(result.a_rows_sha256) == 64

    stale = b_id.model_copy(update={"gguf_sha256": sha("older-build")})
    _rows_file(b_rows, shas, stale)
    with pytest.raises(GateEvidenceError, match="from another run"):
        score_row_files(a_rows=a_rows, b_rows=b_rows, **kwargs)  # type: ignore[arg-type]
    _rows_file(a_rows, shas, None)  # legacy rows without identity fields
    with pytest.raises(GateEvidenceError, match=r"a\.jsonl"):
        score_row_files(a_rows=a_rows, b_rows=b_rows, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(GateEvidenceError, match="do not exist"):
        score_row_files(a_rows=tmp_path / "none.jsonl", b_rows=b_rows, **kwargs)  # type: ignore[arg-type]


def test_a_b_and_p2_must_read_one_parse_snapshot(tmp_path: Path) -> None:
    """REVIEW.md P5-02: rows parsed by another build than the P2 run's snapshot are refused."""
    shas = [sha("d1"), sha("d2")]
    data = _gold_dir(tmp_path, shas)
    a_id = RowIdentity(model="extractor-a", gguf_sha256=sha("a"), prompt_version="pa")
    b_id = RowIdentity(model="extractor-b-challenger", gguf_sha256=sha("b"), prompt_version="pb")
    a_rows = _rows_file(tmp_path / "a.jsonl", shas, a_id)
    b_rows = _rows_file(tmp_path / "b.jsonl", shas, b_id)

    def stamp(path: Path, parser: str) -> None:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text(
            "".join(json.dumps({**r, "parser_identity": parser}) + "\n" for r in rows), "utf-8"
        )

    stamp(a_rows, "a8832889e65c7fd5")
    stamp(b_rows, "a8832889e65c7fd5")
    kwargs = {"data_dir": data, "test_ids": shas, "a_identity": a_id, "b_identity": b_id}
    scores = score_row_files(
        a_rows=a_rows,
        b_rows=b_rows,
        parser_id="a8832889e65c7fd5",
        **kwargs,  # type: ignore[arg-type]
    )
    assert scores.parser_identity == "a8832889e65c7fd5"

    stamp(b_rows, "66f415c5490d9e5b")
    with pytest.raises(GateEvidenceError, match=r"b\.jsonl were parsed by 66f415c5490d9e5b"):
        score_row_files(
            a_rows=a_rows,
            b_rows=b_rows,
            parser_id="a8832889e65c7fd5",
            **kwargs,  # type: ignore[arg-type]
        )


def test_expected_identities_use_the_served_a_gguf_and_the_export(env: Env) -> None:
    a_id, b_id = expected_identities(
        manifest_of(env.export_dir),
        llama_swap_config=env.swap,
        models_lock=env.lock,
        models_root=env.root / "models",
    )
    lock = json.loads(env.lock.read_text(encoding="utf-8"))
    a_entry = next(m for m in lock["models"] if m["intent"] == "extractor_a")
    gguf = next(f for f in a_entry["files"] if f["filename"].endswith("Q4_K_XL.gguf"))
    assert a_id.model == "extractor-a" and a_id.gguf_sha256 == gguf["sha256"]
    assert b_id.gguf_sha256 == manifest_of(env.export_dir).artifact(Q4_NAME).sha256
    assert a_id.prompt_version != b_id.prompt_version


# --- ADR-0019 §9.4: real hold-out reports ------------------------------------------------------


def test_missing_real_holdouts_block_every_side_effect(env: Env) -> None:
    gated(env)
    lock_before, swap_before = env.lock.read_bytes(), env.swap.read_bytes()
    env.server.events.clear()
    with pytest.raises(RealHoldoutsMissingError, match="cord/test"):
        run_promote(env, real_reports=False)
    assert env.server.events == []
    assert env.lock.read_bytes() == lock_before and env.swap.read_bytes() == swap_before
    assert "champion" not in env.server.models["extractor_b"]


def test_docile_val_is_required_once_docile_is_fetched(env: Env) -> None:
    gated(env)
    env.server.events.clear()
    with pytest.raises(RealHoldoutsMissingError, match="docile/val"):
        run_promote(env, docile=True)
    assert env.server.events == []
    write_real_report(env.root / "real_reports", "docile", "val", gguf=challenger_gguf(env))
    result = run_promote(env, docile=True)
    assert [(h.set, h.split) for h in result.real_holdouts] == [("cord", "test"), ("docile", "val")]
    assert result.applied[-1] == STAGE_ALIAS


def test_promotion_records_the_real_holdouts_without_judging_them(env: Env) -> None:
    gated(env)
    result = run_promote(env)
    assert [(h.set, h.split, h.n_docs) for h in result.real_holdouts] == [("cord", "test", 100)]


def test_docile_present_reads_the_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    assert docile_present(manifest) is False
    entry = {"source": "s", "version": "v", "license": "l", "files": []}
    body = {"generated_at": "2026-09-29T00:00:00Z", "datasets": {}}
    body["datasets"] = {"docile": {**entry, "status": "deferred", "blocked": "token"}}
    manifest.write_text(json.dumps(body), encoding="utf-8")
    assert docile_present(manifest) is False
    body["datasets"] = {"docile": {**entry, "status": "present"}}
    manifest.write_text(json.dumps(body), encoding="utf-8")
    assert docile_present(manifest) is True
