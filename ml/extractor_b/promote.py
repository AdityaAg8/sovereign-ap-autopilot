"""Extractor B registration, gate run and promotion (ADR-0009 §1.8 build item 10, ADR-0015 §5):
`register` makes the export a version of MLflow model `extractor_b` with alias `challenger`;
`eval-b` runs the Extractor A harness on UAE test through `extractor-b-challenger`; `gates`
measures P1-P5; `promote` moves `champion` only when every gate passed, in a fixed order.

Usage: python -m ml.extractor_b.promote {register,eval-b,gates,promote} [options]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import yaml

from apx.extract.interfaces import ExtractorId
from evals.extraction_a import load_rows, prompt_builder_for, run_identity
from evals.extraction_f1 import (
    RowsIncompleteError,
    ids_digest,
    score_rows,
    uae_gold,
    valid_json_rate,
)
from evals.extraction_identity import (
    RowIdentity,
    RowIdentityError,
    ServedModelError,
    prompt_version,
    require_identity,
    served_paths,
)
from ml.extractor_b.checkpoints import sha256_file
from ml.extractor_b.export import (
    EXPORT_DIR_NAME,
    MMPROJ_NAME,
    Q4_NAME,
    ExportManifest,
    latest_run_dir,
    verify_export,
)
from ml.extractor_b.gates import (
    A_ALIAS,
    CHALLENGER_ALIAS,
    GateEvidence,
    GateEvidenceError,
    Hygiene,
    LatencyRun,
    check_hygiene,
    evaluate_gates,
    evidence_sha256,
    extract_group_swaps,
    gate_tags,
    latency_subset,
    load_evidence,
    render_markdown,
    write_evidence,
)
from ml.promotion.errors import ChampionChangedError, PromotionApplyError, PromotionError
from ml.promotion.evidence import ExtractorGateReport
from ml.promotion.retrain import (
    STAGE_ALIAS,
    STAGE_INVALIDATE,
    STAGE_MODELS_LOCK,
    CalibratorInvalidator,
    ModelsLockEntry,
    ModelsLockWriter,
    ModelVersion,
    RegistryAlias,
)

REGISTERED_MODEL: Final[str] = "extractor_b"
STAGE_LLAMA_SWAP: Final[str] = "llama_swap"
SERVED_ALIAS: Final[str] = "extractor-b"
DEFAULT_MLFLOW_URI: Final[str] = "http://127.0.0.1:5000"
DEFAULT_RUNS_DIR: Final[Path] = Path("models/extractor_b")
DEFAULT_DATA_DIR: Final[Path] = Path("data/uae/test")
DEFAULT_TRAIN_DATA_DIR: Final[Path] = Path("data/ml/extractor_b")
DEFAULT_A_ROWS: Final[Path] = Path("logs/extraction_a.adr0012.jsonl")
DEFAULT_B_ROWS: Final[Path] = Path("logs/extractor_b_test.jsonl")
B_TEST_REPORT: Final[Path] = Path("reports/extractor_b_test.md")
EVIDENCE_PATH: Final[Path] = Path("reports/extractor_b_gates.json")
GATES_REPORT: Final[Path] = Path("reports/extractor_b_gates.md")
MODELS_LOCK: Final[Path] = Path("infra/models.lock.json")
LLAMA_SWAP_CONFIG: Final[Path] = Path("infra/llama-swap.yaml")
CALIBRATOR_INVALIDATION: Final[Path] = Path("data/trust/calibrator_invalidation.json")
LOCAL_MODELS_KEY: Final[str] = "local_models"
TAG_GGUF: Final[str] = "export.gguf_sha256"
TAG_MMPROJ: Final[str] = "export.mmproj_sha256"
TAG_EVIDENCE: Final[str] = "gates.evidence_sha256"


class RegistryError(PromotionError):
    """The MLflow registry refused a call, is unreachable, or holds an unexpected state."""


class ExportVerificationError(PromotionError):
    """The export's manifest is missing or a file no longer matches its sha256 and size."""


class GatesNotPassedError(PromotionError):
    """`promote` was asked to move `champion` for a version whose gate evidence has a FAIL."""


class EvidenceMismatchError(PromotionError):
    """The gate evidence names another version, export or file digest than the registry."""


class LockConflictError(PromotionError):
    """models.lock already holds a different entry for this version."""


class ChampionMovedDuringApplyError(ChampionChangedError):
    """The `champion` alias moved while promotion side effects ran; the alias was not set.
    `applied` names the steps already done (recovery: ADR-0015 §5 "Partial apply")."""

    def __init__(self, *, evaluated: str, current: str, applied: tuple[str, ...]) -> None:
        PromotionError.__init__(
            self,
            f"champion alias moved from {evaluated} to {current} during promotion; alias not "
            f"set; steps already applied: {', '.join(applied) or 'none'}",
        )
        self.evaluated = evaluated
        self.current = current
        self.applied = applied


class LlamaSwapConfigError(PromotionError):
    """infra/llama-swap.yaml does not have the expected `extractor-b` block to repoint."""


# --- MLflow registry (REST) ------------------------------------------------------------------


class MlflowRegistry:
    """Registered model `extractor_b` over the MLflow REST API (the host venv has no mlflow
    package; the server is ghcr.io/mlflow/mlflow v3.7.0). Satisfies
    `ml.promotion.retrain.ModelAliasRegistry`. Every non-2xx answer raises `RegistryError`."""

    def __init__(
        self, base_uri: str, *, client: httpx.Client | None = None, name: str = REGISTERED_MODEL
    ) -> None:
        self.name = name
        self._http = client or httpx.Client(base_url=base_uri.rstrip("/"), timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        try:
            response = self._http.request(
                method, f"/api/2.0/mlflow/{path}", params=params, json=body
            )
        except httpx.HTTPError as exc:
            raise RegistryError(f"MLflow {method} {path} failed: {exc!r}") from exc
        if response.status_code == 404 and missing_ok:
            return None
        if response.status_code != 200:
            raise RegistryError(
                f"MLflow {method} {path} -> {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RegistryError(f"MLflow {method} {path} returned {type(payload).__name__}")
        return payload

    def _model(self) -> dict[str, Any] | None:
        payload = self._call(
            "GET", "registered-models/get", params={"name": self.name}, missing_ok=True
        )
        return None if payload is None else dict(payload["registered_model"])

    def ensure_model(self) -> None:
        if self._model() is None:
            self._call("POST", "registered-models/create", body={"name": self.name})

    def versions_for_run(self, run_id: str) -> list[dict[str, Any]]:
        payload = self._call(
            "GET",
            "model-versions/search",
            params={"filter": f"name='{self.name}' and run_id='{run_id}'"},
        )
        return list((payload or {}).get("model_versions", []))

    def create_version(self, *, source: str, run_id: str, tags: Mapping[str, str]) -> str:
        payload = self._call(
            "POST",
            "model-versions/create",
            body={
                "name": self.name,
                "source": source,
                "run_id": run_id,
                "tags": [{"key": k, "value": v} for k, v in sorted(tags.items())],
            },
        )
        return str((payload or {})["model_version"]["version"])

    def version_tags(self, version: str) -> dict[str, str]:
        payload = self._call(
            "GET", "model-versions/get", params={"name": self.name, "version": version}
        )
        tags = (payload or {})["model_version"].get("tags", [])
        return {str(t["key"]): str(t["value"]) for t in tags}

    def set_version_tags(self, version: str, tags: Mapping[str, str]) -> None:
        for key, value in sorted(tags.items()):
            self._call(
                "POST",
                "model-versions/set-tag",
                body={"name": self.name, "version": version, "key": key, "value": value},
            )

    def alias(self, alias: RegistryAlias) -> ModelVersion | None:
        model = self._model()
        if model is None:
            raise RegistryError(f"registered model {self.name!r} does not exist; run register")
        for item in model.get("aliases", []):
            if item.get("alias") == alias:
                return ModelVersion(str(item["version"]))
        return None

    def version_for(self, alias: RegistryAlias) -> ModelVersion:
        version = self.alias(alias)
        if version is None:
            raise RegistryError(f"alias {alias!r} of {self.name!r} is not set")
        return version

    def set_alias(self, alias: RegistryAlias, version: ModelVersion) -> None:
        self._call(
            "POST",
            "registered-models/alias",
            body={"name": self.name, "alias": alias, "version": version},
        )


# --- register --------------------------------------------------------------------------------


def export_tags(manifest: ExportManifest, export_dir: Path) -> dict[str, str]:
    """Model-version tags: provenance plus every artifact's sha256 and size."""
    tags = {
        "export.run_id": manifest.run_id,
        "export.step": str(manifest.step),
        "export.base_revision": manifest.base_revision,
        "export.quant": manifest.quant,
        "export.llama_cpp_commit": manifest.llama_cpp_commit,
        "export.adapter_sha256": manifest.adapter_sha256,
        "export.dir": export_dir.as_posix(),
        "export.created_at": manifest.created_at.isoformat(),
        TAG_GGUF: manifest.artifact(Q4_NAME).sha256,
        TAG_MMPROJ: manifest.artifact(MMPROJ_NAME).sha256,
    }
    for item in manifest.artifacts:
        tags[f"artifact.{item.name}.sha256"] = item.sha256
        tags[f"artifact.{item.name}.size_bytes"] = str(item.size_bytes)
    return tags


def verified_manifest(
    export_dir: Path, verify: Callable[[Path], ExportManifest | None] = verify_export
) -> ExportManifest:
    manifest = verify(export_dir)
    if manifest is None:
        raise ExportVerificationError(
            f"{export_dir}: export manifest missing or an artifact no longer matches its sha256"
        )
    return manifest


def register(
    export_dir: Path,
    registry: MlflowRegistry,
    *,
    verify: Callable[[Path], ExportManifest | None] = verify_export,
) -> ModelVersion:
    """Idempotent: an existing version of the same MLflow run with the same Q4_K_M sha256 is
    reused (its tags refreshed); otherwise a new version is created. Then alias `challenger`."""
    manifest = verified_manifest(export_dir, verify)
    tags = export_tags(manifest, export_dir)
    registry.ensure_model()
    matches = [
        str(v["version"])
        for v in registry.versions_for_run(manifest.mlflow_run_id)
        if {str(t["key"]): str(t["value"]) for t in v.get("tags", [])}.get(TAG_GGUF)
        == tags[TAG_GGUF]
    ]
    if len(matches) > 1:
        raise RegistryError(f"versions {matches} all hold export {tags[TAG_GGUF]}")
    if matches:
        version = ModelVersion(matches[0])
        registry.set_version_tags(version, tags)
    else:
        version = ModelVersion(
            registry.create_version(
                source=f"runs:/{manifest.mlflow_run_id}/export",
                run_id=manifest.mlflow_run_id,
                tags=tags,
            )
        )
    registry.set_alias("challenger", version)
    return version


def challenger_matching_export(registry: MlflowRegistry, manifest: ExportManifest) -> ModelVersion:
    """The `challenger` version, which must be this export (same Q4_K_M and mmproj sha256)."""
    version = registry.version_for("challenger")
    tags = registry.version_tags(version)
    want = {
        TAG_GGUF: manifest.artifact(Q4_NAME).sha256,
        TAG_MMPROJ: manifest.artifact(MMPROJ_NAME).sha256,
    }
    for key, value in want.items():
        if tags.get(key) != value:
            raise EvidenceMismatchError(
                f"challenger version {version} has {key}={tags.get(key)!r}, export has {value}"
            )
    return version


# --- promotion side-effect adapters ----------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class JsonModelsLockWriter:
    """`ModelsLockWriter` for `infra/models.lock.json`: local builds live under the top-level
    `local_models` list (Hub entries under `models` are what download/smoke iterate), keyed by
    registered model + version. Re-recording the same entry is a no-op; a different entry for the
    same version raises `LockConflictError`."""

    def __init__(self, path: Path = MODELS_LOCK, registered_model: str = REGISTERED_MODEL) -> None:
        self._path = path
        self._registered_model = registered_model

    def record(self, entry: ModelsLockEntry) -> None:
        lock = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(lock, dict) or not isinstance(lock.get("models"), list):
            raise LockConflictError(f"{self._path} is not a models lock")
        row = {"registered_model": self._registered_model, **entry.model_dump(mode="json")}
        rows: list[dict[str, Any]] = list(lock.get(LOCAL_MODELS_KEY, []))
        for existing in rows:
            same = (existing.get("registered_model"), existing.get("version")) == (
                self._registered_model,
                entry.version,
            )
            if same and existing == row:
                return
            if same:
                raise LockConflictError(
                    f"{self._path} already has a different entry for "
                    f"{self._registered_model} v{entry.version}"
                )
        lock[LOCAL_MODELS_KEY] = [*rows, row]
        _atomic_write_text(self._path, json.dumps(lock, indent=2) + "\n")


class FileCalibratorInvalidator:
    """`CalibratorInvalidator`: writes the durable stale marker the M7.4/M7.5 cal job clears
    after refitting on the new champion (ADR-0009 §5, ADR-0015 §5 step 2)."""

    def __init__(
        self, path: Path = CALIBRATOR_INVALIDATION, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._path = path
        self._clock = clock or (lambda: datetime.now(UTC))

    def invalidate(self, new_champion: ModelVersion, reason: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        marker = {
            "stale": True,
            "extractor_b_champion": new_champion,
            "reason": reason,
            "invalidated_at": self._clock().isoformat(),
        }
        _atomic_write_text(self._path, json.dumps(marker, indent=2) + "\n")


class LlamaSwapRepointer:
    """Rewrites the `extractor-b` entry of infra/llama-swap.yaml to serve an export: only its
    `description`, `--model` and `--mmproj` lines change, every other byte (flags, comments, the
    challenger entry, groups) stays. Idempotent. llama-swap reads its config at start, so the
    change is live after `docker compose restart llama-swap`."""

    def __init__(self, path: Path = LLAMA_SWAP_CONFIG, alias: str = SERVED_ALIAS) -> None:
        self._path = path
        self._alias = alias

    def repoint(self, *, model_path: str, mmproj_path: str, description: str) -> None:
        lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        start = next(
            (i for i, line in enumerate(lines) if line.rstrip() == f"  {self._alias}:"), None
        )
        if start is None:
            raise LlamaSwapConfigError(f"{self._path} has no `{self._alias}:` model entry")
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip(" ")) <= 2
            ),
            len(lines),
        )
        rules = (
            (re.compile(r"^(\s+description: ).*$"), json.dumps(description, ensure_ascii=False)),
            (re.compile(r"^(\s+--model )\S+\s*$"), model_path),
            (re.compile(r"^(\s+--mmproj )\S+\s*$"), mmproj_path),
        )
        for pattern, value in rules:
            hits = [i for i in range(start, end) if pattern.match(lines[i].rstrip("\n"))]
            if len(hits) != 1:
                raise LlamaSwapConfigError(
                    f"`{self._alias}` in {self._path}: expected one line matching "
                    f"{pattern.pattern!r}, found {len(hits)}"
                )
            i = hits[0]
            prefix = pattern.match(lines[i].rstrip("\n"))
            assert prefix is not None
            lines[i] = f"{prefix.group(1)}{value}\n"
        text = "".join(lines)
        try:
            served = yaml.safe_load(text)["models"][self._alias]["cmd"]
        except (yaml.YAMLError, KeyError, TypeError) as exc:
            raise LlamaSwapConfigError(f"repointed {self._path} no longer parses: {exc}") from exc
        if model_path not in served or mmproj_path not in served:
            raise LlamaSwapConfigError(f"repointed `{self._alias}` cmd lacks the export paths")
        _atomic_write_text(self._path, text)


def container_path(path: Path, models_root: Path = Path("models")) -> str:
    """Host `models/...` path → the `/models/...` path llama-swap mounts read-only."""
    try:
        return "/models/" + path.relative_to(models_root).as_posix()
    except ValueError as exc:
        raise LlamaSwapConfigError(f"{path} is not under {models_root}/") from exc


# --- promote ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromotionResult:
    champion_before: ModelVersion | None
    champion_after: ModelVersion
    applied: tuple[str, ...]


def promote(
    *,
    evidence_path: Path,
    export_dir: Path,
    registry: MlflowRegistry,
    invalidator: CalibratorInvalidator,
    lock_writer: ModelsLockWriter,
    repointer: LlamaSwapRepointer,
    verify: Callable[[Path], ExportManifest | None] = verify_export,
    models_root: Path = Path("models"),
    llama_swap_config: Path = LLAMA_SWAP_CONFIG,
) -> PromotionResult:
    """Re-read and re-check the gate evidence, then (ADR-0015 §5 order, plus the serving
    repoint before the alias commit): compare-and-set on `champion` → invalidate calibrator →
    models.lock entry → repoint llama-swap `extractor-b` → alias `champion`. A failure after
    the check raises `PromotionApplyError` with the steps already applied; the alias moves last,
    so it never points at a build that is not locked and served."""
    evidence = load_evidence(evidence_path)
    failed = [g.gate for g in evidence.results if not g.passed]
    if failed or not evidence.all_passed:
        raise GatesNotPassedError(f"gates failed: {', '.join(failed) or 'incomplete'}")
    manifest = verified_manifest(export_dir, verify)
    require_challenger_serves(
        export_dir,
        alias=evidence.challenger_alias,
        llama_swap_config=llama_swap_config,
        models_root=models_root,
    )
    challenger = challenger_matching_export(registry, manifest)
    if challenger != evidence.model_version:
        raise EvidenceMismatchError(
            f"evidence is for version {evidence.model_version}, challenger is {challenger}"
        )
    if manifest.artifact(Q4_NAME).sha256 != evidence.gguf_sha256:
        raise EvidenceMismatchError("evidence was measured on another Q4_K_M build")
    tags = registry.version_tags(challenger)
    if tags.get(TAG_EVIDENCE) != evidence_sha256(evidence_path):
        raise EvidenceMismatchError(
            f"{evidence_path} sha256 does not match tag {TAG_EVIDENCE} of version {challenger}; "
            "rerun `gates`"
        )

    current = registry.alias("champion")
    if current == challenger:
        return PromotionResult(current, challenger, ())  # already promoted, nothing to redo
    if current != evidence.champion_at_measurement:
        raise ChampionChangedError(
            evaluated=evidence.champion_at_measurement or "<unset>", current=current or "<unset>"
        )
    before = current or "<none>"
    applied: list[str] = []
    stage = STAGE_INVALIDATE
    try:
        invalidator.invalidate(challenger, f"extractor_b champion {before} -> {challenger} (M7.1)")
        applied.append(stage)
        stage = STAGE_MODELS_LOCK
        lock_writer.record(
            ModelsLockEntry(
                version=challenger,
                mlflow_run_id=manifest.mlflow_run_id,
                base_revision=manifest.base_revision,
                gguf_sha256=manifest.artifact(Q4_NAME).sha256,
                mmproj_sha256=manifest.artifact(MMPROJ_NAME).sha256,
            )
        )
        applied.append(stage)
        stage = STAGE_LLAMA_SWAP
        repointer.repoint(
            model_path=container_path(export_dir / Q4_NAME, models_root),
            mmproj_path=container_path(export_dir / MMPROJ_NAME, models_root),
            description=(
                f"PRD §4 extraction stage: {REGISTERED_MODEL} v{challenger} (run "
                f"{manifest.run_id}, Q4_K_M sha256 {manifest.artifact(Q4_NAME).sha256}), "
                "promoted by ml.extractor_b.promote after P1-P5 passed "
                "(reports/extractor_b_gates.md)."
            ),
        )
        applied.append(stage)
        stage = STAGE_ALIAS
        moved = registry.alias("champion")
        if moved != current:  # compare-and-set again right before the commit point (M2)
            raise ChampionMovedDuringApplyError(
                evaluated=before, current=moved or "<unset>", applied=tuple(applied)
            )
        registry.set_alias("champion", challenger)
        applied.append(stage)
    except ChampionMovedDuringApplyError:
        raise
    except Exception as exc:
        raise PromotionApplyError(
            champion=before,
            challenger=challenger,
            failed_stage=stage,
            applied=tuple(applied),
            cause=exc,
        ) from exc
    return PromotionResult(current, challenger, tuple(applied))


# --- gates: evidence assembly ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UaeScores:
    """P1/P3 inputs from the A and B rows (evals.extraction_f1)."""

    a_overall: float
    a_en: float
    a_ar: float
    b_overall: float
    b_en: float
    b_ar: float
    b_valid_json: float
    n: int
    n_en: int
    n_ar: int
    test_digest: str
    a_rows_sha256: str
    b_rows_sha256: str
    a_identity: RowIdentity
    b_identity: RowIdentity


def require_challenger_serves(
    export_dir: Path,
    *,
    alias: str = CHALLENGER_ALIAS,
    llama_swap_config: Path = LLAMA_SWAP_CONFIG,
    models_root: Path = Path("models"),
) -> None:
    """The llama-swap `alias` entry must serve exactly this export's Q4_K_M and mmproj, so the
    build that is measured is the build that is registered and promoted (review H1)."""
    try:
        model, mmproj = served_paths(llama_swap_config, alias)
        want = (
            container_path(export_dir / Q4_NAME, models_root),
            container_path(export_dir / MMPROJ_NAME, models_root),
        )
    except (ServedModelError, LlamaSwapConfigError) as exc:
        raise GateEvidenceError(str(exc)) from exc
    if (model, mmproj) != want:
        raise GateEvidenceError(
            f"`{alias}` in {llama_swap_config} serves {model} + {mmproj}, not the export "
            f"{want[0]} + {want[1]}; point it at the export or pass --export-dir"
        )


def expected_identities(
    manifest: ExportManifest,
    *,
    b_alias: str = CHALLENGER_ALIAS,
    llama_swap_config: Path = LLAMA_SWAP_CONFIG,
    models_lock: Path = MODELS_LOCK,
    models_root: Path = Path("models"),
) -> tuple[RowIdentity, RowIdentity]:
    """(A, B) row identities the gates accept: A as served now (models.lock sha256 of the
    extractor-a GGUF), B as this export's Q4_K_M; each with its current prompt version."""
    try:
        a = run_identity(
            ExtractorId.A,
            A_ALIAS,
            llama_swap_config=llama_swap_config,
            models_lock=models_lock,
            models_root=models_root,
        )
    except ServedModelError as exc:
        raise GateEvidenceError(str(exc)) from exc
    b = RowIdentity(
        model=b_alias,
        gguf_sha256=manifest.artifact(Q4_NAME).sha256,
        prompt_version=prompt_version(prompt_builder_for(ExtractorId.B)),
    )
    return a, b


def score_row_files(
    *,
    a_rows: Path,
    b_rows: Path,
    data_dir: Path,
    test_ids: Sequence[str],
    a_identity: RowIdentity,
    b_identity: RowIdentity,
) -> UaeScores:
    """P1/P3 inputs. Every row of each file must carry the expected identity (review H2/M1)."""
    scored = {}
    loaded = {}
    for name, path, identity in (("A", a_rows, a_identity), ("B", b_rows, b_identity)):
        if not path.is_file():
            raise GateEvidenceError(f"Extractor {name} rows {path} do not exist")
        rows = load_rows(path)
        try:
            require_identity(rows, identity, path)
            scored[name] = score_rows(rows, data_dir, test_ids, f"Extractor {name} rows {path}")
        except (RowIdentityError, RowsIncompleteError) as exc:
            raise GateEvidenceError(str(exc)) from exc
        loaded[name] = rows
    a, b = scored["A"], scored["B"]
    langs = [doc.lang for doc in uae_gold(data_dir, test_ids)]
    return UaeScores(
        a_overall=a.overall.f1,
        a_en=a.en.f1,
        a_ar=a.ar.f1,
        b_overall=b.overall.f1,
        b_en=b.en.f1,
        b_ar=b.ar.f1,
        b_valid_json=valid_json_rate(loaded["B"], test_ids),
        n=len(test_ids),
        n_en=langs.count("en"),
        n_ar=langs.count("ar"),
        test_digest=ids_digest(test_ids),
        a_rows_sha256=sha256_file(a_rows),
        b_rows_sha256=sha256_file(b_rows),
        a_identity=a_identity,
        b_identity=b_identity,
    )


def assemble_evidence(
    *,
    version: ModelVersion,
    manifest: ExportManifest,
    champion: ModelVersion | None,
    scores: UaeScores,
    latency: LatencyRun,
    latency_digest: str,
    hygiene: Hygiene,
    swap_recorded: bool,
    session_id: str,
    measured_at: datetime,
    git_sha: str,
) -> GateEvidence:
    report = ExtractorGateReport(
        session_id=session_id,
        uae_test_digest=scores.test_digest,
        latency_subset_digest=latency_digest,
        uae_f1_overall=scores.b_overall,
        uae_f1_en=scores.b_en,
        uae_f1_ar=scores.b_ar,
        uae_n=scores.n,
        valid_json_rate=scores.b_valid_json,
        latency_p50_s=latency.b.p50_s,
        latency_p95_s=latency.b.p95_s,
        latency_n=latency.b.n,
        reference_a_f1_overall=scores.a_overall,
        reference_a_p50_s=latency.a.p50_s,
        extract_peak_vram_gb=latency.peak_vram_gb,
        sequential_fallback_recorded=swap_recorded,
        leakage_green=hygiene.leakage_green,
        base_revision=hygiene.base_revision,
        lock_base_revision=hygiene.lock_base_revision,
    )
    if latency.a.n != latency.b.n:
        raise GateEvidenceError(f"A timed {latency.a.n} documents, B {latency.b.n}")
    if hygiene.base_revision != manifest.base_revision:
        raise GateEvidenceError(
            f"run base {hygiene.base_revision} differs from export base {manifest.base_revision}"
        )
    return GateEvidence(
        registered_model=REGISTERED_MODEL,
        model_version=version,
        run_id=manifest.run_id,
        mlflow_run_id=manifest.mlflow_run_id,
        gguf_sha256=manifest.artifact(Q4_NAME).sha256,
        mmproj_sha256=manifest.artifact(MMPROJ_NAME).sha256,
        champion_at_measurement=champion,
        challenger_alias=CHALLENGER_ALIAS,
        reference_alias=A_ALIAS,
        measured_at=measured_at,
        git_sha=git_sha,
        report=report,
        reference_a_f1_en=scores.a_en,
        reference_a_f1_ar=scores.a_ar,
        reference_a_p95_s=latency.a.p95_s,
        uae_n_en=scores.n_en,
        uae_n_ar=scores.n_ar,
        vram_samples=latency.vram_samples,
        vram_missed_readings=latency.vram_misses,
        a_rows_sha256=scores.a_rows_sha256,
        b_rows_sha256=scores.b_rows_sha256,
        a_rows_identity=scores.a_identity,
        b_rows_identity=scores.b_identity,
        leakage_detail=hygiene.leakage_detail,
        results=evaluate_gates(report),
    )


def publish_evidence(
    evidence: GateEvidence,
    registry: MlflowRegistry,
    *,
    evidence_path: Path = EVIDENCE_PATH,
    report_path: Path = GATES_REPORT,
) -> str:
    """Write the JSON and the markdown, then bind the JSON's sha256 to the model version."""
    sha = write_evidence(evidence_path, evidence)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(evidence), encoding="utf-8", newline="\n")
    registry.set_version_tags(evidence.model_version, gate_tags(evidence, sha))
    return sha


# --- live wiring (GPU, llama-swap, parse service) --------------------------------------------


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise GateEvidenceError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


async def _live_latency(
    args: argparse.Namespace, shas: Sequence[str]
) -> LatencyRun:  # pragma: no cover - needs the GPU, llama-swap and the parse service
    from scripts.smoke import sample_vram_gb

    from apx.llm.client import LlamaSwapClient, LlmSettings
    from apx.parse.client import ParseServiceClient
    from apx.parse.parser import PaddleOcrDocumentParser
    from ml.extractor_b.gates import StrictVramSampler, llm_extract_fn, run_latency

    parse_client = ParseServiceClient()
    parser = PaddleOcrDocumentParser(parse_client)
    documents = [
        await parser.parse(sha, (args.data_dir / f"{sha}.pdf").read_bytes()) for sha in shas
    ]
    await parse_client.unload()  # parse stage over: its VRAM is not part of the extract stage
    client = LlamaSwapClient(
        LlmSettings(base_url=args.base_url, allowed_models=frozenset({A_ALIAS, args.model}))
    )
    extract = llm_extract_fn(client, a_model=A_ALIAS, b_model=args.model)

    try:
        await client.unload()  # start the extract stage from an empty llama-swap
        return await run_latency(
            documents,
            extract,
            StrictVramSampler(sample_vram_gb),
            a_model=A_ALIAS,
            b_model=args.model,
        )
    finally:
        await client.unload()
        await client.aclose()


def run_gates(
    args: argparse.Namespace, registry: MlflowRegistry
) -> GateEvidence:  # pragma: no cover - composes the live measurement
    from evals.splits import load_splits

    manifest = verified_manifest(args.export_dir)
    version = challenger_matching_export(registry, manifest)
    champion = registry.alias("champion")
    test_ids = sorted(load_splits().sets["uae"]["test"])
    missing = [sha for sha in test_ids if not (args.data_dir / f"{sha}.pdf").exists()]
    if missing:
        raise GateEvidenceError(f"{len(missing)} UAE test PDFs missing in {args.data_dir}")
    require_challenger_serves(args.export_dir, alias=args.model)
    a_identity, b_identity = expected_identities(manifest, b_alias=args.model)
    scores = score_row_files(
        a_rows=args.a_rows,
        b_rows=args.b_rows,
        data_dir=args.data_dir,
        test_ids=test_ids,
        a_identity=a_identity,
        b_identity=b_identity,
    )
    hygiene = check_hygiene(args.export_dir.parent, args.train_data_dir, lock_path=MODELS_LOCK)
    swap = extract_group_swaps(LLAMA_SWAP_CONFIG, (A_ALIAS, args.model))
    subset = latency_subset(test_ids)
    latency = asyncio.run(_live_latency(args, subset))
    evidence = assemble_evidence(
        version=version,
        manifest=manifest,
        champion=champion,
        scores=scores,
        latency=latency,
        latency_digest=ids_digest(subset),
        hygiene=hygiene,
        swap_recorded=swap,
        session_id=f"m7.1-{uuid.uuid4().hex[:12]}",
        measured_at=datetime.now(UTC),
        git_sha=_git_sha(),
    )
    publish_evidence(evidence, registry)
    return evidence


# --- CLI -------------------------------------------------------------------------------------


def _default_export_dir() -> Path:
    return latest_run_dir(DEFAULT_RUNS_DIR) / EXPORT_DIR_NAME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.promote", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "eval-b", "gates", "promote"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--export-dir", type=Path, default=None, help="default: latest run")
        cmd.add_argument(
            "--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
        )
        cmd.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-swap")
        cmd.add_argument("--model", default=CHALLENGER_ALIAS, help="llama-swap alias of B")
        cmd.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        cmd.add_argument("--train-data-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR)
        cmd.add_argument("--a-rows", type=Path, default=DEFAULT_A_ROWS)
        cmd.add_argument("--b-rows", type=Path, default=DEFAULT_B_ROWS)
        cmd.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
        cmd.add_argument("--limit", type=int, default=0, help="eval-b only: smoke a few docs")
    args = parser.parse_args(argv)
    if args.export_dir is None:
        args.export_dir = _default_export_dir()
    return args


def eval_b_argv(args: argparse.Namespace) -> list[str]:
    """The Extractor A harness, pointed at B (same scorer, resumable rows, EN/AR count)."""
    argv = [
        "--extractor", "b",
        "--model", args.model,
        "--data-dir", str(args.data_dir),
        "--base-url", args.base_url,
        "--rows", str(args.b_rows),
        "--report", str(B_TEST_REPORT),
    ]  # fmt: skip
    return argv + (["--limit", str(args.limit)] if args.limit else [])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "eval-b":  # pragma: no cover - live llama-swap + parse service
        from evals.extraction_a import main as extraction_main

        try:
            require_challenger_serves(args.export_dir, alias=args.model)
        except PromotionError as exc:
            print(f"promote-b: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return extraction_main(eval_b_argv(args))
    registry = MlflowRegistry(args.mlflow_uri)
    try:
        if args.command == "register":
            version = register(args.export_dir, registry)
            print(
                f"registered {REGISTERED_MODEL} v{version} from {args.export_dir}; alias challenger"
            )
            return 0
        if args.command == "gates":  # pragma: no cover - live measurement
            evidence = run_gates(args, registry)
            print(render_markdown(evidence))
            return 0 if evidence.all_passed else 1
        result = promote(
            evidence_path=args.evidence,
            export_dir=args.export_dir,
            registry=registry,
            invalidator=FileCalibratorInvalidator(),
            lock_writer=JsonModelsLockWriter(),
            repointer=LlamaSwapRepointer(),
        )
        if not result.applied:
            print(f"{REGISTERED_MODEL} v{result.champion_after} is already champion")
        else:
            print(
                f"promoted {REGISTERED_MODEL} v{result.champion_after} "
                f"(was {result.champion_before or 'unset'}): {', '.join(result.applied)}. "
                "Next: `cd infra && docker compose restart llama-swap`, then rerun the "
                "M7.4/M7.5 cal job (the calibrator is invalidated)."
            )
        return 0
    except PromotionError as exc:
        print(f"promote-b: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1 if isinstance(exc, GatesNotPassedError) else 2
    finally:
        registry.close()


if __name__ == "__main__":
    sys.exit(main())
