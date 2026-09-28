"""Persisted trust-gate calibrator (M7.4/M7.5, ADR-0009 §5-§6, ADR-0015 §5): the versioned
artifact under data/trust/ (calibrator parameters, λ̂ per alpha, and the A/B build identities it
was fitted on), the active pointer the runtime gate loads, and the ADR-0015 invalidation marker that
a promotion writes and only a refit on the current champions clears. Never committed (data/).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import Field, ValidationError

from apx.common.errors import TrustGateError
from apx.common.types import Frozen, Sha256Hex
from apx.extract.identity import ExtractorIdentity
from apx.trust.calibration import CalibratorState

ARTIFACT_SCHEMA: Final[int] = 1
RUNTIME_ALPHA: Final[float] = 0.01
"""The gated runtime alpha (ADR-0009 §6); the other reported alphas never drive decisions."""
REPORTED_ALPHAS: Final[tuple[float, ...]] = (0.005, 0.01, 0.02)
LTT_DELTA: Final[float] = 0.05
DEFAULT_TRUST_DIR: Final[Path] = Path("data/trust")
ACTIVE_POINTER_NAME: Final[str] = "active_calibrator.json"
INVALIDATION_MARKER_NAME: Final[str] = "calibrator_invalidation.json"
"""Written by `ml.extractor_b.promote.FileCalibratorInvalidator` / `ml.promotion.retrain`."""
CALIBRATORS_DIR_NAME: Final[str] = "calibrators"


class RefusalReason(StrEnum):
    """Why the runtime gate will not use a calibrator; each one routes the invoice to REVIEW."""

    MISSING = "calibrator_missing"
    CORRUPT = "calibrator_corrupt"
    INVALIDATED = "calibrator_invalidated"
    IDENTITY_MISMATCH = "identity_mismatch"
    SERVED_IDENTITY_UNKNOWN = "served_identity_unknown"
    CHAMPION_ONLY = "calibrator_champion_only"
    """The artifact was fitted on Extractor A alone (ADR-0018 §5) but B output is served: that
    output was never calibrated, so nothing is auto-accepted."""


class CalibratorUnavailableError(TrustGateError):
    """No usable calibrator: the gate refuses and nothing is auto-accepted."""

    def __init__(self, reason: RefusalReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


class AlphaThreshold(Frozen):
    """λ̂ for one alpha on the LTT partition; None means that alpha auto-accepts nothing."""

    alpha: Annotated[float, Field(gt=0.0, lt=1.0)]
    delta: Annotated[float, Field(gt=0.0, lt=1.0)]
    lambda_hat: float | None
    ltt_accepted: Annotated[int, Field(ge=0)]
    ltt_wrong: Annotated[int, Field(ge=0)]


class CalibratorArtifact(Frozen):
    """Everything the runtime gate needs, plus the provenance that ties it to its inputs."""

    schema_version: int = ARTIFACT_SCHEMA
    version: str
    """12-hex digest of the calibrator version, the A/B identities and the thresholds."""
    calibrator: CalibratorState
    extractor_a: ExtractorIdentity
    extractor_b: ExtractorIdentity | None
    """The B build the features saw, or None for a champion-only artifact (ADR-0018 §5): fitted
    on A with the fixed B-absent encoding. The runtime gate refuses such an artifact whenever B
    output is served (`RefusalReason.CHAMPION_ONLY`); it never scores B-derived features with it."""
    arbiter: ExtractorIdentity | None
    """The arbiter build the features saw, or None (JSON null) when none was served: the runtime
    gate refuses when the served arbiter differs, including one appearing or disappearing."""
    thresholds: tuple[AlphaThreshold, ...]
    runtime_alpha: float = RUNTIME_ALPHA
    ltt_min_accepted: Annotated[int, Field(ge=1)]
    fitted_at: datetime
    n_cal_documents: Annotated[int, Field(ge=1)]
    n_fit_fields: Annotated[int, Field(ge=0)]
    n_ltt_fields: Annotated[int, Field(ge=0)]
    cal_documents_digest: Sha256Hex
    """sha256 over the sorted cal document sha256s (one per line)."""
    rows_sha256: dict[str, Sha256Hex]
    """sha256 of the A and B cal row files the calibrator was fitted on (A only when
    champion-only)."""

    @property
    def champion_only(self) -> bool:
        return self.extractor_b is None

    def threshold(self, alpha: float) -> AlphaThreshold:
        for item in self.thresholds:
            if item.alpha == alpha:
                return item
        raise TrustGateError(f"calibrator {self.version} has no threshold for alpha={alpha}")

    def runtime_threshold(self) -> float | None:
        return self.threshold(self.runtime_alpha).lambda_hat


def artifact_version(
    calibrator_version: str,
    extractor_a: ExtractorIdentity,
    extractor_b: ExtractorIdentity | None,
    thresholds: tuple[AlphaThreshold, ...],
    arbiter: ExtractorIdentity | None = None,
) -> str:
    """12-hex version folding the A/B/arbiter build identities into the calibrator version (ADR-0009
    §5: the version names the champions it was fitted on; B None = champion-only)."""
    payload = {
        "calibrator": calibrator_version,
        "a": extractor_a.model_dump(mode="json"),
        "b": None if extractor_b is None else extractor_b.model_dump(mode="json"),
        "arbiter": None if arbiter is None else arbiter.model_dump(mode="json"),
        "thresholds": [t.model_dump(mode="json") for t in thresholds],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class ActivePointer(Frozen):
    version: str
    file: str
    """Path of the artifact relative to the trust directory."""
    sha256: Sha256Hex


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class CalibratorStore:
    """data/trust/: `calibrators/<version>.json` (immutable), `active_calibrator.json` (which one
    the runtime uses, with its sha256) and `calibrator_invalidation.json` (ADR-0015 marker)."""

    def __init__(self, root: Path = DEFAULT_TRUST_DIR) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    @property
    def invalidation_path(self) -> Path:
        return self._root / INVALIDATION_MARKER_NAME

    @property
    def active_path(self) -> Path:
        return self._root / ACTIVE_POINTER_NAME

    def save(self, artifact: CalibratorArtifact) -> Path:
        """Write the versioned artifact (idempotent for the same version) and return its path."""
        path = self._root / CALIBRATORS_DIR_NAME / f"{artifact.version}.json"
        _atomic_write(path, artifact.model_dump_json(indent=2) + "\n")
        return path

    def activate(self, artifact: CalibratorArtifact) -> ActivePointer:
        """Point the runtime at a saved artifact, pinning its file sha256."""
        rel = f"{CALIBRATORS_DIR_NAME}/{artifact.version}.json"
        data = (self._root / rel).read_bytes()
        pointer = ActivePointer(
            version=artifact.version, file=rel, sha256=hashlib.sha256(data).hexdigest()
        )
        _atomic_write(self.active_path, pointer.model_dump_json(indent=2) + "\n")
        return pointer

    def load_active(self) -> tuple[CalibratorArtifact, Sha256Hex]:
        """The active artifact and its verified sha256. Raises CalibratorUnavailableError
        (MISSING or CORRUPT); never falls back to another artifact."""
        if not self.active_path.is_file():
            raise CalibratorUnavailableError(
                RefusalReason.MISSING, f"no active calibrator at {self.active_path}"
            )
        try:
            pointer = ActivePointer.model_validate_json(self.active_path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise CalibratorUnavailableError(
                RefusalReason.CORRUPT, f"unreadable {self.active_path}: {exc}"
            ) from exc
        rel = Path(pointer.file)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[:1] != (CALIBRATORS_DIR_NAME,):
            raise CalibratorUnavailableError(
                RefusalReason.CORRUPT, f"{self.active_path} points outside {CALIBRATORS_DIR_NAME}/"
            )
        path = self._root / rel
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CalibratorUnavailableError(
                RefusalReason.MISSING, f"active calibrator file {path} is missing"
            ) from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != pointer.sha256:
            raise CalibratorUnavailableError(
                RefusalReason.CORRUPT,
                f"{path} sha256 {digest[:12]} != pinned {pointer.sha256[:12]}",
            )
        try:
            artifact = CalibratorArtifact.model_validate_json(data)
        except ValidationError as exc:
            raise CalibratorUnavailableError(
                RefusalReason.CORRUPT, f"{path} is not a calibrator artifact: {exc}"
            ) from exc
        if artifact.schema_version != ARTIFACT_SCHEMA or artifact.version != pointer.version:
            raise CalibratorUnavailableError(
                RefusalReason.CORRUPT,
                f"{path}: schema {artifact.schema_version}, version {artifact.version} does not "
                f"match the pointer ({ARTIFACT_SCHEMA}, {pointer.version})",
            )
        return artifact, digest

    def invalidation(self) -> dict[str, Any] | None:
        """The stale marker when the calibrator is invalidated, else None. An unreadable marker
        counts as invalidated (fail closed)."""
        path = self.invalidation_path
        if not path.exists():
            return None
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"stale": True, "reason": f"unreadable marker: {exc}"}
        if not isinstance(marker, dict):
            return {"stale": True, "reason": "marker is not a JSON object"}
        return marker if marker.get("stale") is not False else None

    def clear_invalidation(self, artifact: CalibratorArtifact, now: datetime) -> bool:
        """Mark the calibrator fresh after a refit on the current champions; keeps the previous
        marker for the record. Returns whether a stale marker was cleared."""
        previous = self.invalidation()
        if previous is None:
            return False
        cleared = {
            "stale": False,
            "cleared_at": now.isoformat(),
            "cleared_by_calibrator": artifact.version,
            "extractor_a": artifact.extractor_a.model_dump(mode="json"),
            "extractor_b": None
            if artifact.extractor_b is None
            else artifact.extractor_b.model_dump(mode="json"),
            "previous": previous,
        }
        _atomic_write(self.invalidation_path, json.dumps(cleared, indent=2) + "\n")
        return True
