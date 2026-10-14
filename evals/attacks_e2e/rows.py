"""Resumable per-case rows for the M6.6 attack run: one JSONL file of `snapshot`,
`dropped`, `approved` and `outcome` events keyed by case id, attack sha256 and the run identity
(defending code, running images, served models, variant, ERP, approve mode). A resumed run keeps
only rows of its own identity; a changed attack or stack starts that case over."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from apx.common.types import Frozen
from evals.attacks_e2e.outcome import CaseState, CaseTruth, HarnessFacts, MasterSnapshot

DEFENSE_GLOBS: tuple[str, ...] = (
    "apx/**/*.py",
    "infra/compose.yaml",
    "infra/llama-swap.yaml",
    "infra/models.lock.json",
)
"""What decides how an attack is handled: the product code and the served models/config."""


def defense_fingerprint(root: Path, globs: Iterable[str] = DEFENSE_GLOBS) -> str:
    """SHA-256 over every defending file's repo-relative path and content hash, sorted, so an
    edit anywhere in the pipeline code or model config gives a new fingerprint."""
    digest = hashlib.sha256()
    paths = sorted({p for pattern in globs for p in root.glob(pattern) if p.is_file()})
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii") + b"\n")
    return digest.hexdigest()


class RunIdentity(Frozen):
    """What a row was measured against: the host checkout's defending files, the running
    containers' image ids and the served model hashes (`evals.attacks_e2e.identity`), the
    variant label, the target ERP and the approve mode."""

    defense_sha256: str
    images: dict[str, str]
    """Compose service -> running image id."""
    models: dict[str, str]
    """llama-swap alias -> served GGUF sha256."""
    variant: str
    """Operator label for the stack configuration, e.g. `guard-on` / `guard-off`."""
    erp: str
    """`seed/vendors/users` of the ERP the cases were rendered against."""
    approve: bool

    def key(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:16]


class ApproverView(Frozen):
    """What the review screen shows a human approver: the route reasons and fraud flags."""

    reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


class SnapshotRow(Frozen):
    event: Literal["snapshot"] = "snapshot"
    identity_key: str
    snapshot: MasterSnapshot


class DroppedRow(Frozen):
    event: Literal["dropped"] = "dropped"
    identity_key: str
    case_id: str
    attack_sha256: str
    dropped_at: float


class ApprovedRow(Frozen):
    """The harness approver's click on a review-routed case: the HTTP status it got and, for a
    refusal, whether it was a typed error (`{"detail": "<str>"}`) and its text."""

    event: Literal["approved"] = "approved"
    identity_key: str
    case_id: str
    attack_sha256: str
    approve_http: int
    approve_typed: bool = False
    detail: str = ""
    at: float


class OutcomeRow(Frozen):
    event: Literal["outcome"] = "outcome"
    identity_key: str
    case_id: str
    attack_sha256: str
    truth: CaseTruth
    state: CaseState
    harness: HarnessFacts
    view: ApproverView
    dropped_at: float
    routed_at: float | None
    settled_at: float


Row = SnapshotRow | DroppedRow | ApprovedRow | OutcomeRow
_ROW: TypeAdapter[Row] = TypeAdapter(Annotated[Row, Field(discriminator="event")])


class RowStore:
    """Append-only JSONL of one run's rows. `load` keeps only rows with this store's identity key
    and, per case, only the ones for the attack sha256 it is asked about."""

    def __init__(self, path: Path, identity_key: str) -> None:
        self.path = path
        self.identity_key = identity_key
        self.snapshot: MasterSnapshot | None = None
        self._dropped: dict[tuple[str, str], DroppedRow] = {}
        self._approved: dict[tuple[str, str], ApprovedRow] = {}
        self._outcomes: dict[tuple[str, str], OutcomeRow] = {}
        self.ignored = 0
        """Rows skipped on load because another identity wrote them."""

    def load(self) -> RowStore:
        if not self.path.exists():
            return self
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _ROW.validate_python(json.loads(line))
            if row.identity_key != self.identity_key:
                self.ignored += 1
                continue
            self._remember(row)
        return self

    def _remember(self, row: Row) -> None:
        if isinstance(row, SnapshotRow):
            if self.snapshot is None:  # the first snapshot is the run's baseline
                self.snapshot = row.snapshot
        elif isinstance(row, DroppedRow):
            self._dropped.setdefault((row.case_id, row.attack_sha256), row)
        elif isinstance(row, ApprovedRow):
            self._approved.setdefault((row.case_id, row.attack_sha256), row)
        else:
            self._outcomes[(row.case_id, row.attack_sha256)] = row

    def append(self, row: Row) -> None:
        if row.identity_key != self.identity_key:
            raise ValueError(f"row for identity {row.identity_key}, store is {self.identity_key}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(row.model_dump_json() + "\n")
            handle.flush()
        self._remember(row)

    def knows(self, attack_sha256: str) -> bool:
        """Whether this run dropped (or finished) a document with this sha256."""
        return any(sha == attack_sha256 for _, sha in (*self._dropped, *self._outcomes))

    def outcome(self, case_id: str, attack_sha256: str) -> OutcomeRow | None:
        return self._outcomes.get((case_id, attack_sha256))

    def dropped(self, case_id: str, attack_sha256: str) -> DroppedRow | None:
        return self._dropped.get((case_id, attack_sha256))

    def approved(self, case_id: str, attack_sha256: str) -> ApprovedRow | None:
        return self._approved.get((case_id, attack_sha256))

    def outcomes(self) -> list[OutcomeRow]:
        return list(self._outcomes.values())
