"""Typed rows and reports of the real-data track (ADR-0019): one row per document per extractor
version, one aggregate report per set, and the before/after comparison evals/scoreboard.py reads,
plus the typed errors that enforce the track's honesty rules (§4)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from apx.common.errors import ApxError

RealSet = Literal["cord", "extractbench", "docile", "fatura", "reality"]
ErrorKind = Literal["timeout", "server_error", "rejected", "parse_error", "invalid_json"]
ExtractorName = Literal["a", "b"]
Distribution = Literal["zero-shot", "in-distribution"]

ERROR_KINDS: Final[tuple[ErrorKind, ...]] = (
    "timeout",
    "server_error",
    "rejected",
    "parse_error",
    "invalid_json",
)
IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "model",
    "gguf_sha256",
    "prompt_version",
    "serving_args_sha256",
    "parser_identity",
)
"""Stamp keys a report requires to be equal on every row (§4.3). `git_sha` is stamped too but
is listed, not compared: a resume after an unrelated commit stays one run."""


class RealTrackError(ApxError):
    """Base of the real-data track's typed errors."""


class DirtyTreeError(RealTrackError):
    """A protected (non-dev) split was requested while extraction or scoring code has
    uncommitted changes (§4.1)."""


class ConsecutiveFailuresError(RealTrackError):
    """Three documents in a row failed on the server side (timeout or HTTP 5xx; the ADR-0016
    rule), so the run stops instead of burning the queue."""


class MixedIdentityError(RealTrackError):
    """Rows of one report, or arms of a before/after, do not share an identity or doc set."""


class DatasetMissingError(RealTrackError):
    """The dataset a real-data target needs is not fetched (e.g. DocILE without its token)."""


class RealRow(BaseModel):
    """One document's outcome. A failure is a row (`error_kind` set, `prediction` None) that
    stays in every denominator; nothing is skipped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set: RealSet
    split: str
    doc_sha256: str
    doc_id: str | None = None
    """The dataset's own id when it has one (ExtractBench pdf stem, DocILE id, CORD row)."""
    extractor: ExtractorName
    model_version: str
    stamp: dict[str, Any]
    prediction: dict[str, Any] | None
    error_kind: ErrorKind | None
    latency_s: float | None


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float
    method: Literal["bootstrap", "paired_bootstrap", "wilson"]


class MetricValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    ci: Interval
    n_docs: int


class Baseline(BaseModel):
    """A published number shown beside ours, with whether it is like-for-like and why not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    source: str
    comparable: bool
    note: str


class RealEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    set: RealSet
    split: str
    extractor: ExtractorName
    model_version: str
    n_docs: int
    n_failed: int
    failures_by_kind: dict[ErrorKind, int]
    valid_json_rate: MetricValue
    metrics: list[MetricValue]
    baselines: list[Baseline] = Field(default_factory=list)
    mapped_gold_share: float | None
    distribution: Distribution
    normaliser_sha256: str
    identity: dict[str, Any]
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    doc_ids_sha256: str
    """sha256 over the sorted document sha256s: the before/after check compares these."""
    doc_sha256s: list[str] = Field(default_factory=list)
    """The scored documents, sorted; `per_doc_counts` lists follow this order."""
    per_doc_counts: dict[str, list[tuple[int, int, int]]] = Field(default_factory=dict)
    """Per F1 metric, each document's (tp, fp, fn): counts only, no values, so a before/after
    can pair arms document by document."""
    generated_at: datetime

    def metric(self, name: str) -> MetricValue | None:
        return next((m for m in self.metrics if m.name == name), None)


class BeforeAfterReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    set: RealSet
    split: str
    doc_ids_sha256: str
    arms: list[RealEvalReport]
    deltas: list[MetricValue]
    """Paired, each arm vs the first arm, named `<metric>:<arm>-<first arm>`."""
    generated_at: datetime
