"""The runtime injection guard (P4, ADR-0017): scores a document's
text spans with an `InjectionClassifier`, reads the review and block thresholds chosen on the
cal split from the checkpoint's guard.json, and turns every span at or above review into an
INJECTION scan flag whose score lands on the policy engine's own P4 scale. Fails closed: a
missing artifact or a classifier failure raises `ScanError`, never an empty (clean) result."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from apx.common.errors import ScanError
from apx.scan.interfaces import InjectionClassifier, ScanFlag, ScanFlagKind
from apx.scan.spans import TextSpan, evidence_spans, page_spans, scorable_spans
from apx.scan.text import build_evidence

GUARD_META_FILE: Final[str] = "guard.json"
MAX_SPANS: Final[int] = 5000
"""Spans scored per document. A real invoice has a few hundred; a PDF built to exhaust CPU with
more text lines than this fails the scan (closed) instead of stalling the worker."""

_Probability = Annotated[float, Field(gt=0.0, le=1.0)]


class GuardThresholds(BaseModel):
    """guard.json as `ml.guard.train` / `ml.guard.thresholds` write it. `threshold` is the cal
    max-F1 point (reports/guard.md): at or above it a span routes REVIEW. `block_threshold` is
    the lowest cal score no benign cal row reached: at or above it the document is BLOCKED."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    threshold: _Probability
    block_threshold: _Probability
    smoke: bool = False

    @property
    def review(self) -> float:
        return self.threshold

    @property
    def block(self) -> float:
        return self.block_threshold

    @model_validator(mode="after")
    def _ordered_and_real(self) -> GuardThresholds:
        if self.block_threshold < self.threshold:
            raise ValueError(
                f"block_threshold {self.block_threshold} is below threshold {self.threshold}"
            )
        if self.smoke:
            raise ValueError("guard.json is from a smoke run (--max-steps/--subset), not a model")
        return self


def load_guard_thresholds(model_dir: Path) -> GuardThresholds:
    """Read and validate `<model_dir>/guard.json`. Raises `ScanError` when it is missing,
    unreadable, lacks either threshold, or came from a smoke run."""
    path = model_dir / GUARD_META_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return GuardThresholds.model_validate(raw)
    except FileNotFoundError as exc:
        raise ScanError(f"guard thresholds not found: {path}") from exc
    except (OSError, ValueError, ValidationError) as exc:
        raise ScanError(f"guard thresholds invalid in {path}: {exc}") from exc


def policy_score(
    probability: float, thresholds: GuardThresholds, *, scan_review: float, attack_block: float
) -> float:
    """Map P(injection) onto the policy engine's P4 scale, piecewise linearly and monotonically:
    the model's review threshold lands exactly on `scan_review` and its block threshold exactly
    on `attack_block`, so `DefaultPolicyEngine` routes REVIEW/BLOCK at the cal-chosen points
    without knowing the model exists."""
    review, block = thresholds.review, thresholds.block
    if probability < review:
        # nextafter: rounding must never lift a below-threshold probability onto the threshold.
        return min(probability / review * scan_review, math.nextafter(scan_review, 0.0))
    if probability < block:
        mapped = scan_review + (probability - review) / (block - review) * (
            attack_block - scan_review
        )
        return min(mapped, math.nextafter(attack_block, 0.0))
    if block >= 1.0:
        return 1.0
    return min(1.0, attack_block + (probability - block) / (1.0 - block) * (1.0 - attack_block))


class InjectionGuard:
    """Scores spans and emits INJECTION flags. `scan_review` / `attack_block` are the same
    `PolicyThresholds` values the worker's policy engine routes with."""

    def __init__(
        self,
        classifier: InjectionClassifier,
        thresholds: GuardThresholds,
        *,
        scan_review: float,
        attack_block: float,
        max_spans: int = MAX_SPANS,
    ) -> None:
        if not 0.0 < scan_review <= attack_block <= 1.0:
            raise ScanError(
                f"need 0 < scan_review <= attack_block <= 1, got {scan_review}, {attack_block}"
            )
        self._classifier = classifier
        self._thresholds = thresholds
        self._scan_review = scan_review
        self._attack_block = attack_block
        self._max_spans = max_spans

    def detect(self, pdf_bytes: bytes, detector_flags: Sequence[ScanFlag]) -> list[ScanFlag]:
        """Score every text line of the PDF (visible and hidden) and the document text in
        `detector_flags` (metadata, annotations, attachments), Latin script only (ADR-0017)."""
        try:
            found = page_spans(pdf_bytes)
        except Exception as exc:
            raise ScanError(f"guard could not read the PDF text layer: {exc}") from exc
        return self.flags(scorable_spans([*found, *evidence_spans(detector_flags)]))

    def flags(self, spans: Sequence[TextSpan]) -> list[ScanFlag]:
        """One INJECTION flag per span whose P(injection) reaches the review threshold, scored
        on the P4 scale. Raises `ScanError` if the classifier fails or returns a malformed
        result; an empty list means every span was scored and none reached review."""
        if len(spans) > self._max_spans:
            raise ScanError(f"{len(spans)} text spans exceed the guard's limit {self._max_spans}")
        if not spans:
            return []
        try:
            probabilities = self._classifier.score([span.text for span in spans])
        except ScanError:
            raise
        except Exception as exc:
            raise ScanError(f"injection classifier failed: {exc}") from exc
        if len(probabilities) != len(spans):
            raise ScanError(
                f"injection classifier returned {len(probabilities)} scores for {len(spans)} spans"
            )
        flags: list[ScanFlag] = []
        for span, probability in zip(spans, probabilities, strict=True):
            if not (math.isfinite(probability) and 0.0 <= probability <= 1.0):
                raise ScanError(f"injection classifier returned an invalid score {probability}")
            if probability < self._thresholds.review:
                continue
            score = policy_score(
                probability,
                self._thresholds,
                scan_review=self._scan_review,
                attack_block=self._attack_block,
            )
            flags.append(
                ScanFlag(
                    kind=ScanFlagKind.INJECTION,
                    page=span.page,
                    score=score,
                    evidence=build_evidence(ScanFlagKind.INJECTION, span.text),
                    bbox=span.bbox,
                )
            )
        return flags
