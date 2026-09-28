"""Runtime trust gate (M7.5, PRD §6.1): loads the active calibrator artifact from data/trust/ and
auto-accepts a field only when its calibrated P(correct) >= λ̂ at the runtime alpha. It refuses,
and routes the invoice to REVIEW, when the artifact is missing, corrupt, invalidated (ADR-0015) or
fitted on other A/B/arbiter builds than the ones served, and (`refuse_truncated_parse`) when the
OCR transcription of a page was cut short. A champion-only artifact (fitted on A alone, ADR-0018
§5) is refused while B output is served. It never authorises payment (P1-P5 do).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from apx.common.errors import TrustGateError
from apx.extract.identity import (
    ExtractorIdentity,
    ServedBuilds,
    ServedModelError,
    prompt_version,
    served_gguf_sha256,
)
from apx.extract.interfaces import FieldName, PromptBuilder
from apx.trust.artifact import (
    CalibratorArtifact,
    CalibratorStore,
    CalibratorUnavailableError,
    RefusalReason,
)
from apx.trust.calibration import FrozenCalibrator
from apx.trust.interfaces import FieldFeatures, TrustDecision, TrustGate
from apx.trust.ltt import CalibratedTrustGate

log = logging.getLogger(__name__)

IDENTITY_FAILURES: tuple[type[Exception], ...] = (
    ServedModelError,
    ValidationError,
    OSError,
    ValueError,
    KeyError,
    AttributeError,
    TypeError,
)
"""Any of these while establishing the served builds means "identity unknown": a refusal routed
to REVIEW and audited, never an activity error that Temporal would retry forever."""


class ServedIdentities(Protocol):
    def current(self) -> ServedBuilds:
        """Identities of the builds served right now. Raises ServedModelError (or another
        `IDENTITY_FAILURES` type) when they cannot be established."""
        ...


class FileServedIdentities:
    """Served identities from infra/llama-swap.yaml, infra/models.lock.json and the export
    manifests under models/ (re-read per call, so a repoint is seen), with the prompt versions of
    this process's own prompt builders. The arbiter is identified only when one is served."""

    def __init__(
        self,
        a_alias: str,
        b_alias: str,
        a_prompt: PromptBuilder,
        b_prompt: PromptBuilder,
        *,
        llama_swap_config: Path,
        models_lock: Path,
        models_root: Path,
        arbiter: tuple[str, PromptBuilder] | None = None,
    ) -> None:
        self._served: list[tuple[str, PromptBuilder]] = [(a_alias, a_prompt), (b_alias, b_prompt)]
        if arbiter is not None:
            self._served.append(arbiter)
        self._paths = {
            "llama_swap_config": llama_swap_config,
            "models_lock": models_lock,
            "models_root": models_root,
        }
        self._prompt_versions: list[str] | None = None

    def current(self) -> ServedBuilds:
        if self._prompt_versions is None:
            self._prompt_versions = [prompt_version(builder) for _, builder in self._served]
        identities = [
            ExtractorIdentity(
                model=alias,
                gguf_sha256=served_gguf_sha256(alias, **self._paths),
                prompt_version=version,
            )
            for (alias, _), version in zip(self._served, self._prompt_versions, strict=True)
        ]
        return ServedBuilds(*identities)


def _describe(identity: ExtractorIdentity | None) -> str:
    return "none" if identity is None else f"{identity.gguf_sha256[:12]}/{identity.prompt_version}"


def builds_match(artifact: CalibratorArtifact, served: ServedBuilds) -> str | None:
    """None when the artifact was fitted on exactly the served builds (aliases may differ), else
    what differs. An artifact fitted without an arbiter does not match a served arbiter, and vice
    versa (ADR-0009 §5: the version names the A/B/arbiter builds)."""
    pairs = (
        ("A", artifact.extractor_a, served.a),
        ("B", artifact.extractor_b, served.b),
        ("arbiter", artifact.arbiter, served.arbiter),
    )
    for which, fitted, now in pairs:
        same = fitted is None and now is None
        if fitted is not None and now is not None:
            same = fitted.same_build(now)
        if not same:
            return f"{which}: fitted on {_describe(fitted)}, served {_describe(now)}"
    return None


class ArtifactTrustGate:
    """`TrustGate` over the active calibrator artifact. Every decision re-checks the invalidation
    marker, the artifact's pinned sha256 and the served identities, so a promotion or a repoint
    takes effect on the next invoice. When any check fails the decision comes from `fallback`
    (the interim no-calibrator gate) with every field forced to not-accepted and `refusal` set."""

    def __init__(
        self, store: CalibratorStore, served: ServedIdentities, fallback: TrustGate
    ) -> None:
        self._store = store
        self._served = served
        self._fallback = fallback
        self._cached: tuple[str, CalibratedTrustGate] | None = None
        self._last_refusal: str | None = None

    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        try:
            artifact, gate = self._active()
        except CalibratorUnavailableError as exc:
            return self._refuse(features, exc)
        self._last_refusal = None
        decision = gate.decide(features)
        return decision.model_copy(update={"calibrator_version": artifact.version})

    def _active(self) -> tuple[CalibratorArtifact, CalibratedTrustGate]:
        marker = self._store.invalidation()
        if marker is not None:
            raise CalibratorUnavailableError(
                RefusalReason.INVALIDATED,
                f"{self._store.invalidation_path}: {marker.get('reason', 'stale')}",
            )
        artifact, digest = self._store.load_active()
        try:
            served = self._served.current()
        except IDENTITY_FAILURES as exc:
            raise CalibratorUnavailableError(
                RefusalReason.SERVED_IDENTITY_UNKNOWN, f"{type(exc).__name__}: {exc}"
            ) from exc
        if artifact.champion_only:
            # The served pipeline always produces B output (`ServedBuilds.b`), whose features this
            # calibrator never saw: route to REVIEW rather than score them or drop B silently.
            raise CalibratorUnavailableError(
                RefusalReason.CHAMPION_ONLY,
                f"calibrator {artifact.version} was fitted on Extractor A alone; the served "
                f"Extractor B {_describe(served.b)} is uncalibrated",
            )
        mismatch = builds_match(artifact, served)
        if mismatch is not None:
            raise CalibratorUnavailableError(
                RefusalReason.IDENTITY_MISMATCH, f"calibrator {artifact.version}: {mismatch}"
            )
        if self._cached is None or self._cached[0] != digest:
            try:
                calibrator = FrozenCalibrator(artifact.calibrator)
                lambda_hat = artifact.runtime_threshold()
            except TrustGateError as exc:
                raise CalibratorUnavailableError(RefusalReason.CORRUPT, str(exc)) from exc
            self._cached = (digest, CalibratedTrustGate(calibrator, lambda_hat))
        return artifact, self._cached[1]

    def _refuse(
        self, features: Mapping[FieldName, FieldFeatures], exc: CalibratorUnavailableError
    ) -> TrustDecision:
        refusal = str(exc)
        if refusal != self._last_refusal:
            log.warning("trust gate refuses to auto-accept (routes REVIEW): %s", refusal)
            self._last_refusal = refusal
        interim = self._fallback.decide(features)
        return TrustDecision(
            fields=tuple(f.model_copy(update={"accepted": False}) for f in interim.fields),
            all_accepted=False,
            refusal=refusal,
        )


PARSE_TRUNCATED = "parse_truncated"
"""Refusal reason (ADR-0013 amendment 1): the OCR service cut at least one page short."""


def refuse_truncated_parse(
    decision: TrustDecision, truncated_pages: Sequence[int], page_count: int
) -> TrustDecision:
    """`decision` unchanged when no page was truncated. Otherwise no field is auto-accepted and
    `refusal` names the pages (0-based indexes shown 1-based), so the invoice routes to REVIEW:
    values read from a partial transcription are never trusted on calibration alone. A refusal
    the gate already made is kept after this one."""
    if not truncated_pages:
        return decision
    shown = ", ".join(str(index + 1) for index in truncated_pages)
    refusal = f"{PARSE_TRUNCATED}: OCR cut short on page(s) {shown} of {page_count}"
    if decision.refusal is not None:
        refusal = f"{refusal}; {decision.refusal}"
    return decision.model_copy(
        update={
            "fields": tuple(f.model_copy(update={"accepted": False}) for f in decision.fields),
            "all_accepted": False,
            "refusal": refusal,
        }
    )
