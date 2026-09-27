"""Typed exception hierarchy for apx. The money path raises these instead of returning
fallbacks, so every failure is explicit and routable (PRD §8, CLAUDE.md standards)."""

from __future__ import annotations

from enum import StrEnum


class ApxError(Exception):
    """Base of every apx error."""


class ConfigError(ApxError):
    """Missing or invalid configuration."""


class IdempotencyKeyError(ApxError):
    """Malformed idempotency key or step name."""


class IdempotencyConflictError(ApxError):
    """Same idempotency key replayed with a different payload (§6.4)."""


class IngestError(ApxError):
    """Document could not be accepted (unsupported type, unreadable, too large)."""


class FailureNotRecordedError(ApxError):
    """`mark_failed` found neither the invoice row nor the `ingested_documents` row to record a
    workflow failure on. Retryable: the ingest transaction that starts the workflow commits the
    document row just after the start, so a very fast failure can arrive first."""


class ScanError(ApxError):
    """Hidden-content scan could not complete; the document is never treated as clean."""


class ParseError(ApxError):
    """Parser failed to produce words and boxes."""


class ParsedArtifactMissingError(ParseError):
    """A stored parse (document JSON or a page PNG) named by a `ParsedDocumentRef` is not on
    disk. Recoverable: the consuming activity re-parses and checks the result against the ref."""


class ParsedArtifactIntegrityError(ParseError):
    """A stored parse does not match its `ParsedDocumentRef` (corrupt or tampered file, or a
    re-parse that produced different content). Never repaired silently."""


class ModelServingError(ApxError):
    """llama-swap/llama-server or a local transformers model failed or timed out."""


class ModelBusyError(ModelServingError):
    """llama-swap answered 429 (its per-model request limit) or 503 (model still loading) and
    stayed busy through the client's bounded backoff. Transient: Temporal retries it."""


class ModelContextError(ModelServingError):
    """llama-server refused a prompt that does not fit its per-slot context (HTTP 400/500
    "exceed(s) the context size"). Deterministic for the same prompt and serving config, so it
    is never retried; the fix is the serving budget (ADR-0013), not another attempt."""


class ModelNotAllowedError(ConfigError):
    """A request named a model alias outside the runtime allowlist (offline-only models are never
    reachable from the runtime pipeline, PRD §4)."""


class VramBudgetError(ModelServingError):
    """A stage would exceed the 11.5 GB VRAM budget (§2 C2)."""


class GpuProbeError(ModelServingError):
    """Free GPU or host memory could not be measured (`nvidia-smi` missing, failed or printed
    something unparsable; `/proc/meminfo` unreadable). A preflight never passes blind."""


class GpuProbeTransientError(GpuProbeError):
    """`nvidia-smi` was found but timed out or exited non-zero (NVML busy, GPU reset in
    progress): the probe may succeed on a retry, so Temporal retries it (ADR-0013 addendum)."""


class InsufficientGpuMemoryError(ModelServingError):
    """Free VRAM is below a stage's need plus the safety margin (ADR-0013 preflight)."""


class InsufficientHostMemoryError(ModelServingError):
    """Available host RAM is below a stage's need plus the safety margin (ADR-0013 preflight)."""


class ModelOutputError(ApxError):
    """Model output violated the JSON schema or was otherwise unusable."""


class GroundingError(ApxError):
    """A field could not be linked to parse words (I1)."""


class TrustGateError(ApxError):
    """Calibration or threshold selection failed (for example, empty cal split)."""


class ValidationError(ApxError):
    """Validation could not be evaluated (not a rule violation; those are data)."""


class FraudError(ApxError):
    """A fraud rule was misconfigured (for example, a non-positive threshold) or could not be
    evaluated (for example, a page image that could not be decoded for hashing)."""


class EInvoiceError(ApxError):
    """UBL build or schematron validation could not run."""


class AspError(ApxError):
    """Mock ASP rejected or failed a submission."""


class PolicyDeniedError(ApxError):
    """An action was refused by the policy engine (§6.3)."""


class PayeeRefusal(StrEnum):
    """Why P1 could not select a payee (ADR-0014). Carried by PayeeRefusedError and PolicyReason."""

    UNKNOWN_VENDOR = "unknown_vendor"
    """The vendor is None."""
    MASTER_IBAN_MALFORMED = "master_iban_malformed"
    """A verified admin-callback account's stored IBAN fails `electronic_iban`."""
    NO_VERIFIED_ACCOUNT = "no_verified_account"
    """No account has `verified=True` and `source="admin_callback"`."""
    INVOICE_IBAN_MISSING = "invoice_iban_missing"
    """Two or more verified accounts and the invoice IBAN is None or blank."""
    INVOICE_IBAN_MALFORMED = "invoice_iban_malformed"
    """Two or more verified accounts and the invoice IBAN fails `electronic_iban`."""
    NO_MATCH = "no_match"
    """Two or more verified accounts and none equals the invoice IBAN."""
    AMBIGUOUS_MATCH = "ambiguous_match"
    """Two or more verified records equal the invoice IBAN (a data-integrity fault)."""


class PayeeRefusedError(PolicyDeniedError):
    """P1 refused to select a payee; `refusal` is the typed reason (ADR-0014)."""

    def __init__(self, refusal: PayeeRefusal, message: str) -> None:
        super().__init__(message)
        self.refusal: PayeeRefusal = refusal
        self._message = message

    def __reduce__(self) -> tuple[type[PayeeRefusedError], tuple[PayeeRefusal, str]]:
        """Keep the typed refusal across pickling (the default reduce would drop it)."""
        return (type(self), (self.refusal, self._message))


class PayeeChangedError(PolicyDeniedError):
    """The payee `select_payee` returns at e-invoice build differs from the one pinned when the
    invoice was routed (a vendor-master or selection change in between, ADR-0014 §7). The build
    is refused and audited; nothing is paid to either account."""


class SegregationOfDutiesError(PolicyDeniedError):
    """Actor is barred from approving by segregation of duties (P2, I3)."""


class AuditChainError(ApxError):
    """Audit chain append failed or verification found a broken link (I4)."""


class AuditUnavailableError(ApxError):
    """The audit chain or its anchors could not be read or written -- a database connectivity
    failure, a failed query, or a filesystem error on the external anchor file. Distinct from
    `AuditChainError`: this means verification never completed, not that it completed and
    found tampering (I4, §10 M10.4's could-not-run/broken exit-code distinction)."""
