"""Typed exception hierarchy for apx. The money path raises these instead of returning
fallbacks, so every failure is explicit and routable (PRD §8, CLAUDE.md standards)."""

from __future__ import annotations


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


class SegregationOfDutiesError(PolicyDeniedError):
    """Actor is barred from approving by segregation of duties (P2, I3)."""


class AuditChainError(ApxError):
    """Audit chain append failed or verification found a broken link (I4)."""


class AuditUnavailableError(ApxError):
    """The audit chain or its anchors could not be read or written -- a database connectivity
    failure, a failed query, or a filesystem error on the external anchor file. Distinct from
    `AuditChainError`: this means verification never completed, not that it completed and
    found tampering (I4, §10 M10.4's could-not-run/broken exit-code distinction)."""
