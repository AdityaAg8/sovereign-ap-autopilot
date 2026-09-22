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


class ModelServingError(ApxError):
    """llama-swap/llama-server or a local transformers model failed or timed out."""


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
