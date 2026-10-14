"""Mock-ASP-specific error carrying the schematron rule IDs that failed a submission, layered on
`apx.common.errors.AspError` so callers can still catch the common ASP error type."""

from __future__ import annotations

from apx.common.errors import AspError
from apx.einvoice.interfaces import SchematronFailure


class DocumentRejectedError(AspError):
    """UBL XML failed the runtime PINT AE schematron gate. `failures` carries the fatal rule
    violations (rule_id, flag, location, message) so the API layer can report them instead of a
    generic message."""

    def __init__(self, failures: tuple[SchematronFailure, ...]) -> None:
        self.failures = failures
        rule_ids = ", ".join(f.rule_id for f in failures) or "unknown"
        super().__init__(f"schematron validation failed: {rule_ids}")
