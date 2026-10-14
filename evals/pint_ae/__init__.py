"""PINT AE (UAE) schematron validation for the evaluation harness: fetches
and pins the official OpenPeppol validation artifacts, compiles the schematron to XSLT with
SchXslt on saxonche, and exposes a typed validator that returns SVRL failed-asserts."""

from evals.pint_ae.compile import SchematronCompileError, compile_schematron
from evals.pint_ae.fetch import PinMismatchError, ensure_pint_ae, ensure_schxslt, load_pin
from evals.pint_ae.validator import (
    FailedAssert,
    PintAeValidator,
    ValidationResult,
    load_pint_ae_validator,
)

__all__ = [
    "FailedAssert",
    "PinMismatchError",
    "PintAeValidator",
    "SchematronCompileError",
    "ValidationResult",
    "compile_schematron",
    "ensure_pint_ae",
    "ensure_schxslt",
    "load_pin",
    "load_pint_ae_validator",
]
