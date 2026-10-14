"""Typed errors for `apx.learning`: the retraining export never silently drops or widens
which rows it reads (I5)."""

from __future__ import annotations

from apx.common.errors import ApxError


class SplitViolationError(ApxError):
    """A row that is not on the training split reached the retraining export. The export
    query already filters for the training split before this point, so this signals a bug in
    that filter rather than a normal data condition -- it exists so a future change can never
    silently widen the export to read eval data (I5) without a loud failure."""
