"""Predictor interface the eval harness scores against (PRD §5, §10 M1.4): anything that maps a
document id to predicted field values, so evals/run.py can swap in a real model later without
changing how any scorer is called."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class Predictor(Protocol):
    """A predictor extracts field values for one document, keyed by field name. `name`
    identifies it in reports/runs/ filenames and JSON output."""

    name: str

    def predict_fields(self, doc_id: str) -> Mapping[str, str]:
        """Returns predicted `{field_name: value}` for `doc_id`. Never raises for an unknown
        doc_id; an empty mapping means "nothing predicted" (every gold field becomes a false
        negative when scored)."""
        ...
