"""The M1.4 baseline predictor
("a dummy predictor is scored end to end on every available set"): always predicts no fields, so
every scorer's real, unmocked code path -- and
its vacuous-truth conventions for "nothing predicted" -- runs at least once before any trained
extractor exists."""

from __future__ import annotations

from collections.abc import Mapping


class DummyPredictor:
    """Predicts an empty field mapping for every document, regardless of its id or content."""

    name = "dummy"

    def predict_fields(self, doc_id: str) -> Mapping[str, str]:
        return {}
