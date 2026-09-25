"""UAE test micro field F1 over the 14 header fields for rows written by `evals.extraction_a`
(ADR-0009 §1.8 P1): gold from the factory `*.labels.json`, language from `*.truth.json`
(`arabic_first`), scored by the M1.3 scorer `evals.scorers.uae.score_uae` (EN, AR and pooled).
Extractor A and B rows go through the same function, so the P1 comparison is like for like."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from apx.common.errors import ApxError
from apx.extract.interfaces import FieldName
from evals.scorers.uae import Language, UaeDocument, UaeFieldF1Result, score_uae

HEADER_FIELDS: tuple[str, ...] = tuple(name.value for name in FieldName)


class RowsIncompleteError(ApxError):
    """An eval row file does not cover every requested document, so no F1 can be claimed."""


def ids_digest(shas: Iterable[str]) -> str:
    """sha256 over the sorted document sha256s, one per line (ADR-0015 §1 `uae_test_digest`)."""
    return hashlib.sha256("".join(f"{sha}\n" for sha in sorted(shas)).encode("ascii")).hexdigest()


def uae_gold(data_dir: Path, shas: Sequence[str]) -> list[UaeDocument]:
    """Gold header fields and language per document of a factory output directory."""
    documents: list[UaeDocument] = []
    for sha in shas:
        labels = json.loads((data_dir / f"{sha}.labels.json").read_text(encoding="utf-8"))
        truth = json.loads((data_dir / f"{sha}.truth.json").read_text(encoding="utf-8"))
        lang: Language = "ar" if truth["arabic_first"] else "en"
        fields = {
            label["field"]: label["value"]
            for label in labels["labels"]
            if label["field"] in HEADER_FIELDS and isinstance(label["value"], str)
        }
        documents.append(UaeDocument(doc_id=sha, lang=lang, fields=fields))
    return documents


def row_predictions(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """Predicted header fields per document. An invalid-JSON row predicts nothing, so each of
    its gold fields is a miss (the same rule as `ml.extractor_b.evaluator.score_generations`)."""
    predicted: dict[str, dict[str, str]] = {}
    for sha, row in rows.items():
        if not row["valid_json"]:
            continue
        predicted[sha] = {
            name: cell["pred"]
            for name, cell in row["fields"].items()
            if name in HEADER_FIELDS and isinstance(cell["pred"], str) and cell["pred"].strip()
        }
    return predicted


def require_rows(rows: Mapping[str, Mapping[str, Any]], shas: Sequence[str], what: str) -> None:
    missing = [sha for sha in shas if sha not in rows]
    if missing:
        raise RowsIncompleteError(
            f"{what}: {len(missing)} of {len(shas)} documents have no eval row, e.g. {missing[0]}"
        )


def score_rows(
    rows: Mapping[str, Mapping[str, Any]], data_dir: Path, shas: Sequence[str], what: str
) -> UaeFieldF1Result:
    """`score_uae` over exactly `shas`; every document must have a row."""
    require_rows(rows, shas, what)
    selected = {sha: rows[sha] for sha in shas}
    return score_uae(uae_gold(data_dir, shas), row_predictions(selected))


def valid_json_rate(rows: Mapping[str, Mapping[str, Any]], shas: Sequence[str]) -> float:
    if not shas:
        raise RowsIncompleteError("no documents to rate")
    return sum(bool(rows[sha]["valid_json"]) for sha in shas) / len(shas)
