"""DocILE val for the real-data track (ADR-0019 §1-3): the 200-document subset stratified by layout
cluster (seed 0), gold through the existing `DOCILE_KILE`/`DOCILE_LIR` tables, and text-only KILE
and LIR field F1 (the official location-aware AP is not computed, and the report says so)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from apx.extract.interfaces import PRINTED_LINE_ITEM_KEYS
from evals.real.models import DatasetMissingError
from evals.real.normalise import ValueKind, normalise
from ml.extractor_b.mappings import DOCILE_KILE, DOCILE_LIR, docile_target

SUBSET_SIZE: Final = 200
SEED: Final = 0
_AMOUNTS: Final[frozenset[str]] = frozenset(
    {"subtotal", "vat_total", "total", "unit_price", "net_amount"}
)
KILE_FIELDS: Final[tuple[str, ...]] = tuple(sorted(set(DOCILE_KILE.values())))
LIR_FIELDS: Final[tuple[str, ...]] = tuple(
    k for k in PRINTED_LINE_ITEM_KEYS if k in set(DOCILE_LIR.values())
)


def kind_of(field: str) -> ValueKind:
    if field in _AMOUNTS:
        return "amount"
    return "count" if field == "quantity" else "text"


def _root(data_root: Path) -> Path:
    root = data_root / "docile"
    if not (root / "val.json").is_file() or not (root / "annotations").is_dir():
        raise DatasetMissingError(
            f"DocILE val is not fetched under {root}; set DOCILE_TOKEN and run "
            "`uv run python -m evals.datasets fetch --set docile`"
        )
    return root


def load_annotation(data_root: Path, doc_id: str) -> dict[str, Any]:
    path = _root(data_root) / "annotations" / f"{doc_id}.json"
    annotation: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return annotation


def pdf_path(data_root: Path, doc_id: str) -> Path:
    return _root(data_root) / "pdfs" / f"{doc_id}.pdf"


def doc_sha256(data_root: Path, doc_id: str) -> str:
    return hashlib.sha256(pdf_path(data_root, doc_id).read_bytes()).hexdigest()


def _cluster(annotation: Mapping[str, Any]) -> str:
    metadata = annotation.get("metadata") or {}
    return str(metadata.get("cluster_id", "unclustered"))


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of `n` over strata (largest remainder, ties by stratum name),
    never more than a stratum holds."""
    total = sum(sizes.values())
    if total <= n:
        return dict(sizes)
    quotas = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for key in by_remainder[: n - sum(alloc.values())]:
        alloc[key] += 1
    return alloc


def docile_subset(data_root: Path, n: int = SUBSET_SIZE, seed: int = SEED) -> list[str]:
    """Val docs stratified by layout cluster; DatasetMissingError when DocILE is not fetched."""
    root = _root(data_root)
    val_ids = sorted(str(d) for d in json.loads((root / "val.json").read_text(encoding="utf-8")))
    strata: dict[str, list[str]] = defaultdict(list)
    for doc_id in val_ids:
        strata[_cluster(load_annotation(data_root, doc_id))].append(doc_id)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = random.Random(seed)
    chosen: list[str] = []
    for cluster in sorted(strata):
        chosen.extend(rng.sample(strata[cluster], alloc[cluster]))
    return sorted(chosen)


def mapped_counts(annotation: Mapping[str, Any]) -> tuple[int, int]:
    """(mapped gold instances, all gold instances) of one document."""
    kile = [e["fieldtype"] for e in annotation.get("field_extractions", [])]
    lir = [e["fieldtype"] for e in annotation.get("line_item_extractions", [])]
    mapped = sum(f in DOCILE_KILE for f in kile) + sum(f in DOCILE_LIR for f in lir)
    return mapped, len(kile) + len(lir)


def _counts(p: Counter[Any], g: Counter[Any]) -> tuple[int, int, int]:
    tp = sum((p & g).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def _header_pairs(target: Mapping[str, Any]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for field in KILE_FIELDS:
        value = normalise(kind_of(field), _str(target.get(field)))
        if value is not None:
            pairs[(field, value)] += 1
    return pairs


def _line_pairs(target: Mapping[str, Any]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for item in target.get("line_items") or []:
        if not isinstance(item, Mapping):
            continue
        for field in LIR_FIELDS:
            value = normalise(kind_of(field), _str(item.get(field)))
            if value is not None:
                pairs[(field, value)] += 1
    return pairs


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def kile_counts(
    prediction: Mapping[str, Any] | None, annotation: Mapping[str, Any]
) -> tuple[int, int, int]:
    """Text-only KILE (tp, fp, fn): normalised (field, value) multisets over mapped fields."""
    return _counts(_header_pairs(prediction or {}), _header_pairs(docile_target(annotation)))


def lir_counts(
    prediction: Mapping[str, Any] | None, annotation: Mapping[str, Any]
) -> tuple[int, int, int]:
    """Text-only LIR (tp, fp, fn) over all line values, line grouping not required."""
    return _counts(_line_pairs(prediction or {}), _line_pairs(docile_target(annotation)))
