"""CORD v2 scoring for the real-data track (ADR-0019 §2-3): loads the receipts `cord_fetcher`
unpacked, maps an `apx.extract` answer onto CORD keys through `CORD_FIELDS`, and scores it with a
Donut-style multiset field F1 and normalised tree-edit-distance accuracy (Zhang-Shasha, in-repo)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from evals.real.models import DatasetMissingError
from evals.real.normalise import ValueKind, normalise
from ml.extractor_b.mappings import CORD_FIELDS, cord_leaves

CordSplit = Literal["validation", "test", "train"]
DEV_DOCS: Final = 20
LINE_PREFIX: Final = "line_items[]."
CORD_KIND: Final[Mapping[str, ValueKind]] = {
    key: ("text" if key == "menu.nm" else "count" if key == "menu.cnt" else "amount")
    for key in CORD_FIELDS
}
_GROUPS: Final[tuple[str, ...]] = ("menu", "sub_total", "total")


@dataclass(frozen=True, slots=True)
class CordDoc:
    """One receipt: its row in the split, image sha256 and file, and `gt_parse`."""

    row: int
    sha256: str
    image: Path
    gt_parse: Mapping[str, Any]


def load_cord(split: CordSplit, data_root: Path) -> list[CordDoc]:
    """Receipts of `split` in row order from data/cord/<split>/index.jsonl."""
    index = data_root / "cord" / split / "index.jsonl"
    if not index.is_file():
        raise DatasetMissingError(
            f"{index} is missing; run `uv run python -m evals.datasets fetch --set cord "
            f"--splits {split}`"
        )
    docs: list[CordDoc] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        docs.append(
            CordDoc(
                row=int(record["row"]),
                sha256=str(record["sha256"]),
                image=index.parent / str(record["image"]),
                gt_parse=record["ground_truth"]["gt_parse"],
            )
        )
    return docs


def unique_docs(docs: Sequence[CordDoc]) -> list[CordDoc]:
    """One document per image sha256 (the first row wins): a row set keyed by sha256 scores a
    duplicated image once."""
    seen: dict[str, CordDoc] = {}
    for doc in docs:
        seen.setdefault(doc.sha256, doc)
    return list(seen.values())


def dev_subset(docs: Sequence[CordDoc], n: int = DEV_DOCS) -> list[CordDoc]:
    """The first `n` validation receipts by sha256 (ADR-0019 §1: mapping/normaliser work only)."""
    return sorted(unique_docs(docs), key=lambda d: d.sha256)[:n]


def gold_keys(gt_parse: Mapping[str, Any]) -> dict[str, list[str]]:
    """Mapped CORD keys of a gold parse with their printed values, in document order."""
    out: dict[str, list[str]] = {}
    for key, value in cord_leaves(gt_parse):
        if key in CORD_FIELDS:
            out.setdefault(key, []).append(value)
    return out


def mapped_counts(gt_parse: Mapping[str, Any]) -> tuple[int, int]:
    """(mapped gold instances, all gold instances) of one document."""
    keys = [key for key, _ in cord_leaves(gt_parse)]
    return sum(key in CORD_FIELDS for key in keys), len(keys)


def to_cord_keys(extraction: Mapping[str, Any]) -> dict[str, list[str]]:
    """apx.extract output -> CORD keys via ml.extractor_b.mappings.CORD_FIELDS; unmapped dropped."""
    out: dict[str, list[str]] = {}
    items = extraction.get("line_items") or []
    for cord_key, ours in CORD_FIELDS.items():
        if ours.startswith(LINE_PREFIX):
            line_key = ours.removeprefix(LINE_PREFIX)
            values = [item.get(line_key) for item in items if isinstance(item, Mapping)]
        else:
            values = [extraction.get(ours)]
        kept = [str(v) for v in values if v is not None]
        if kept:
            out[cord_key] = kept
    return out


def _pairs(keys: Mapping[str, Sequence[str]]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for key, values in keys.items():
        if key not in CORD_KIND:
            continue
        for value in values:
            norm = normalise(CORD_KIND[key], value)
            if norm is not None:
                pairs[(key, norm)] += 1
    return pairs


def field_counts(
    pred: Mapping[str, Sequence[str]], gold: Mapping[str, Sequence[str]]
) -> tuple[int, int, int]:
    """Donut-style multiset (tp, fp, fn) over normalised (key, value) pairs, mapped keys only."""
    p, g = _pairs(pred), _pairs(gold)
    tp = sum((p & g).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


# --- trees -------------------------------------------------------------------------------------


def _norm_leaf(group: str, key: str, value: Any) -> str | list[str] | None:
    """A mapped leaf normalised; CORD labels a value printed twice as a list of strings."""
    full = f"{group}.{key}"
    if full not in CORD_KIND or value is None:
        return None
    if isinstance(value, list):
        kept = [n for v in value if (n := normalise(CORD_KIND[full], str(v))) is not None]
        return kept or None
    return normalise(CORD_KIND[full], str(value))


def gold_tree(gt_parse: Mapping[str, Any]) -> dict[str, Any]:
    """The mapped part of a gold parse as a tree (menu items as a list), values normalised."""
    tree: dict[str, Any] = {}
    for group in _GROUPS:
        node = gt_parse.get(group)
        entries = node if isinstance(node, list) else [node] if isinstance(node, Mapping) else []
        kept = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            leaves = {k: v for k in entry if (v := _norm_leaf(group, k, entry[k])) is not None}
            if leaves:
                kept.append(leaves)
        if kept:
            tree[group] = kept if group == "menu" else kept[0]
    return tree


def prediction_tree(extraction: Mapping[str, Any]) -> dict[str, Any]:
    """An apx.extract answer in the same mapped tree shape as `gold_tree`."""
    menu: list[dict[str, Any]] = []
    for item in extraction.get("line_items") or []:
        if not isinstance(item, Mapping):
            continue
        leaves: dict[str, Any] = {}
        for cord_key, ours in CORD_FIELDS.items():
            if ours.startswith(LINE_PREFIX):
                group, key = cord_key.split(".", 1)
                value = _norm_leaf(group, key, item.get(ours.removeprefix(LINE_PREFIX)))
                if value is not None:
                    leaves[key] = value
        if leaves:
            menu.append(leaves)
    tree: dict[str, Any] = {"menu": menu} if menu else {}
    for cord_key, ours in CORD_FIELDS.items():
        if ours.startswith(LINE_PREFIX):
            continue
        group, key = cord_key.split(".", 1)
        value = _norm_leaf(group, key, extraction.get(ours))
        if value is not None:
            tree.setdefault(group, {})[key] = value
    return tree


@dataclass
class _Node:
    label: str
    children: list[_Node] = field(default_factory=list)


def _to_node(label: str, value: Any) -> _Node:
    if isinstance(value, Mapping):
        return _Node(label, [_to_node(str(k), value[k]) for k in sorted(value)])
    if isinstance(value, list):
        return _Node(label, [_to_node("<item>", v) for v in value])
    return _Node(label, [_Node(str(value))])


def _postorder(root: _Node) -> tuple[list[str], list[int]]:
    """Labels in postorder and each node's leftmost-leaf postorder index."""
    labels: list[str] = []
    leftmost: list[int] = []

    def walk(node: _Node) -> int:
        first = -1
        for child in node.children:
            left = walk(child)
            if first < 0:
                first = left
        labels.append(node.label)
        index = len(labels) - 1
        leftmost.append(index if first < 0 else first)
        return leftmost[index]

    walk(root)
    return labels, leftmost


def tree_edit_distance(a: _Node, b: _Node) -> int:
    """Zhang-Shasha ordered tree edit distance, unit insert/delete/relabel costs."""
    la, lla = _postorder(a)
    lb, llb = _postorder(b)
    keyroots_a = sorted({max(i for i in range(len(la)) if lla[i] == v) for v in set(lla)})
    keyroots_b = sorted({max(j for j in range(len(lb)) if llb[j] == v) for v in set(llb)})
    tree_dist = [[0] * len(lb) for _ in range(len(la))]
    for i in keyroots_a:
        for j in keyroots_b:
            li, lj = lla[i], llb[j]
            rows, cols = i - li + 2, j - lj + 2
            forest = [[0] * cols for _ in range(rows)]
            for x in range(1, rows):
                forest[x][0] = forest[x - 1][0] + 1
            for y in range(1, cols):
                forest[0][y] = forest[0][y - 1] + 1
            for x in range(1, rows):
                for y in range(1, cols):
                    ai, bj = li + x - 1, lj + y - 1
                    if lla[ai] == li and llb[bj] == lj:
                        forest[x][y] = min(
                            forest[x - 1][y] + 1,
                            forest[x][y - 1] + 1,
                            forest[x - 1][y - 1] + (la[ai] != lb[bj]),
                        )
                        tree_dist[ai][bj] = forest[x][y]
                    else:
                        forest[x][y] = min(
                            forest[x - 1][y] + 1,
                            forest[x][y - 1] + 1,
                            forest[lla[ai] - li][llb[bj] - lj] + tree_dist[ai][bj],
                        )
    return tree_dist[len(la) - 1][len(lb) - 1]


def ted_accuracy(pred: Mapping[str, Any], gold: Mapping[str, Any]) -> float:
    """max(0, 1 - TED(pred, gold) / TED(empty, gold)); Zhang-Shasha in-repo, no new dependency.
    With an empty gold tree it is 1.0 for an empty prediction and 0.0 otherwise."""
    gold_root = _to_node("<root>", gold)
    distance = tree_edit_distance(_to_node("<root>", pred), gold_root)
    baseline = tree_edit_distance(_Node("<root>"), gold_root)
    if baseline == 0:
        return 1.0 if distance == 0 else 0.0
    return max(0.0, 1.0 - distance / baseline)
