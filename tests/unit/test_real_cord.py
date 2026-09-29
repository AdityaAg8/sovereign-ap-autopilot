"""Unit tests for the CORD v2 side of the real-data track (ADR-0019 §2-3): mapping totality over
every key in the pinned revision, the key mapping of an apx.extract answer, Donut-style multiset
field counts, Zhang-Shasha nTED accuracy and the index loader. Pure CPU, no dataset needed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evals.real import cord
from evals.real.cord import _Node, tree_edit_distance
from evals.real.models import DatasetMissingError
from ml.extractor_b.mappings import (
    CORD_FIELDS,
    CORD_IGNORED,
    cord_leaves,
    unmapped_cord_keys,
)

# Every leaf key in naver-clova-ix/cord-v2 @ 7f0115a4 (all 1000 rows, enumerated 2026-09-29).
CORD_KEYS_AT_REVISION = (
    "menu.cnt", "menu.discountprice", "menu.etc", "menu.itemsubtotal", "menu.nm", "menu.num",
    "menu.price", "menu.sub.cnt", "menu.sub.nm", "menu.sub.price", "menu.sub.unitprice",
    "menu.unitprice", "menu.vatyn", "sub_total.discount_price", "sub_total.etc",
    "sub_total.othersvc_price", "sub_total.service_price", "sub_total.subtotal_price",
    "sub_total.tax_price", "total.cashprice", "total.changeprice", "total.creditcardprice",
    "total.emoneyprice", "total.menuqty_cnt", "total.menutype_cnt", "total.total_etc",
    "total.total_price", "void_menu.nm", "void_menu.price",
)  # fmt: skip

GOLD: dict[str, Any] = {
    "menu": [
        {"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"},
        {"nm": "EGG TART", "cnt": "2", "unitprice": "6,500", "price": "13,000",
         "sub": {"nm": "EXTRA", "price": "0"}},
    ],
    "sub_total": {"subtotal_price": "29,500", "tax_price": "2,950"},
    "total": {"total_price": "32,450", "cashprice": "50,000", "changeprice": "17,550"},
}  # fmt: skip
PRED: dict[str, Any] = {
    "invoice_number": None,
    "subtotal": "29.500",
    "vat_total": "2.950",
    "total": "Rp 32,450",
    "line_items": [
        {"description": "REAL GANACHE", "quantity": "1", "unit_price": None,
         "net_amount": "16.500", "vat_rate": None, "vat_category": None},
        {"description": "EGG TART", "quantity": "2x", "unit_price": "6.500",
         "net_amount": "13.000", "vat_rate": None, "vat_category": None},
    ],
}  # fmt: skip


def test_every_cord_key_is_mapped_or_ignored() -> None:
    known = {*CORD_FIELDS, *CORD_IGNORED}
    assert [k for k in CORD_KEYS_AT_REVISION if k not in known] == []
    assert not set(CORD_FIELDS) & CORD_IGNORED


def test_unmapped_cord_keys_flags_a_new_label() -> None:
    assert unmapped_cord_keys(GOLD) == []
    assert unmapped_cord_keys({"total": {"tip_price": "1"}}) == ["total.tip_price"]


def test_cord_leaves_walks_objects_and_lists() -> None:
    leaves = list(cord_leaves({"menu": {"nm": "A", "sub": [{"nm": "B"}]}, "x": ["1", "2"]}))
    assert leaves == [("menu.nm", "A"), ("menu.sub.nm", "B"), ("x", "1"), ("x", "2")]


def test_mapping_covers_the_adr_table() -> None:
    assert CORD_FIELDS["menu.price"] == "line_items[].net_amount"
    assert CORD_FIELDS["total.total_price"] == "total"
    assert len(CORD_FIELDS) == 7


def test_to_cord_keys_maps_header_and_lines_and_drops_the_rest() -> None:
    keys = cord.to_cord_keys(PRED)
    assert keys["menu.nm"] == ["REAL GANACHE", "EGG TART"]
    assert keys["menu.unitprice"] == ["6.500"]
    assert keys["total.total_price"] == ["Rp 32,450"]
    assert "invoice_number" not in keys


def test_field_counts_identical_after_normalisation() -> None:
    tp, fp, fn = cord.field_counts(cord.to_cord_keys(PRED), cord.gold_keys(GOLD))
    assert (fp, fn) == (0, 0)
    assert tp == 10


def test_field_counts_empty_prediction_misses_everything() -> None:
    tp, fp, fn = cord.field_counts({}, cord.gold_keys(GOLD))
    assert (tp, fp) == (0, 0)
    assert fn == 10


def test_field_counts_is_a_multiset() -> None:
    gold = {"sub_total.subtotal_price": ["46.636", "46.636"]}
    assert cord.field_counts({"sub_total.subtotal_price": ["46,636"]}, gold) == (1, 0, 1)
    assert cord.field_counts({"menu.nm": ["X"]}, {}) == (0, 1, 0)


def test_mapped_counts_share() -> None:
    assert cord.mapped_counts(GOLD) == (10, 14)


def test_ted_accuracy_identical_is_one_and_empty_is_zero() -> None:
    gold_tree = cord.gold_tree(GOLD)
    assert cord.ted_accuracy(cord.prediction_tree(PRED), gold_tree) == 1.0
    assert cord.ted_accuracy({}, gold_tree) == 0.0
    assert cord.ted_accuracy({}, {}) == 1.0
    assert cord.ted_accuracy({"total": {"total_price": "1"}}, {}) == 0.0


def test_ted_accuracy_partial_answer_is_between() -> None:
    partial = dict(PRED, line_items=PRED["line_items"][:1])
    value = cord.ted_accuracy(cord.prediction_tree(partial), cord.gold_tree(GOLD))
    assert 0.0 < value < 1.0


def test_gold_tree_keeps_values_printed_twice() -> None:
    tree = cord.gold_tree({"sub_total": {"subtotal_price": ["46.636", "46.636"]}})
    assert tree == {"sub_total": {"subtotal_price": ["46636", "46636"]}}


def test_zhang_shasha_known_distances() -> None:
    a = _Node("f", [_Node("d", [_Node("a"), _Node("c", [_Node("b")])]), _Node("e")])
    b = _Node("f", [_Node("c", [_Node("d", [_Node("a"), _Node("b")])]), _Node("e")])
    assert tree_edit_distance(a, b) == 2  # the classic Zhang-Shasha example
    assert tree_edit_distance(a, a) == 0
    assert tree_edit_distance(_Node("r"), a) == 6  # relabel the root, insert 5


_values = st.text(alphabet="0123456789,.ABC ", min_size=1, max_size=6)
_trees = st.fixed_dictionaries(
    {},
    optional={
        "menu": st.lists(
            st.fixed_dictionaries({}, optional={"nm": _values, "cnt": _values, "price": _values}),
            max_size=4,
        ),
        "total": st.fixed_dictionaries({}, optional={"total_price": _values}),
    },
)


@settings(max_examples=60, deadline=None)
@given(_trees, _trees)
def test_ted_accuracy_is_within_unit_interval(pred: dict[str, Any], gold: dict[str, Any]) -> None:
    value = cord.ted_accuracy(cord.gold_tree(pred), cord.gold_tree(gold))
    assert 0.0 <= value <= 1.0


def _index(root: Path, split: str, rows: list[dict[str, Any]]) -> None:
    folder = root / "cord" / split
    folder.mkdir(parents=True)
    (folder / "index.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_load_cord_and_dev_subset(tmp_path: Path) -> None:
    rows = [
        {
            "row": i,
            "sha256": f"{c}" * 64,
            "image": f"{c * 64}.png",
            "ground_truth": {"gt_parse": {}},
        }
        for i, c in enumerate("fedcba9876543210fedcba")
    ]
    _index(tmp_path, "validation", rows)
    docs = cord.load_cord("validation", tmp_path)
    assert len(docs) == 22
    dev = cord.dev_subset(docs)
    assert len(dev) == 16  # 22 rows, 16 distinct images: duplicates count once
    assert [d.sha256 for d in dev] == sorted(d.sha256 for d in dev)


def test_load_cord_missing_is_typed(tmp_path: Path) -> None:
    with pytest.raises(DatasetMissingError, match="fetch --set cord"):
        cord.load_cord("test", tmp_path)
