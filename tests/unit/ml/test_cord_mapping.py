"""Tests for the CORD v2 → Extractor B target mapping and the per-source loss-mask key sets
(ADR-0019 §2, §9.3): the tables are total over every CORD v2 label and disjoint, targets keep
values as printed with one line per `menu` entry, unknown labels raise, and the mask keys are
exactly the keys each source does not annotate (none for UAE and FATURA)."""

from __future__ import annotations

from typing import Any

import pytest

from apx.extract.interfaces import INVOICE_TYPE_CODE_KEY, LINE_ITEM_KEYS
from ml.extractor_b.interfaces import SampleSource, TrainingError
from ml.extractor_b.mappings import (
    ALL_TARGET_KEYS,
    CORD_FIELDS,
    CORD_IGNORED,
    CORD_V2_LABELS,
    HEADER_KEYS,
    UNANNOTATED,
    cord_label_known,
    cord_target,
    empty_target,
)
from ml.extractor_b.tokens import LINE_KEY_PREFIX

RECEIPT: dict[str, Any] = {
    "menu": [
        {"nm": "NASI GORENG", "cnt": "2", "unitprice": "25.000", "price": "50.000"},
        {
            "nm": ["ES", "TEH"],
            "cnt": "1",
            "price": "5.000",
            "sub": [{"nm": "less sugar", "price": "0"}],
        },
        {"num": "#12", "etc": "note"},  # no mapped key: not a line
    ],
    "void_menu": {"nm": "KERUPUK", "price": "2.000"},
    "sub_total": {"subtotal_price": "55.000", "tax_price": "5.500", "service_price": "1.000"},
    "total": {"total_price": "61.500", "cashprice": "100.000", "changeprice": "38.500"},
}


def test_every_cord_v2_label_is_mapped_or_ignored_and_never_both() -> None:
    assert all(cord_label_known(label) for label in CORD_V2_LABELS)
    assert not set(CORD_FIELDS) & CORD_IGNORED
    explicit = {label for label in CORD_IGNORED if not label.endswith(".*")}
    assert set(CORD_FIELDS) | explicit <= CORD_V2_LABELS
    for wildcard in (label for label in CORD_IGNORED if label.endswith(".*")):
        prefix = wildcard.removesuffix("*")
        assert any(label.startswith(prefix) for label in CORD_V2_LABELS)
        assert not any(label.startswith(prefix) for label in CORD_FIELDS)


def test_cord_fields_map_to_three_header_totals_and_four_line_keys() -> None:
    header = {v for v in CORD_FIELDS.values() if not v.startswith(LINE_KEY_PREFIX)}
    lines = {v.removeprefix(LINE_KEY_PREFIX) for v in CORD_FIELDS.values()} - header
    assert header == {"subtotal", "vat_total", "total"} and header <= set(HEADER_KEYS)
    assert lines == {"description", "quantity", "unit_price", "net_amount"}
    assert lines <= set(LINE_ITEM_KEYS)


def test_cord_target_keeps_values_as_printed() -> None:
    target = cord_target(RECEIPT)
    assert list(target) == list(empty_target())
    assert (target["subtotal"], target["vat_total"], target["total"]) == (
        "55.000",
        "5.500",
        "61.500",
    )
    assert target[INVOICE_TYPE_CODE_KEY] is None
    assert all(
        target[k] is None for k in HEADER_KEYS if k not in {"subtotal", "vat_total", "total"}
    )
    assert target["line_items"] == [
        {
            "description": "NASI GORENG",
            "quantity": "2",
            "unit_price": "25.000",
            "net_amount": "50.000",
            "vat_rate": None,
            "vat_category": None,
        },
        {
            "description": "ES TEH",
            "quantity": "1",
            "unit_price": None,
            "net_amount": "5.000",
            "vat_rate": None,
            "vat_category": None,
        },
    ]


def test_a_single_menu_object_is_one_line() -> None:
    target = cord_target(
        {"menu": {"nm": "KOPI", "price": "18.000"}, "total": {"total_price": "18.000"}}
    )
    assert [line["description"] for line in target["line_items"]] == ["KOPI"]
    assert target["total"] == "18.000"


@pytest.mark.parametrize(
    "gt_parse",
    [
        {"menu": [{"nm": "A", "colour": "red"}]},
        {"total": {"total_price": "1", "tip_price": "2"}},
        {"receipt_header": {"store": "X"}},
        {"menu": [{"nm": "A", "extra": {"x": "1"}}]},
    ],
)
def test_an_unknown_cord_label_raises(gt_parse: dict[str, Any]) -> None:
    with pytest.raises(TrainingError, match="unmapped CORD label"):
        cord_target(gt_parse)


def test_a_non_text_cord_value_raises() -> None:
    with pytest.raises(TrainingError, match="non-text"):
        cord_target({"total": {"total_price": 61500}})


def test_unannotated_keys_per_source() -> None:
    assert set(UNANNOTATED) == set(SampleSource)
    assert UNANNOTATED[SampleSource.UAE] == frozenset()
    assert UNANNOTATED[SampleSource.FATURA] == frozenset()
    assert UNANNOTATED[SampleSource.DOCILE] == {
        INVOICE_TYPE_CODE_KEY,
        f"{LINE_KEY_PREFIX}vat_category",
    }
    cord = UNANNOTATED[SampleSource.CORD]
    assert cord == ALL_TARGET_KEYS - set(CORD_FIELDS.values())
    assert len(ALL_TARGET_KEYS) == len(HEADER_KEYS) + 1 + len(LINE_ITEM_KEYS)
    assert cord == {k for k in HEADER_KEYS if k not in {"subtotal", "vat_total", "total"}} | {
        INVOICE_TYPE_CODE_KEY,
        f"{LINE_KEY_PREFIX}vat_rate",
        f"{LINE_KEY_PREFIX}vat_category",
    }
