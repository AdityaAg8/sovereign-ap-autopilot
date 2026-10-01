"""Tests for the CORD data guards (ADR-0019 amendment 1, §9.3a): CORD amounts read with the frozen
real-track rule, the CORD "not empty" rule (total + a line item), the receipt identity on the raw
`gt_parse` (tax-inclusive, discount already netted), and UAE/FATURA/DocILE guard results that are
byte-identical with or without the CORD trees."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from ml.extractor_b.data import target_json
from ml.extractor_b.guards import cord_amount, cord_label_sanity, label_sanity, run_guards
from ml.extractor_b.interfaces import GuardConfig, QuarantineReason, SampleSource, TrainingError
from ml.extractor_b.mappings import cord_target
from tests.unit.ml.conftest import RowFactory, good_target

CFG = GuardConfig()
LOOSE = GuardConfig(max_quarantine_rate=1.0)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("61.500", Decimal("61500")),
        ("1.234.500", Decimal("1234500")),
        ("Rp 12,000", Decimal("12000")),
        ("@12.000", Decimal("12000")),
        ("12.50", Decimal("12.5")),
        ("154,000,00", Decimal("154000")),
        ("10% 4,818", Decimal("4818")),
        (["10.000%", "546"], Decimal("546")),
        ("-", None),
        ("", None),
    ],
)
def test_cord_amount(raw: str | list[str], expected: Decimal | None) -> None:
    assert cord_amount(raw) == expected


def _gt(
    sub_total: dict[str, Any] | None, total: str | None, *, menu: bool = True
) -> dict[str, Any]:
    gt: dict[str, Any] = {}
    if menu:
        gt["menu"] = [{"nm": "NASI GORENG", "cnt": "1 x", "price": "50.000"}]
    if sub_total is not None:
        gt["sub_total"] = sub_total
    if total is not None:
        gt["total"] = {"total_price": total, "cashprice": "100.000"}
    return gt


def _check(gt: dict[str, Any]) -> QuarantineReason | None:
    return cord_label_sanity(target_json(cord_target(gt)), gt, CFG)


@pytest.mark.parametrize(
    "gt,reason",
    [
        (_gt({"subtotal_price": "50.000", "tax_price": "5.000"}, "55.000"), None),
        (_gt({"subtotal_price": "25,900", "tax_price": "2,355"}, "25,900"), None),  # inclusive
        (_gt({"subtotal_price": "48,182", "tax_price": "10% 4,818"}, "53,000"), None),
        (
            _gt(
                {
                    "subtotal_price": "650,800",
                    "discount_price": "143,200 -",
                    "service_price": "47,640",
                    "tax_price": "79,400",
                },
                "777,840",
            ),
            None,
        ),  # the subtotal already nets the discount
        (
            _gt(
                {"subtotal_price": "13,801", "discount_price": "3,901", "tax_price": "( 1,236)"},
                "9,900",
            ),
            None,
        ),
        (_gt(None, "36.000"), None),  # no subtotal: nothing to add up
        (_gt({"subtotal_price": "50.000", "service_price": "-"}, "50.000"), None),
        (_gt({"subtotal_price": "50.000"}, None), QuarantineReason.LABEL_EMPTY),
        (_gt({"subtotal_price": "50.000"}, "50.000", menu=False), QuarantineReason.LABEL_EMPTY),
        (
            _gt({"subtotal_price": "134.000", "tax_price": "-"}, "134.000"),
            QuarantineReason.LABEL_FORMAT,
        ),
        (
            _gt({"subtotal_price": "98,180", "discount_price": "9,818"}, "107,998"),
            QuarantineReason.LABEL_ARITHMETIC,
        ),
        (
            _gt({"subtotal_price": "45,545.45", "tax_price": "4,454.55"}, "49,000.00"),
            QuarantineReason.LABEL_ARITHMETIC,
        ),
    ],
)
def test_cord_label_sanity(gt: dict[str, Any], reason: QuarantineReason | None) -> None:
    assert _check(gt) is reason


def test_uae_rule_would_reject_a_sane_receipt() -> None:
    """Why CORD needs its own rule: UAE's 3-header-field floor rejects a well-labelled receipt."""
    gt = _gt({"subtotal_price": "50.000"}, "50.000")
    assert label_sanity(target_json(cord_target(gt)), CFG) is QuarantineReason.LABEL_EMPTY
    assert _check(gt) is None
    dotted = _gt({"subtotal_price": "1.234.500", "tax_price": "123.450"}, "1.357.950")
    assert label_sanity(target_json(cord_target(dotted)), CFG) is QuarantineReason.LABEL_FORMAT
    assert _check(dotted) is None


def test_cord_row_without_its_raw_tree_stops_the_guards(
    make_row: RowFactory, tmp_path: Path
) -> None:
    gt = _gt({"subtotal_price": "50.000", "tax_price": "5.000"}, "55.000")
    row = make_row("cord-0", source=SampleSource.CORD, target=cord_target(gt))
    shas = frozenset({row.document_sha256})
    common: dict[str, Any] = {
        "train_split_sha256": shas,
        "page1_phash": {"cord-0": 1},
        "cfg": LOOSE,
        "max_seq_len": 4096,
        "image_root": tmp_path,
    }
    with pytest.raises(TrainingError, match="no raw gt_parse"):
        run_guards([row], **common)
    kept, quarantined, _ = run_guards([row], cord_trees={row.document_sha256: gt}, **common)
    assert [r.sample_id for r in kept] == ["cord-0"] and quarantined == []
    bad = {**gt, "total": {"total_price": "99.000"}}
    bad_row = make_row("cord-1", source=SampleSource.CORD, target=cord_target(bad))
    kept, quarantined, _ = run_guards(
        [bad_row],
        **{
            **common,
            "train_split_sha256": frozenset({bad_row.document_sha256}),
            "page1_phash": {"cord-1": 1},
        },
        cord_trees={bad_row.document_sha256: bad},
    )
    assert kept == [] and [q.reason for q in quarantined] == [QuarantineReason.LABEL_ARITHMETIC]


def test_non_cord_guard_results_are_byte_identical(make_row: RowFactory, tmp_path: Path) -> None:
    """UAE, FATURA and DocILE rows keep the v1 `label_sanity`: same kept rows, same quarantine,
    same manifest hash with or without CORD trees."""
    rows = [
        make_row("u-ok", target=good_target()),
        make_row(
            "u-empty",
            target=good_target(
                invoice_number=None,
                issue_date=None,
                supplier_name=None,
                supplier_trn=None,
                currency=None,
                subtotal=None,
                vat_total=None,
            ),
        ),
        make_row("u-arith", target=good_target(total="999.00")),
        make_row("f-ok", source=SampleSource.FATURA, target=good_target(invoice_number="F-2")),
        make_row("d-fmt", source=SampleSource.DOCILE, target=good_target(currency="ZZZ")),
    ]
    common: dict[str, Any] = {
        "train_split_sha256": frozenset(r.document_sha256 for r in rows),
        "page1_phash": {r.sample_id: n * 0xFFFF for n, r in enumerate(rows)},
        "cfg": LOOSE,
        "max_seq_len": 4096,
        "image_root": tmp_path,
    }
    plain = run_guards(rows, **common)
    with_trees = run_guards(rows, cord_trees={"0" * 64: {"menu": []}}, **common)
    assert plain == with_trees
    assert {q.sample_id: q.reason for q in plain[1]} == {
        "u-empty": QuarantineReason.LABEL_EMPTY,
        "u-arith": QuarantineReason.LABEL_ARITHMETIC,
        "d-fmt": QuarantineReason.LABEL_FORMAT,
    }
