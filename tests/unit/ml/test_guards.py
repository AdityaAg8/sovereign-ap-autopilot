"""Tests for the Extractor B pre-run data guards (ADR-0011 §5): one planted row per
quarantine reason, the leak stop, the per-source rate stop (file still written), dev winning a
near duplicate and stable output order and hashes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml.extractor_b.checkpoints import sha256_file
from ml.extractor_b.guards import (
    compute_page1_phashes,
    guard_outputs_current,
    label_sanity,
    parse_raw_rows,
    raw_stamp,
    run_guards,
    write_guard_outputs,
)
from ml.extractor_b.interfaces import (
    DataLeakError,
    GuardConfig,
    QuarantineRateError,
    QuarantineReason,
    SampleSource,
    TrainingError,
    TrainingSample,
)
from tests.unit.ml.conftest import RowFactory, good_target, sha

CFG = GuardConfig()
# Rate cap off for the per-reason tests, which plant one bad row in a small set.
LOOSE = GuardConfig(max_quarantine_rate=1.0)


def _phash(rows: list[TrainingSample], tmp_path: Path) -> dict[str, int]:
    return {r.sample_id: n * 0xFFFF for n, r in enumerate(rows)}


def _run(
    rows: list[TrainingSample],
    tmp_path: Path,
    *,
    cfg: GuardConfig = LOOSE,
    phash: dict[str, int] | None = None,
    **kw: object,
) -> tuple[list[TrainingSample], dict[str, QuarantineReason]]:
    kept, quarantined, _ = run_guards(
        rows,
        train_split_sha256=frozenset(r.document_sha256 for r in rows),
        page1_phash=phash if phash is not None else _phash(rows, tmp_path),
        cfg=cfg,
        max_seq_len=4096,
        image_root=tmp_path,
        **kw,  # type: ignore[arg-type]
    )
    return kept, {q.sample_id: q.reason for q in quarantined}


# ---- label sanity ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({}, None),
        (
            {
                "supplier_name": None,
                "supplier_trn": None,
                "currency": None,
                "subtotal": None,
                "vat_total": None,
                "total": None,
                "issue_date": None,
            },
            QuarantineReason.LABEL_EMPTY,
        ),
        ({"total": "one hundred"}, QuarantineReason.LABEL_FORMAT),
        ({"issue_date": "31/02/2026"}, QuarantineReason.LABEL_FORMAT),
        ({"supplier_trn": "12345"}, QuarantineReason.LABEL_FORMAT),
        ({"buyer_trn": "ABC"}, QuarantineReason.LABEL_FORMAT),
        ({"currency": "DOLLARS"}, QuarantineReason.LABEL_FORMAT),
        ({"total": "110.00"}, QuarantineReason.LABEL_ARITHMETIC),
        ({"total": "105.02"}, None),  # within 0.02
        ({"subtotal": "10000", "vat_total": "500", "total": "10540"}, None),  # within 0.5%
        (
            {"subtotal": "10000", "vat_total": "500", "total": "10600"},
            QuarantineReason.LABEL_ARITHMETIC,
        ),
        ({"currency": "usd"}, None),
        ({"currency": "Dhs"}, None),
        ({"total": None}, None),  # arithmetic only when all three are present
        ({"issue_date": "05/01/2026", "supplier_trn": "100-1234-5670-0003"}, None),
    ],
)
def test_label_sanity(overrides: dict[str, str | None], reason: QuarantineReason | None) -> None:
    assert label_sanity(json.dumps(good_target(**overrides)), CFG) is reason


def test_label_sanity_rejects_non_object() -> None:
    assert label_sanity("[1, 2]", CFG) is QuarantineReason.LABEL_FORMAT
    assert label_sanity("{bad", CFG) is QuarantineReason.LABEL_FORMAT


# ---- one planted row per reason ---------------------------------------------------------------


def test_one_planted_row_per_reason(make_row: RowFactory, tmp_path: Path) -> None:
    bad_schema = good_target()
    bad_schema["surprise"] = "x"
    rows = [
        make_row("a-ok"),
        make_row("b-schema", target=bad_schema),
        make_row("c-unreadable", image_bytes=b"not an image"),
        make_row("d-small", size=(100, 400)),
        make_row("e-pages", n_pages=3),
        make_row(
            "f-empty",
            target=good_target(
                supplier_name=None,
                supplier_trn=None,
                currency=None,
                subtotal=None,
                vat_total=None,
                total=None,
                issue_date=None,
            ),
        ),
        make_row("g-arith", target=good_target(invoice_number="g", total="999.00")),
        make_row("h-format", target=good_target(invoice_number="h", issue_date="not a date")),
        make_row("i-dup-exact", doc="a-ok"),
        make_row("j-long", tokens=(4000, 200)),
        make_row("k-near", target=good_target(invoice_number="a-ok")),
        make_row("l-dev", role="dev", target=good_target(invoice_number="dev-twin")),
        make_row("m-train-twin", target=good_target(invoice_number="DEV twin")),
    ]
    phash = _phash(rows, tmp_path)
    phash["k-near"] = phash["a-ok"] ^ 0b111  # distance 3
    phash["m-train-twin"] = phash["l-dev"] ^ 0b1  # distance 1, normalised target equal
    small = GuardConfig(max_quarantine_rate=1.0, max_pages=2)
    kept, reasons = _run(rows, tmp_path, cfg=small, phash=phash)
    assert reasons == {
        "b-schema": QuarantineReason.SCHEMA_INVALID,
        "c-unreadable": QuarantineReason.IMAGE_UNREADABLE,
        "d-small": QuarantineReason.IMAGE_OUT_OF_BOUNDS,
        "e-pages": QuarantineReason.IMAGE_OUT_OF_BOUNDS,
        "f-empty": QuarantineReason.LABEL_EMPTY,
        "g-arith": QuarantineReason.LABEL_ARITHMETIC,
        "h-format": QuarantineReason.LABEL_FORMAT,
        "i-dup-exact": QuarantineReason.DUPLICATE_EXACT,
        "j-long": QuarantineReason.TOO_LONG,
        "k-near": QuarantineReason.DUPLICATE_NEAR,
        "m-train-twin": QuarantineReason.DEV_NEAR_DUPLICATE,
    }
    assert [r.sample_id for r in kept] == ["a-ok", "l-dev"]
    assert set(QuarantineReason) == set(reasons.values())


def test_near_duplicate_needs_close_hash(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [make_row("a"), make_row("b", target=good_target(invoice_number="a"))]
    far = {"a": 0, "b": 0b11111}  # distance 5 > 4
    kept, reasons = _run(rows, tmp_path, phash=far)
    assert reasons == {} and len(kept) == 2


def test_dev_wins_even_with_a_later_sample_id(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [make_row("a-train", doc="same"), make_row("z-dev", role="dev", doc="same")]
    kept, reasons = _run(rows, tmp_path)
    assert [r.sample_id for r in kept] == ["z-dev"]
    assert reasons == {"a-train": QuarantineReason.DUPLICATE_EXACT}


def test_missing_phash_for_kept_row_raises(make_row: RowFactory, tmp_path: Path) -> None:
    with pytest.raises(TrainingError):
        _run([make_row("a")], tmp_path, phash={})


# ---- leak and rate --------------------------------------------------------------------------


def test_leak_raises_and_is_never_quarantined(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [make_row("a"), make_row("b")]
    with pytest.raises(DataLeakError):
        run_guards(
            rows,
            train_split_sha256=frozenset({rows[0].document_sha256}),
            page1_phash=_phash(rows, tmp_path),
            cfg=CFG,
            max_seq_len=4096,
            image_root=tmp_path,
        )


def test_rate_above_five_percent_raises_after_writing(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [make_row(f"u{n:02d}") for n in range(47)]
    rows += [make_row(f"x{n}", tokens=(5000, 10)) for n in range(3)]  # 3/50 = 6%
    rows += [make_row(f"f{n}", source=SampleSource.FATURA) for n in range(5)]
    out = tmp_path / "q" / "quarantine.jsonl"
    with pytest.raises(QuarantineRateError, match="uae 3/50"):
        _run(rows, tmp_path, cfg=CFG, quarantine_path=out)
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [w["sample_id"] for w in written] == ["x0", "x1", "x2"]


def test_rate_at_five_percent_passes(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [make_row(f"u{n:02d}") for n in range(19)] + [make_row("x", tokens=(5000, 10))]
    kept, reasons = _run(rows, tmp_path, cfg=CFG)  # 1/20 = 5%, not above
    assert len(kept) == 19 and reasons == {"x": QuarantineReason.TOO_LONG}


# ---- raw parsing, outputs and order ---------------------------------------------------------


def test_parse_raw_rows_quarantines_invalid_rows(make_row: RowFactory) -> None:
    good = make_row("a")
    bad = {"sample_id": "b", "document_sha256": sha("b"), "source": "docile", "role": "train"}
    rows, invalid = parse_raw_rows([good.model_dump_json(), "", json.dumps(bad)])
    assert rows == [good]
    assert [(q.sample_id, q.reason) for q in invalid] == [("b", QuarantineReason.SCHEMA_INVALID)]
    with pytest.raises(TrainingError):
        parse_raw_rows(['{"sample_id": "c"}'])


def test_pre_quarantined_rows_count_and_are_leak_checked(
    make_row: RowFactory, tmp_path: Path
) -> None:
    rows = [make_row("a")]
    _, invalid = parse_raw_rows(
        [json.dumps({"sample_id": "b", "document_sha256": sha("elsewhere"), "source": "uae"})]
    )
    with pytest.raises(DataLeakError):
        _run(rows, tmp_path, pre_quarantined=invalid)


def test_outputs_hash_and_currency(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [
        make_row("b"),
        make_row("a"),
        make_row("c", role="dev"),
        make_row("d", tokens=(9000, 1)),
    ]
    kept, quarantined, report = run_guards(
        rows,
        train_split_sha256=frozenset(r.document_sha256 for r in rows),
        page1_phash=compute_page1_phashes(rows, tmp_path),
        cfg=LOOSE,
        max_seq_len=4096,
        image_root=tmp_path,
    )
    out = tmp_path / "out"
    raw = tmp_path / "raw.jsonl"
    raw.write_text("x", encoding="utf-8")
    assert not guard_outputs_current(out, [raw])
    write_guard_outputs(out, kept, quarantined, report)
    (out / "raw.sha256").write_text(raw_stamp([raw]), encoding="utf-8")
    assert guard_outputs_current(out, [raw])
    raw.write_text("y", encoding="utf-8")
    assert not guard_outputs_current(out, [raw])
    assert sha256_file(out / "manifest.jsonl") == report.manifest_sha256
    assert sha256_file(out / "quarantine.jsonl") == report.quarantine_sha256
    assert [
        json.loads(line)["sample_id"] for line in (out / "train.jsonl").read_text().splitlines()
    ] == ["a", "b"]
    assert report.by_source_reason == {"uae:too_long": 1}
    first = json.loads((out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["image_paths"] == ["pages/a-0.png"]
    with pytest.raises(TrainingError):
        write_guard_outputs(out, kept[:1], quarantined, report)


def test_compute_page1_phashes_skips_unreadable(make_row: RowFactory, tmp_path: Path) -> None:
    rows = [make_row("a"), make_row("b", image_bytes=b"junk")]
    hashes = compute_page1_phashes(rows, tmp_path)
    assert set(hashes) == {"a"} and 0 <= hashes["a"] < 2**64


@settings(max_examples=25, deadline=None)
@given(order=st.permutations(list(range(6))))
def test_output_is_independent_of_input_order(
    tmp_path_factory: pytest.TempPathFactory, order: list[int]
) -> None:
    from PIL import Image

    root = tmp_path_factory.mktemp("perm")
    (root / "pages").mkdir()
    rows = []
    for n in range(6):
        rel = Path("pages") / f"{n}.png"
        Image.new("RGB", (300, 300), (n * 30, 0, 0)).save(root / rel)
        rows.append(
            TrainingSample(
                sample_id=f"s{n}",
                document_sha256=sha("dup" if n in (1, 4) else f"d{n}"),
                source=SampleSource.UAE,
                role="dev" if n == 5 else "train",
                image_paths=(rel,),
                target_json=json.dumps(good_target(invoice_number=f"i{n}")),
                n_prompt_tokens=10,
                n_target_tokens=10,
            )
        )
    shuffled = [rows[i] for i in order]
    shas = frozenset(r.document_sha256 for r in rows)
    phash = {r.sample_id: i << 8 for i, r in enumerate(rows)}
    a = run_guards(
        rows, train_split_sha256=shas, page1_phash=phash, cfg=LOOSE, max_seq_len=99, image_root=root
    )
    b = run_guards(
        shuffled,
        train_split_sha256=shas,
        page1_phash=phash,
        cfg=LOOSE,
        max_seq_len=99,
        image_root=root,
    )
    assert a == b
