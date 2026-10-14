"""Unit tests for `factory.restore_scans`: on a tiny scan-free factory corpus, restoring the planned
scan variants writes decodable page-sized scan PNGs and flips `has_scan` without touching any PDF
or label byte, is deterministic and idempotent, and refuses (typed errors) a corpus that does not
match the plan or whose PDF the renderer cannot reproduce (ADR-0012 amendment)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from factory.build import generate_dataset
from factory.restore_scans import (
    SCAN_PAGE_SIZE,
    CorpusIntegrityError,
    PlanMismatchError,
    RenderMismatchError,
    count_scans,
    main,
    restore_scans,
)

PLAN: dict[str, Any] = dict(
    seed=777, n_total=4, n_train=2, n_cal=1, n_test=1, n_vendors=3, n_users=4, scale=1.0
)
RESTORE: dict[str, Any] = {**PLAN, "scan_fraction": 0.5}


@pytest.fixture(scope="module")
def base_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("uae_base")
    generate_dataset(out_dir=out, scan_fraction=0.0, workers=1, **PLAN)
    return out


@pytest.fixture
def corpus(base_corpus: Path, tmp_path: Path) -> Path:
    out = tmp_path / "uae"
    shutil.copytree(base_corpus, out)
    return out


def _digests(root: Path, pattern: str) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob(pattern))
        if p.is_file()
    }


def _truths(root: Path) -> dict[str, dict[str, Any]]:
    return {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(root.rglob("*.truth.json"))
    }


def test_restore_writes_scans_and_touches_nothing_else(corpus: Path) -> None:
    pdfs = _digests(corpus, "*.pdf")
    labels = _digests(corpus, "*.labels.json")
    before = _truths(corpus)
    assert not list(corpus.rglob("*.scan.png"))

    results = restore_scans(data_dir=corpus, workers=1, **RESTORE)

    assert len(results) == 2  # round(4 * 0.5)
    assert not any(r.skipped for r in results)
    assert _digests(corpus, "*.pdf") == pdfs
    assert _digests(corpus, "*.labels.json") == labels
    after = _truths(corpus)
    restored = {f"{r.sha256}.truth.json" for r in results}
    for name, truth in after.items():
        expected = {**before[name], "has_scan": name in restored}
        assert truth == expected
    counts = count_scans(corpus)  # re-checks decode + size of every scan PNG
    assert sum(c.n_scan_docs for c in counts) == 2
    assert sum(c.n_scan_pngs for c in counts) == sum(r.n_pages for r in results)
    assert SCAN_PAGE_SIZE == (793, 1122)
    assert not list(corpus.rglob("*.tmp"))


def test_restore_is_deterministic_and_idempotent(corpus: Path, tmp_path: Path) -> None:
    restore_scans(data_dir=corpus, workers=1, **RESTORE)
    first = _digests(corpus, "*")
    mtimes = {p: p.stat().st_mtime_ns for p in corpus.rglob("*.scan.png")}

    again = restore_scans(data_dir=corpus, workers=1, **RESTORE)
    assert all(r.skipped for r in again)
    assert _digests(corpus, "*") == first
    assert {p: p.stat().st_mtime_ns for p in corpus.rglob("*.scan.png")} == mtimes

    # An interrupted run (scan PNGs gone, truth already flipped) is redone to the same bytes.
    victim = sorted(corpus.rglob("*.page0.scan.png"))[0]
    victim.unlink()
    redo = restore_scans(data_dir=corpus, workers=2, **RESTORE)
    assert sum(not r.skipped for r in redo) == 1
    assert _digests(corpus, "*") == first


def test_refuses_a_plan_mismatch(corpus: Path) -> None:
    truth_path = sorted(corpus.rglob("*.truth.json"))[0]
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["tax_case"] = "zero_rated" if truth["tax_case"] != "zero_rated" else "standard"
    truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(PlanMismatchError):
        restore_scans(data_dir=corpus, workers=1, **RESTORE)


def test_refuses_a_different_seed(corpus: Path) -> None:
    with pytest.raises(PlanMismatchError):
        restore_scans(data_dir=corpus, workers=1, **{**RESTORE, "seed": 778})


def test_refuses_a_pdf_whose_bytes_do_not_match_its_name(corpus: Path) -> None:
    for pdf in corpus.rglob("*.pdf"):
        pdf.write_bytes(pdf.read_bytes() + b"\n")
    with pytest.raises(CorpusIntegrityError):
        restore_scans(data_dir=corpus, workers=1, **RESTORE)
    assert not list(corpus.rglob("*.scan.png"))


def test_refuses_a_pdf_the_renderer_does_not_reproduce(corpus: Path) -> None:
    # Re-key every document to a consistent but different PDF: files agree with each other,
    # yet re-rendering the planned HTML yields the original bytes, not these.
    for truth_path in list(corpus.rglob("*.truth.json")):
        old = truth_path.name.removesuffix(".truth.json")
        split_dir = truth_path.parent
        pdf_bytes = (split_dir / f"{old}.pdf").read_bytes() + b"%tampered\n"
        new = hashlib.sha256(pdf_bytes).hexdigest()
        (split_dir / f"{new}.pdf").write_bytes(pdf_bytes)
        (split_dir / f"{old}.pdf").unlink()
        (split_dir / f"{old}.labels.json").rename(split_dir / f"{new}.labels.json")
        text = truth_path.read_text(encoding="utf-8").replace(old, new)
        (split_dir / f"{new}.truth.json").write_text(text, encoding="utf-8", newline="\n")
        truth_path.unlink()
    with pytest.raises(RenderMismatchError):
        restore_scans(data_dir=corpus, workers=1, **RESTORE)
    assert not list(corpus.rglob("*.scan.png"))
    assert '\n  "has_scan": true,' not in "".join(
        p.read_text(encoding="utf-8") for p in corpus.rglob("*.truth.json")
    )


def test_cli_prints_counts(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--data-dir", str(corpus), "--seed", "777", "--n-total", "4", "--n-train", "2",
        "--n-cal", "1", "--n-test", "1", "--n-vendors", "3", "--n-users", "4",
        "--scale", "1.0", "--scan-fraction", "0.5", "--workers", "1",
    ]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "restored now 2" in out
    assert "scan docs" in out
