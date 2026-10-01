"""Tests for the ADR-0019 §9.4 real hold-out check (`ml.extractor_b.gates.require_real_holdouts`):
only Extractor B reports measured on the challenger's own Q4_K_M count, CORD test is always
required and DocILE val once DocILE is fetched, an unreadable report refuses, and the values are
carried unjudged. Reports are read through a stand-in with the `RealEvalReport` field shape."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ml.extractor_b.gates import (
    REAL_BEFORE_AFTER,
    RealHoldoutReport,
    RealHoldoutsMissingError,
    require_real_holdouts,
)
from tests.unit.ml.conftest import sha

GGUF = sha("challenger-q4")


@dataclass(frozen=True)
class FakeRealReport:
    """The `RealHoldoutReport` fields of `evals.real.models.RealEvalReport` (ADR-0019 sketch)."""

    set: str
    split: str
    extractor: str
    model_version: str
    n_docs: int
    identity: dict[str, Any] = field(default_factory=dict)


def load_fake_reports(path: Path) -> list[RealHoldoutReport]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    return [FakeRealReport(**item) for item in items]


def write_real_report(
    reports_dir: Path,
    set_: str,
    split: str,
    *,
    gguf: str,
    extractor: str = "b",
    n_docs: int = 100,
    name: str | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / (name or f"real_{set_}.{split}.{extractor}.json")
    report = {
        "set": set_,
        "split": split,
        "extractor": extractor,
        "model_version": "r1",
        "n_docs": n_docs,
        "identity": {"row_stamp": {"model_sha256": gguf, "git_sha": "abc"}},
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _require(reports_dir: Path, *, docile: bool = False) -> Any:
    return require_real_holdouts(
        reports_dir, gguf_sha256=GGUF, docile_present=docile, load=load_fake_reports
    )


def test_cord_test_report_of_the_challenger_satisfies_the_check(tmp_path: Path) -> None:
    write_real_report(tmp_path, "cord", "test", gguf=GGUF)
    (held,) = _require(tmp_path)
    assert (held.set, held.split, held.n_docs, held.model_version) == ("cord", "test", 100, "r1")
    assert held.report_file == "real_cord.test.b.json"


def test_no_reports_at_all_refuses(tmp_path: Path) -> None:
    with pytest.raises(RealHoldoutsMissingError, match="cord/test"):
        _require(tmp_path)


@pytest.mark.parametrize(
    ("extractor", "gguf", "n_docs", "split"),
    [
        ("a", GGUF, 100, "test"),  # Extractor A's report is not the challenger's
        ("b", sha("b-v1"), 100, "test"),  # B v1's report is measured on another build
        ("b", GGUF, 0, "test"),  # an empty report measured nothing
        ("b", GGUF, 20, "validation"),  # the dev split is never a hold-out result
    ],
)
def test_reports_that_are_not_the_challengers_hold_out_do_not_count(
    tmp_path: Path, extractor: str, gguf: str, n_docs: int, split: str
) -> None:
    write_real_report(tmp_path, "cord", split, gguf=gguf, extractor=extractor, n_docs=n_docs)
    with pytest.raises(RealHoldoutsMissingError, match="cord/test"):
        _require(tmp_path)


def test_docile_val_is_required_only_when_docile_is_present(tmp_path: Path) -> None:
    write_real_report(tmp_path, "cord", "test", gguf=GGUF)
    assert len(_require(tmp_path, docile=False)) == 1
    with pytest.raises(RealHoldoutsMissingError, match="docile/val"):
        _require(tmp_path, docile=True)
    write_real_report(tmp_path, "docile", "val", gguf=GGUF)
    assert [(h.set, h.split) for h in _require(tmp_path, docile=True)] == [
        ("cord", "test"),
        ("docile", "val"),
    ]


def test_a_file_holding_several_reports_is_read_whole(tmp_path: Path) -> None:
    one = write_real_report(tmp_path, "cord", "test", gguf=sha("b-v1"))
    two = write_real_report(tmp_path, "cord", "test", gguf=GGUF, name="other.json")
    combined = [json.loads(one.read_text("utf-8")), json.loads(two.read_text("utf-8"))]
    one.write_text(json.dumps(combined), encoding="utf-8")
    two.unlink()
    assert len(_require(tmp_path)) == 1


def test_an_unreadable_report_refuses_instead_of_being_skipped(tmp_path: Path) -> None:
    write_real_report(tmp_path, "cord", "test", gguf=GGUF)
    (tmp_path / "real_docile.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RealHoldoutsMissingError, match="cannot read"):
        _require(tmp_path)


def test_the_before_after_report_is_not_read_as_a_hold_out(tmp_path: Path) -> None:
    write_real_report(tmp_path, "cord", "test", gguf=GGUF)
    (tmp_path / REAL_BEFORE_AFTER).write_text('{"arms": []}', encoding="utf-8")
    assert len(_require(tmp_path)) == 1
