"""Unit tests for evals.real.run (ADR-0019 §4-§6): the dirty-tree refusal, the resumable loop's
failure accounting (rows, retries, three server failures in a row), report building with failures
in every denominator, identity and document-set checks, before/after and report_all. The row
function is a CPU fake; no model, parse service, GPU or network is used."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evals.real import run as real_run
from evals.real.models import (
    ConsecutiveFailuresError,
    DirtyTreeError,
    MixedIdentityError,
    RealEvalReport,
    RealRow,
    RealTrackError,
)
from evals.real.run import (
    IncompleteRunError,
    RealDoc,
    RowResult,
    build_report,
    compare_arms,
    distribution,
    first_line,
    load_rows,
    render_report,
    report_all,
    require_clean,
    rows_path,
    run_docs,
)

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
STAMP = {
    "model": "extractor-a",
    "gguf_sha256": "a" * 64,
    "prompt_version": "p1",
    "serving_args_sha256": "s1",
    "parser_identity": "parser1",
    "git_sha": "abc123",
}


def _gt(i: int) -> dict[str, Any]:
    return {
        "menu": [{"nm": f"ITEM {i}", "cnt": "1", "price": f"{i},000"}],
        "total": {"total_price": f"{i},000"},
    }


def _answer(i: int) -> dict[str, Any]:
    return {
        "total": f"{i}.000",
        "line_items": [{"description": f"ITEM {i}", "quantity": "1", "net_amount": f"{i}.000"}],
    }


@pytest.fixture
def cord_root(tmp_path: Path) -> Path:
    """A 4-receipt CORD test split under tmp_path/data."""
    folder = tmp_path / "data" / "cord" / "test"
    folder.mkdir(parents=True)
    lines = [
        json.dumps(
            {"row": i, "sha256": f"{i:064x}", "image": f"{i:064x}.png",
             "ground_truth": {"gt_parse": _gt(i + 1)}}
        )
        for i in range(4)
    ]  # fmt: skip
    (folder / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "data"


def _docs(root: Path) -> list[RealDoc]:
    return real_run.set_docs("cord", "test", root)


def _run(docs: list[RealDoc], rows: Path, results: dict[str, RowResult], **kwargs: Any) -> int:
    calls: list[str] = kwargs.pop("calls", [])

    async def row_fn(doc: RealDoc) -> RowResult:
        calls.append(doc.sha256)
        return results[doc.sha256]

    return asyncio.run(
        run_docs(
            docs,
            set_="cord",
            split="test",
            extractor="a",
            model_version="base",
            stamp=kwargs.pop("stamp", STAMP),
            rows=rows,
            row_fn=row_fn,
            log=lambda _: None,
            **kwargs,
        )
    )


def _ok(i: int) -> RowResult:
    return RowResult(_answer(i), None, 1.0 + i)


# --- dirty tree ---------------------------------------------------------------------------------


def test_protected_split_refuses_a_dirty_tree() -> None:
    with pytest.raises(DirtyTreeError, match="protected split"):
        require_clean("cord", "test", dirty=lambda: ["evals/real/cord.py"])
    with pytest.raises(DirtyTreeError):
        require_clean("extractbench", "eval", dirty=lambda: ["apx/extract/prompt.py"])


def test_dev_split_and_clean_tree_run() -> None:
    require_clean("cord", "validation", dirty=lambda: ["evals/real/cord.py"])
    require_clean("cord", "test", dirty=lambda: [])


# --- the loop -----------------------------------------------------------------------------------


def test_failures_are_rows_and_counted(cord_root: Path, tmp_path: Path) -> None:
    docs = _docs(cord_root)
    rows = tmp_path / "rows.jsonl"
    results = {d.sha256: _ok(i + 1) for i, d in enumerate(docs)}
    results[docs[1].sha256] = RowResult(None, "invalid_json", 2.0, "finish_reason=length")
    results[docs[2].sha256] = RowResult({"total": "9"}, "parse_error", None, "ocr down")

    failed = _run(docs, rows, results)

    loaded = load_rows(rows)
    assert failed == 2
    assert len(loaded) == 4
    assert loaded[docs[1].sha256].error_kind == "invalid_json"
    assert loaded[docs[2].sha256].prediction is None  # a failed row never keeps an answer
    failures = real_run.failures_path(rows).read_text(encoding="utf-8").splitlines()
    assert [json.loads(f)["error_kind"] for f in failures] == ["invalid_json", "parse_error"]


def test_resume_skips_done_and_retry_failed_reruns_every_failed_row(
    cord_root: Path, tmp_path: Path
) -> None:
    docs = _docs(cord_root)
    rows = tmp_path / "rows.jsonl"
    results = {d.sha256: RowResult(None, "invalid_json", 1.0) for d in docs}
    results[docs[0].sha256] = _ok(1)
    _run(docs, rows, results)

    calls: list[str] = []
    assert _run(docs, rows, results, calls=calls) == 3
    assert calls == []  # everything has a row: resume runs nothing

    fixed = {d.sha256: _ok(i + 1) for i, d in enumerate(docs)}
    assert _run(docs, rows, fixed, calls=calls, retry_failed=True) == 0
    assert calls == [d.sha256 for d in docs[1:]]  # all failed rows, never a subset
    assert all(r.error_kind is None for r in load_rows(rows).values())


def test_three_server_failures_in_a_row_stop_the_run(cord_root: Path, tmp_path: Path) -> None:
    docs = _docs(cord_root)
    rows = tmp_path / "rows.jsonl"
    unloads: list[int] = []

    async def unload() -> None:
        unloads.append(1)

    results = {d.sha256: RowResult(None, "timeout", None, "300 s") for d in docs}
    with pytest.raises(ConsecutiveFailuresError, match="3 documents in a row"):
        _run(docs, rows, results, on_server_failure=unload)
    assert len(load_rows(rows)) == 3  # the three failures are recorded rows
    assert len(unloads) == 3


def test_document_failures_do_not_count_as_server_failures(cord_root: Path, tmp_path: Path) -> None:
    docs = _docs(cord_root)
    kinds = ["timeout", "server_error", "rejected", "timeout"]
    results = {d.sha256: RowResult(None, k, None) for d, k in zip(docs, kinds, strict=True)}  # type: ignore[arg-type]
    assert _run(docs, tmp_path / "rows.jsonl", results) == 4


def test_resume_refuses_rows_from_another_identity(cord_root: Path, tmp_path: Path) -> None:
    docs = _docs(cord_root)
    rows = tmp_path / "rows.jsonl"
    results = {d.sha256: _ok(i + 1) for i, d in enumerate(docs)}
    _run(docs[:2], rows, results)
    with pytest.raises(MixedIdentityError, match="another run"):
        _run(docs, rows, results, stamp={**STAMP, "gguf_sha256": "b" * 64})
    # A new commit alone is not a new identity: git_sha is listed, not compared.
    _run(docs, rows, results, stamp={**STAMP, "git_sha": "def456"})


# --- reports ------------------------------------------------------------------------------------


def _complete(cord_root: Path, rows: Path, results: dict[str, RowResult] | None = None) -> None:
    docs = _docs(cord_root)
    _run(docs, rows, results or {d.sha256: _ok(i + 1) for i, d in enumerate(docs)})


def test_report_of_perfect_rows(cord_root: Path, tmp_path: Path) -> None:
    rows = tmp_path / "cord.test.a-base.jsonl"
    _complete(cord_root, rows)
    report = build_report(rows, data_root=cord_root, now=NOW)
    assert report.n_docs == 4
    assert report.metric("field_f1").value == 1.0  # type: ignore[union-attr]
    assert report.metric("nted_accuracy").value == 1.0  # type: ignore[union-attr]
    assert report.valid_json_rate.value == 1.0
    assert report.mapped_gold_share == 1.0
    assert report.distribution == "zero-shot"
    assert report.baselines and not any(b.comparable for b in report.baselines)
    assert report.latency_p50_s == pytest.approx(3.5)


def test_a_failed_row_lowers_f1_and_valid_json_instead_of_being_dropped(
    cord_root: Path, tmp_path: Path
) -> None:
    docs = _docs(cord_root)
    rows = tmp_path / "rows.jsonl"
    results = {d.sha256: _ok(i + 1) for i, d in enumerate(docs)}
    results[docs[3].sha256] = RowResult(None, "timeout", None)
    _complete(cord_root, rows, results)

    report = build_report(rows, data_root=cord_root, now=NOW)

    assert report.n_docs == 4
    assert report.n_failed == 1
    assert report.failures_by_kind == {"timeout": 1}
    assert report.valid_json_rate.value == 0.75
    f1 = report.metric("field_f1")
    assert f1 is not None and f1.n_docs == 4
    assert f1.value == pytest.approx(2 * 9 / (2 * 9 + 3))  # its 3 gold fields are misses


def test_report_refuses_an_incomplete_run(cord_root: Path, tmp_path: Path) -> None:
    docs = _docs(cord_root)
    rows = tmp_path / "rows.jsonl"
    _run(docs[:3], rows, {d.sha256: _ok(1) for d in docs})
    with pytest.raises(IncompleteRunError, match="1 of 4 documents"):
        build_report(rows, data_root=cord_root)


def test_report_refuses_mixed_identities(cord_root: Path, tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    _complete(cord_root, rows)
    first = next(iter(load_rows(rows).values()))
    foreign = first.model_copy(update={"stamp": {**STAMP, "prompt_version": "p2"}})
    with rows.open("a", encoding="utf-8") as sink:
        sink.write(foreign.model_dump_json() + "\n")
    with pytest.raises(MixedIdentityError, match="different runs"):
        build_report(rows, data_root=cord_root)


def test_distribution_labels() -> None:
    assert distribution("cord", "a", "base") == "zero-shot"
    assert distribution("cord", "b", "v1") == "zero-shot"
    assert distribution("cord", "b", "r1") == "in-distribution"
    assert distribution("docile", "b", "r1") == "zero-shot"
    with pytest.raises(RealTrackError):
        distribution("cord", "b", "v9")


def test_render_report_first_line_states_n_split_extractor_and_scope(
    cord_root: Path, tmp_path: Path
) -> None:
    rows = tmp_path / "rows.jsonl"
    _complete(cord_root, rows)
    report = build_report(rows, data_root=cord_root, now=NOW)
    markdown = render_report(report)
    assert markdown.splitlines()[2] == first_line(report)
    assert first_line(report).startswith("n = 4, cord test, Extractor A `base`")
    assert first_line(report).endswith("zero-shot")
    assert "Donut" in markdown and "not like-for-like" in markdown
    assert "ITEM 1" not in markdown  # aggregates only


def _arm(report: RealEvalReport, version: str, extractor: str = "b") -> RealEvalReport:
    return report.model_copy(update={"model_version": version, "extractor": extractor})


def test_before_after_pairs_identical_documents(cord_root: Path, tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    _complete(cord_root, rows)
    report = build_report(rows, data_root=cord_root, now=NOW)
    worse = report.model_copy(
        update={"per_doc_counts": {"field_f1": [(0, 0, 3)] * 4}, "model_version": "v1",
                "extractor": "b"}
    )  # fmt: skip
    comparison = compare_arms([worse, _arm(report, "r1")], now=NOW)
    (delta,) = comparison.deltas
    assert delta.name == "field_f1:b-r1 - b-v1"
    assert delta.value == 1.0
    assert delta.n_docs == 4


def test_before_after_refuses_different_documents(cord_root: Path, tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    _complete(cord_root, rows)
    report = build_report(rows, data_root=cord_root, now=NOW)
    other = _arm(report, "r1").model_copy(update={"doc_ids_sha256": "0" * 64})
    with pytest.raises(MixedIdentityError, match="differ"):
        compare_arms([_arm(report, "v1"), other])
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(_arm(report, "v1").model_dump_json(), encoding="utf-8")
    path_b.write_text(other.model_dump_json(), encoding="utf-8")
    with pytest.raises(MixedIdentityError):
        real_run.before_after([path_a, path_b])


def test_report_all_writes_reports_and_before_after(cord_root: Path, tmp_path: Path) -> None:
    rows_dir = tmp_path / "logs"
    reports = tmp_path / "reports"
    docs = _docs(cord_root)
    for version, extractor in (("base", "a"), ("v1", "b"), ("r1", "b")):
        path = rows_path("cord", "test", extractor, version, rows_dir)  # type: ignore[arg-type]
        asyncio.run(
            run_docs(
                docs, set_="cord", split="test", extractor=extractor,  # type: ignore[arg-type]
                model_version=version, stamp={**STAMP, "model": f"m-{version}"}, rows=path,
                row_fn=lambda d: _async(_ok(1)), log=lambda _: None,
            )
        )  # fmt: skip
    incomplete = rows_path("cord", "test", "b", "r2", rows_dir)
    asyncio.run(
        run_docs(
            docs[:1], set_="cord", split="test", extractor="b", model_version="r2", stamp=STAMP,
            rows=incomplete, row_fn=lambda d: _async(_ok(1)), log=lambda _: None,
        )
    )  # fmt: skip

    status = report_all(rows_dir, reports, data_root=cord_root, dev_report=tmp_path / "dev.md",
                        log=lambda _: None)  # fmt: skip

    assert status == 1  # r2 is incomplete: not reported, and the exit says so
    assert (reports / "real_cord.a-base.json").is_file()
    assert (reports / "real_cord.b-v1.md").is_file()
    assert not (reports / "real_cord.b-r2.json").exists()
    summary = (reports / "real_cord.md").read_text(encoding="utf-8")
    assert summary.index("Extractor B v1") < summary.index("Extractor B r1")
    before_after = json.loads((reports / "real_before_after.json").read_text(encoding="utf-8"))
    (comparison,) = before_after["comparisons"]
    assert [a["model_version"] for a in comparison["arms"]] == ["v1", "r1", "base"]
    assert len(comparison["deltas"]) == 2


async def _async(result: RowResult) -> RowResult:
    return result


def test_rows_path_and_row_schema(tmp_path: Path) -> None:
    assert rows_path("cord", "test", "b", "v1", tmp_path).name == "cord.test.b-v1.jsonl"
    with pytest.raises(ValueError):
        RealRow.model_validate(
            {"set": "cord", "split": "test", "doc_sha256": "x", "extractor": "a",
             "model_version": "base", "stamp": {}, "prediction": None,
             "error_kind": "crashed", "latency_s": None}
        )  # fmt: skip


def test_set_docs_refuses_cord_train(cord_root: Path) -> None:
    with pytest.raises(RealTrackError, match="never run"):
        real_run.set_docs("cord", "train", cord_root)
