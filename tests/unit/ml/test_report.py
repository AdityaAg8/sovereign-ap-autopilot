"""Unit tests for the Extractor B training report (training-loop.md step 13): evidence from a fake
run directory, MLflow over a fake fetch (and its stated gap when down), the loss summary, the
selection reason and the invalid-JSON failure kinds. CPU only, no network."""

from __future__ import annotations

import json
import math
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ml.extractor_b.checkpoints import FsCheckpointStore, atomic_write_json
from ml.extractor_b.evaluator import DEV_OUTPUTS_DIR, FinishReason
from ml.extractor_b.interfaces import (
    CheckpointMeta,
    EvalResult,
    GuardReport,
    LoopState,
    RunRecord,
    SpikeGuardState,
)
from ml.extractor_b.report import (
    InvalidKind,
    MlflowRun,
    MlflowUnavailableError,
    ReportError,
    classify_invalid,
    collect,
    fetch_mlflow_run,
    read_dev_outputs,
    render,
    selection_reason,
    summarize_loss,
)
from tests.unit.ml.conftest import make_fingerprint

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
EVALS = (
    EvalResult(step=50, kind="dev_loss", dev_loss=0.0065),
    EvalResult(step=128, kind="generative", dev_f1=84.23, valid_json_rate=0.730),
    EvalResult(step=255, kind="generative", dev_f1=96.99, valid_json_rate=0.949),
    EvalResult(step=383, kind="generative", dev_f1=97.17, valid_json_rate=0.955),
    EvalResult(step=500, kind="dev_loss", dev_loss=0.0039),
    EvalResult(step=510, kind="generative", dev_f1=97.17, valid_json_rate=0.955),
)


def _history(n: int = 10) -> list[dict[str, Any]]:
    return [
        {"step": s, "epoch": s / (n / 2), "loss": 1.0 / s, "grad_norm": 0.1 * s}
        for s in range(1, n + 1)
    ]


def _run(tmp_path: Path, *, guard_sha: str | None = None) -> tuple[Path, Path]:
    fp = make_fingerprint()
    run_dir, data_dir = tmp_path / "runs" / "b-1", tmp_path / "data"
    atomic_write_json(
        run_dir / "run.json",
        RunRecord(run_id="b-1", mlflow_run_id="mlf-9", fingerprint=fp, created_at=NOW),
    )
    store = FsCheckpointStore(run_dir, fingerprint=fp, keep_last=2)
    for step in (383, 500, 510):
        out = store.begin(step)
        state = LoopState(
            spike=SpikeGuardState(total_skipped_steps=0),
            best_step=383,
            resume_count=3,
            evals=tuple(e for e in EVALS if e.step <= step),
        )
        (out / "loop_state.json").write_text(state.model_dump_json(), encoding="utf-8")
        (out / "trainer_state.json").write_text(
            json.dumps({"log_history": _history()}), encoding="utf-8"
        )
        store.commit(
            step, CheckpointMeta(step=step, epoch=2.0, created_at=NOW, fingerprint=fp, files={})
        )
        if step == 383:
            store.set_best(383)
    data_dir.mkdir()
    guard = GuardReport(
        kept=10,
        quarantined=2,
        by_source_reason={"uae:schema_invalid": 1, "fatura:empty_target": 1},
        manifest_sha256=fp.manifest_sha256,
        quarantine_sha256=guard_sha or fp.quarantine_sha256,
    )
    (data_dir / "guard_report.json").write_text(guard.model_dump_json(), encoding="utf-8")
    (data_dir / "dev.jsonl").write_text("{}\n" * 178, encoding="utf-8")
    return run_dir, data_dir


def _mlflow(run_id: str) -> MlflowRun:
    assert run_id == "mlf-9"
    return MlflowRun(
        status="FINISHED",
        start_ms=1_758_659_100_000,
        end_ms=1_758_704_700_000,
        params={},
        tags={},
        metrics={"train_loss": 0.0042, "best_step": 383.0},
    )


def _down(run_id: str) -> MlflowRun:
    raise MlflowUnavailableError("MLflow run mlf-9 not readable at http://x: refused")


def _collect(run_dir: Path, data_dir: Path, mlflow: Any = _mlflow) -> Any:
    return collect(run_dir, data_dir, tie_epsilon=0.1, keep_last=2, mlflow=mlflow)


# --- evidence and rendering -----------------------------------------------------------------


def test_report_has_selection_curve_quarantine_resumes_and_loss(tmp_path: Path) -> None:
    ev = _collect(*_run(tmp_path))
    assert ev.best_step == 383 and ev.reselected_step == 383
    assert ev.state_step == 510 and ev.state.resume_count == 3
    text = render(ev, generated_at=NOW)
    assert "| Selected step (`best.json`) | **383** |" in text
    assert "steps 383, 510 tie within 0.1" in text and "earlier step wins" in text
    assert "| 383 (selected) | generative | 97.17 | 95.5% |" in text
    assert "| 510 | generative | 97.17 | 95.5% |" in text
    assert "| 500 | dev_loss |  |  | 0.00390 |" in text
    assert "| uae | schema_invalid | 1 |" in text and "| fatura | empty_target | 1 |" in text
    assert "| Resumes | 3 |" in text
    assert "| Logged steps | 10 |" in text
    assert "Status FINISHED" in text and "| train_loss | 0.0042 |" in text
    assert "about 8 of 178" in text  # 4.5% invalid JSON on 178 dev documents
    assert "No verified export yet" in text
    assert f"`{make_fingerprint().config_sha256}`" in text


def test_missing_dev_outputs_is_a_stated_gap(tmp_path: Path) -> None:
    ev = _collect(*_run(tmp_path))
    assert any("per-row dev generations were not persisted" in g for g in ev.gaps)
    assert "failure kinds (looping / truncated / malformed) are unknown" in render(
        ev, generated_at=NOW
    )


def test_mlflow_down_is_a_gap_not_a_failure(tmp_path: Path) -> None:
    ev = _collect(*_run(tmp_path), mlflow=_down)
    assert ev.mlflow is None
    text = render(ev, generated_at=NOW)
    assert "MLflow unavailable" in text and "refused" in text
    assert "Not available; see Gaps." in text


def test_stale_guard_report_is_flagged(tmp_path: Path) -> None:
    ev = _collect(*_run(tmp_path, guard_sha="9" * 64))
    assert any("another data prep" in g for g in ev.gaps)


def test_missing_best_json_is_an_error(tmp_path: Path) -> None:
    run_dir, data_dir = _run(tmp_path)
    (run_dir / "best.json").unlink()
    with pytest.raises(ReportError, match=r"best.json"):
        _collect(run_dir, data_dir)


def test_dev_outputs_are_classified_when_persisted(tmp_path: Path) -> None:
    run_dir, data_dir = _run(tmp_path)
    rows = [
        {"sample_id": "a", "text": '{"invoice_number": "1"}'},
        {"sample_id": "b", "text": '{"invoice_number": "1", "notes": "' + "abcdefgh" * 6},
        {"sample_id": "c", "text": '{"invoice_number": "INV-2026-4'},
        {"sample_id": "d", "text": '{"invoice_number": "1",, }'},
    ]
    out = run_dir / DEV_OUTPUTS_DIR / "step-000383.jsonl"
    out.parent.mkdir()
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ev = _collect(run_dir, data_dir)
    assert ev.dev_outputs is not None and ev.dev_outputs.rows == 4
    assert ev.dev_outputs.invalid == {
        InvalidKind.LOOPING: ["b"],
        InvalidKind.TRUNCATED: ["c"],
        InvalidKind.MALFORMED: ["d"],
    }
    assert "| looping | 1 | b |" in render(ev, generated_at=NOW)


def test_bad_dev_outputs_row_is_typed(tmp_path: Path) -> None:
    out = tmp_path / DEV_OUTPUTS_DIR / "step-000001.jsonl"
    out.parent.mkdir()
    out.write_text('{"sample_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ReportError, match=":1 is not"):
        read_dev_outputs(tmp_path, 1)
    assert read_dev_outputs(tmp_path, 2) is None


# --- pure pieces ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('{"invoice_number": "INV-1", "total": "10.00"}', None),
        ('  {"a": null}  ', None),
        ('{"invoice_number": "INV-1", "supplier_name": "Acme', InvalidKind.TRUNCATED),
        ('{"line_items": [{"a": "1"}, {"b": ', InvalidKind.TRUNCATED),
        ('{"x": "' + "Total AED " * 8, InvalidKind.LOOPING),
        ('{"a": 1} trailing', InvalidKind.MALFORMED),
        ("[1, 2]", InvalidKind.MALFORMED),
        ('{"a": 1]', InvalidKind.MALFORMED),
        ("", InvalidKind.MALFORMED),
    ],
)
def test_classify_invalid(text: str, kind: InvalidKind | None) -> None:
    assert classify_invalid(text) is kind


def test_escaped_quote_does_not_end_a_string() -> None:
    assert classify_invalid('{"a": "say \\"hi') is InvalidKind.TRUNCATED


def test_summarize_loss() -> None:
    history = [*_history(), {"step": 11, "loss": math.nan, "epoch": 2.0}, {"eval_loss": 0.5}]
    s = summarize_loss(history, last_n=3)
    assert s.steps == 11 and s.non_finite == 1
    assert s.first == 1.0 and s.minimum == 0.1 and s.minimum_step == 10
    assert s.mean_last_n == pytest.approx((1 / 8 + 1 / 9 + 1 / 10) / 3)
    assert set(s.epoch_means) == {1, 2}
    assert s.max_grad_norm == pytest.approx(1.0)
    with pytest.raises(ReportError):
        summarize_loss([{"eval_loss": 1.0}])
    with pytest.raises(ReportError, match="non-finite"):
        summarize_loss([{"step": 1, "loss": math.inf}])


def test_selection_reason_variants() -> None:
    assert selection_reason(EVALS[:4], 383, 0.1) == "highest dev F1 (97.17)"
    tied_with_loss = (*EVALS, EvalResult(step=510, kind="dev_loss", dev_loss=0.003))
    assert "lower dev loss wins" in selection_reason(tied_with_loss, 510, 0.1)
    assert selection_reason(EVALS[:1], 50, 0.1) == "no generative eval exists"


def test_fetch_mlflow_run_parses_the_rest_payload() -> None:
    seen: list[str] = []

    def fetch(url: str, timeout: float) -> bytes:
        seen.append(url)
        return json.dumps(
            {
                "run": {
                    "info": {"status": "FINISHED", "start_time": "1", "end_time": None},
                    "data": {
                        "metrics": [{"key": "train_loss", "value": 0.0042}],
                        "params": [{"key": "rung", "value": "L0"}],
                        "tags": [{"key": "mlflow.runName", "value": "b-1"}],
                    },
                }
            }
        ).encode()

    run = fetch_mlflow_run("http://m:5000/", "abc", fetch=fetch)
    assert seen == ["http://m:5000/api/2.0/mlflow/runs/get?run_id=abc"]
    assert run.status == "FINISHED" and run.start_ms == 1 and run.end_ms is None
    assert run.metrics == {"train_loss": 0.0042} and run.params == {"rung": "L0"}


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("refused"), b"not json", b'{"error_code": "RESOURCE_DOES_NOT_EXIST"}'],
)
def test_fetch_mlflow_run_failures_are_typed(failure: object) -> None:
    def fetch(url: str, timeout: float) -> bytes:
        if isinstance(failure, Exception):
            raise failure
        assert isinstance(failure, bytes)
        return failure

    with pytest.raises(MlflowUnavailableError):
        fetch_mlflow_run("http://m", "abc", fetch=fetch)


@pytest.mark.parametrize(
    ("text", "reason", "kind"),
    [
        ('{"a": "cut', FinishReason.MAX_NEW_TOKENS, InvalidKind.TRUNCATED),
        ('{"a": 1,, }', FinishReason.MAX_NEW_TOKENS, InvalidKind.TRUNCATED),
        ('{"a": "cut', FinishReason.EOS, InvalidKind.MALFORMED),
        ('{"x": "' + "Total AED " * 8, FinishReason.MAX_NEW_TOKENS, InvalidKind.LOOPING),
        ('{"x": "' + "Total AED " * 8, FinishReason.EOS, InvalidKind.LOOPING),
        ('{"a": "ok"}', FinishReason.MAX_NEW_TOKENS, None),
    ],
)
def test_recorded_finish_reason_decides_truncated_vs_malformed(
    text: str, reason: FinishReason, kind: InvalidKind | None
) -> None:
    assert classify_invalid(text, reason) is kind


def test_unknown_finish_reason_is_typed(tmp_path: Path) -> None:
    out = tmp_path / DEV_OUTPUTS_DIR / "step-000001.jsonl"
    out.parent.mkdir()
    out.write_text('{"sample_id": "a", "text": "{", "finish_reason": "stop"}\n', encoding="utf-8")
    with pytest.raises(ReportError, match=":1 is not"):
        read_dev_outputs(tmp_path, 1)
