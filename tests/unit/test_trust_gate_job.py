"""Unit tests for the M7.4/M7.5 cal job (`evals.trust_gate`) on synthetic, recorded-style eval
rows: it fits and persists a calibrator, activates it and clears the ADR-0015 invalidation only
for the served champions, scores UAE test with the persisted artifact, and reports a failed gate
(never a smaller denominator) when test rows are missing or come from another build."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest

from apx.extract.identity import ExtractorIdentity, ServedBuilds, ServedModelError
from apx.extract.interfaces import FieldName
from apx.trust.artifact import CalibratorArtifact, CalibratorStore
from apx.trust.gate import ArtifactTrustGate
from apx.trust.interfaces import FIELD_KINDS, FieldDecision, TrustDecision
from evals.extraction_a import ROW_SCHEMA
from evals.trust_gate import (
    AlphaOutcome,
    CalibrateResult,
    TrustExtractors,
    anova_icc,
    calibrate,
    evaluate_test,
    evidence,
    main,
    render_report,
    verify,
)
from evals.trust_rows import TrustRowsError, load_split_rows
from ml.extractor_b.promote import FileCalibratorInvalidator
from ml.promotion.retrain import ModelVersion
from tests.unit import _trust_fixtures as tf

A_ID = ExtractorIdentity(model="extractor-a", gguf_sha256="a" * 64, prompt_version="pa")
B_ID = ExtractorIdentity(model="extractor-b-challenger", gguf_sha256="b" * 64, prompt_version="pb")
HEADER = [f for f in FieldName if f in FIELD_KINDS]
N_CAL, N_TEST = 320, 160


def _cell(pred: str | None, logprob: float, support: float) -> dict[str, Any]:
    return {
        "pred": pred,
        "correct": None,
        "grounded": pred is not None,
        "iou": None,
        "min_logprob": 2 * logprob,
        "mean_logprob": logprob,
        "final": pred,
        "ocr_support": support,
    }


def _write_split(root: Path, shas: list[str], seed: int, *, b_id: ExtractorIdentity = B_ID) -> None:
    """A and B rows plus factory gold for `shas`. A is wrong on ~5% of fields; most of those have
    low logprobs and OCR support, a few look confident, and B usually repeats A's reading."""
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    a_rows, b_rows = [], []
    for sha in shas:
        a_fields, b_fields, labels = {}, {}, []
        for i, field in enumerate(HEADER):
            gold = f"TOKEN{sha[:6]}N{i}"
            labels.append({"field": field.value, "value": gold, "page": 0, "bbox": [0, 0, 1, 1]})
            wrong = rng.random() < 0.05
            confident = not wrong or rng.random() < 0.08
            a_value = f"WRONG{sha[:6]}N{i}" if wrong else gold
            logprob = rng.uniform(-0.05, 0.0) if confident else rng.uniform(-4.0, -1.0)
            support = rng.uniform(0.85, 1.0) if confident else rng.uniform(0.0, 0.5)
            b_value = a_value if rng.random() < (0.97 if not wrong else 0.7) else gold + "B"
            a_fields[field.value] = _cell(a_value, logprob, support)
            b_fields[field.value] = _cell(b_value, rng.uniform(-0.1, 0.0), support)
        common = {"sha": sha, "row_schema": ROW_SCHEMA, "valid_json": True, "arabic": False}
        a_rows.append({**common, "fields": a_fields, **A_ID.as_row_fields()})
        b_rows.append({**common, "fields": b_fields, **b_id.as_row_fields()})
        (root / f"{sha}.labels.json").write_text(json.dumps({"labels": labels}), "utf-8")
        (root / f"{sha}.truth.json").write_text(json.dumps({"context": {}}), "utf-8")
    for name, rows in (("a.jsonl", a_rows), ("b.jsonl", b_rows)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


CAL_SHAS = [f"{i:064x}" for i in range(N_CAL)]
TEST_SHAS = [f"{i + 10_000:064x}" for i in range(N_TEST)]


@pytest.fixture(scope="module")
def splits(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("trust-job")
    _write_split(root / "cal", CAL_SHAS, seed=1)
    _write_split(root / "test", TEST_SHAS, seed=2)
    return root


def _champions(a: ExtractorIdentity = A_ID, b: ExtractorIdentity = B_ID) -> Any:
    # The champion alias serves the same GGUF the challenger rows were produced with.
    return lambda: ServedBuilds(a, b.model_copy(update={"model": "extractor-b"}), None)


def _run(
    splits: Path,
    trust: Path,
    champions: Any = None,
    extractors: TrustExtractors = TrustExtractors.BOTH,
    **paths: Path,
) -> CalibrateResult:
    return calibrate(
        cal_a=paths.get("cal_a", splits / "cal" / "a.jsonl"),
        cal_b=paths.get("cal_b", splits / "cal" / "b.jsonl"),
        cal_dir=splits / "cal",
        cal_shas=CAL_SHAS,
        test_a=paths.get("test_a", splits / "test" / "a.jsonl"),
        test_b=paths.get("test_b", splits / "test" / "b.jsonl"),
        test_dir=splits / "test",
        test_shas=TEST_SHAS,
        store=CalibratorStore(trust),
        champions=champions or _champions(),
        now=tf.NOW,
        extractors=extractors,
    )


@pytest.fixture(scope="module")
def fitted(splits: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[CalibrateResult, Path]:
    trust = tmp_path_factory.mktemp("trust-store")
    FileCalibratorInvalidator(trust / "calibrator_invalidation.json").invalidate(
        ModelVersion("3"), "extractor_b champion 2 -> 3"
    )
    return _run(splits, trust), trust


def test_calibrate_fits_activates_and_passes_the_gate(
    fitted: tuple[CalibrateResult, Path],
) -> None:
    result, trust = fitted
    art = result.calibration.artifact
    assert result.activation.activated and result.activation.cleared_invalidation
    assert CalibratorStore(trust).invalidation() is None
    assert CalibratorStore(trust).load_active()[0] == art
    assert art.extractor_a == A_ID and art.extractor_b == B_ID
    assert art.n_cal_documents == N_CAL
    assert {t.alpha for t in art.thresholds} == {0.005, 0.01, 0.02}
    assert art.runtime_threshold() is not None
    assert result.test is not None
    outcome = result.test.outcomes[0.01]
    assert outcome.n_fields == N_TEST * len(HEADER)  # hard-rejected fields stay in the denominator
    assert outcome.accepted > 0 and result.gate_passed, result.gate_detail


def test_report_and_evidence_carry_the_gate(fitted: tuple[CalibrateResult, Path]) -> None:
    result, _ = fitted
    report = render_report(result)
    for heading in ("## Test gate", "## Reliability", "## STP and error curves", "ECE"):
        assert heading in report
    assert "xychart-beta" in report and "UAE test" in report
    data = evidence(result)
    assert data["gate"]["passed"] is True and data["activated"] is True
    assert data["artifact_version"] == result.calibration.artifact.version
    json.dumps(data)


def test_verify_passes_only_with_active_fresh_artifact(
    fitted: tuple[CalibrateResult, Path], tmp_path: Path
) -> None:
    result, trust = fitted
    report = tmp_path / "trust_gate.md"
    evidence_path = tmp_path / "trust_gate.json"
    report.write_text(render_report(result), "utf-8")
    evidence_path.write_text(json.dumps(evidence(result)), "utf-8")
    store = CalibratorStore(trust)
    checks = verify(evidence_path, report, store, _champions())
    assert all(ok for _, ok, _ in checks), checks

    def unknown() -> ServedBuilds:
        raise ServedModelError("no infra")

    assert not all(ok for _, ok, _ in verify(evidence_path, report, store, unknown))
    other_b = B_ID.model_copy(update={"gguf_sha256": "c" * 64})
    assert not all(ok for _, ok, _ in verify(evidence_path, report, store, _champions(b=other_b)))
    assert verify(tmp_path / "absent.json", report, store, _champions())[0][1] is False


def test_runtime_gate_uses_the_calibrated_artifact(fitted: tuple[CalibrateResult, Path]) -> None:
    result, trust = fitted

    class _Fallback:
        def decide(self, features: Any) -> Any:  # pragma: no cover - must not be used
            raise AssertionError("fallback used although the artifact is valid")

    gate = ArtifactTrustGate(
        CalibratorStore(trust),
        type("S", (), {"current": staticmethod(_champions())})(),
        _Fallback(),
    )
    features = {f: tf.field_features(True, random.Random(1)) for f in HEADER}
    decision = gate.decide(features)
    assert decision.refusal is None
    assert decision.calibrator_version == result.calibration.artifact.version


def test_calibration_is_deterministic(
    splits: Path, fitted: tuple[CalibrateResult, Path], tmp_path: Path
) -> None:
    again = _run(splits, tmp_path / "trust")
    assert again.calibration.artifact.version == fitted[0].calibration.artifact.version


def test_challenger_fit_is_saved_but_not_activated(splits: Path, tmp_path: Path) -> None:
    trust = tmp_path / "trust"
    FileCalibratorInvalidator(trust / "calibrator_invalidation.json").invalidate(
        ModelVersion("4"), "promoted"
    )
    other_b = B_ID.model_copy(update={"gguf_sha256": "c" * 64})
    result = _run(splits, trust, champions=_champions(b=other_b))
    assert not result.activation.activated and "not activated" in result.activation.note
    assert result.artifact_path.is_file()
    store = CalibratorStore(trust)
    assert store.invalidation() is not None
    assert not store.active_path.exists()


def test_unknown_champions_do_not_activate(splits: Path, tmp_path: Path) -> None:
    def unknown() -> ServedBuilds:
        raise ServedModelError("infra/llama-swap.yaml missing")

    result = _run(splits, tmp_path / "trust", champions=unknown)
    assert not result.activation.activated and "unknown" in result.activation.note


def _drop_last_row(src: Path, dst: Path) -> Path:
    lines = src.read_text("utf-8").splitlines(keepends=True)
    dst.write_text("".join(lines[:-1]), "utf-8")
    return dst


def test_missing_test_rows_fail_the_gate_without_shrinking(splits: Path, tmp_path: Path) -> None:
    partial = _drop_last_row(splits / "test" / "b.jsonl", tmp_path / "b.jsonl")
    result = _run(splits, tmp_path / "trust", test_b=partial)
    assert result.test is None and not result.gate_passed
    assert result.gate_detail.startswith("MISSING")
    assert not result.activation.activated and "test gate did not pass" in result.activation.note
    assert not CalibratorStore(tmp_path / "trust").active_path.exists()
    assert f"1 of {N_TEST}" in result.gate_detail
    assert "missing" in render_report(result)
    assert evidence(result)["test"] is None


def test_test_rows_of_another_build_fail_the_gate(splits: Path, tmp_path: Path) -> None:
    foreign = tmp_path / "test"
    _write_split(foreign, TEST_SHAS, seed=2, b_id=B_ID.model_copy(update={"prompt_version": "px"}))
    result = _run(splits, tmp_path / "trust", test_b=foreign / "b.jsonl")
    assert result.test is None and "build" in (result.test_error or "")


def test_incomplete_cal_rows_raise_before_anything_is_saved(splits: Path, tmp_path: Path) -> None:
    partial = _drop_last_row(splits / "cal" / "a.jsonl", tmp_path / "a.jsonl")
    with pytest.raises(TrustRowsError, match=f"1 of {N_CAL}"):
        _run(splits, tmp_path / "trust", cal_a=partial)
    assert not (tmp_path / "trust").exists()


def test_cli_fails_loudly_without_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "calibrate",
            "--cal-a-rows",
            str(tmp_path / "cal_a.jsonl"),
            "--cal-b-rows",
            str(tmp_path / "cal_b.jsonl"),
            "--trust-dir",
            str(tmp_path / "trust"),
            "--report",
            str(tmp_path / "report.md"),
            "--evidence",
            str(tmp_path / "evidence.json"),
        ]
    )
    assert code == 2 and "TrustRowsError" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()
    verify_code = main(
        [
            "verify",
            "--evidence",
            str(tmp_path / "evidence.json"),
            "--trust-dir",
            str(tmp_path / "trust"),
        ]
    )
    assert verify_code == 1


def test_anova_icc_and_clopper_pearson() -> None:
    clustered = [[False] * 5] + [[True] * 5 for _ in range(9)]
    spread = [[False, True, True, True, True] for _ in range(10)]
    high, low = anova_icc(clustered), anova_icc(spread)
    assert high is not None and low is not None and high > 0.9 and low < 0
    assert anova_icc([[True, True], [True]]) is None  # no errors: undefined
    assert anova_icc([[True]]) is None
    outcome = AlphaOutcome(
        0.01, 0.9, accepted=1000, wrong=5, n_fields=2000, docs_all_accepted=0, n_docs=10
    )
    ci = outcome.error_ci()
    assert ci is not None and ci[0] < 0.005 < ci[1] and outcome.stp == 0.5
    assert AlphaOutcome(0.01, None, 0, 0, 10, 0, 1).error_ci() is None


def test_failed_test_gate_keeps_the_artifact_inactive_and_the_marker_stale(
    splits: Path, tmp_path: Path
) -> None:
    """ADR-0009 §5.A: activation and clearing the ADR-0015 invalidation follow the test gate.
    Test rows whose A readings are all wrong push the accepted-set error far above 1%."""
    bad = tmp_path / "test"
    _write_split(bad, TEST_SHAS, seed=2)
    rows = [json.loads(line) for line in (bad / "a.jsonl").read_text("utf-8").splitlines()]
    for row in rows:
        for cell in row["fields"].values():
            cell["pred"] = cell["final"] = "WRONG" + str(cell["pred"])
    (bad / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    (bad / "b.jsonl").write_text(
        (bad / "b.jsonl").read_text("utf-8").replace('"pred": "TOKEN', '"pred": "WRONGTOKEN'),
        "utf-8",
    )
    trust = tmp_path / "trust"
    FileCalibratorInvalidator(trust / "calibrator_invalidation.json").invalidate(
        ModelVersion("5"), "promoted"
    )
    result = _run(splits, trust, test_a=bad / "a.jsonl", test_b=bad / "b.jsonl")
    assert result.test is not None and not result.gate_passed, result.gate_detail
    assert not result.activation.activated and "test gate did not pass" in result.activation.note
    store = CalibratorStore(trust)
    assert store.invalidation() is not None and not store.active_path.exists()
    assert result.artifact_path.is_file()  # saved for the record, never active


# --- champion-only mode (ADR-0018 §5) --------------------------------------------------------


def _run_a(splits: Path, trust: Path, champions: Any = None, **paths: Path) -> CalibrateResult:
    """Champion-only run whose B row paths do not exist: they must never be read."""
    nowhere = trust.parent / "no-such-b.jsonl"
    return _run(
        splits,
        trust,
        champions,
        **{"cal_b": nowhere, "test_b": nowhere, **paths},
        extractors=TrustExtractors.CHAMPION_ONLY,
    )


def test_missing_b_rows_without_the_explicit_mode_still_raise(splits: Path, tmp_path: Path) -> None:
    with pytest.raises(TrustRowsError, match="cal B"):
        _run(splits, tmp_path / "trust", cal_b=tmp_path / "no-such-b.jsonl")
    assert not (tmp_path / "trust").exists()


def test_champion_only_fits_and_gates_a_without_activating(
    splits: Path, fitted: tuple[CalibrateResult, Path], tmp_path: Path
) -> None:
    trust = tmp_path / "trust"
    shutil.copytree(fitted[1], trust)
    FileCalibratorInvalidator(trust / "calibrator_invalidation.json").invalidate(
        ModelVersion("6"), "promoted"
    )
    active_before = CalibratorStore(trust).active_path.read_bytes()
    result = _run_a(splits, trust)
    art = result.calibration.artifact
    two = fitted[0].calibration.artifact
    assert art.champion_only and art.extractor_b is None and art.extractor_a == A_ID
    assert set(art.rows_sha256) == {"a"} and art.version != two.version
    # Same LTT settings as the two-extractor calibrator: alphas, delta, k_min.
    assert [(t.alpha, t.delta) for t in art.thresholds] == [
        (t.alpha, t.delta) for t in two.thresholds
    ]
    assert art.ltt_min_accepted == two.ltt_min_accepted
    assert result.test is not None
    assert result.test.outcomes[0.01].n_fields == N_TEST * len(HEADER)
    assert result.gate_passed, result.gate_detail
    # Never activated, the active calibrator and the ADR-0015 marker are left alone.
    assert not result.activation.activated and result.activation.champion_a_served is True
    assert "champion-only" in result.activation.note
    store = CalibratorStore(trust)
    assert store.active_path.read_bytes() == active_before
    assert store.invalidation() is not None
    assert CalibratorArtifact.model_validate_json(result.artifact_path.read_bytes()) == art
    data = evidence(result)
    assert data["extractors"] == "a" and data["extractor_b"] is None
    assert data["champion_a_served"] is True and data["activated"] is False
    assert "champion-only" in render_report(result)


def test_champion_only_records_a_foreign_champion(splits: Path, tmp_path: Path) -> None:
    other_a = A_ID.model_copy(update={"gguf_sha256": "f" * 64})
    result = _run_a(splits, tmp_path / "trust", champions=_champions(a=other_a))
    assert result.activation.champion_a_served is False
    assert not result.activation.activated and "served" in result.activation.note


def test_evaluate_test_refuses_a_mode_mismatch(
    splits: Path, fitted: tuple[CalibrateResult, Path], tmp_path: Path
) -> None:
    a_rows = load_split_rows(splits / "test" / "a.jsonl", TEST_SHAS, "test A")
    b_rows = load_split_rows(splits / "test" / "b.jsonl", TEST_SHAS, "test B")
    two = fitted[0].calibration.artifact
    with pytest.raises(TrustRowsError, match="missing for a two-extractor"):
        evaluate_test(two, a_rows, None, splits / "test")
    alone = _run_a(splits, tmp_path / "trust").calibration.artifact
    with pytest.raises(TrustRowsError, match="given for a champion-only"):
        evaluate_test(alone, a_rows, b_rows, splits / "test")


def test_runtime_gate_refuses_a_champion_only_artifact(splits: Path, tmp_path: Path) -> None:
    """Even if activated by hand, the runtime never scores served B output with it: REVIEW."""
    trust = tmp_path / "trust"
    art = _run_a(splits, trust).calibration.artifact
    store = CalibratorStore(trust)
    store.activate(art)

    class _AcceptAll:
        def decide(self, features: Any) -> TrustDecision:
            return TrustDecision(
                fields=tuple(
                    FieldDecision(field=f, p_correct=1.0, accepted=True) for f in features
                ),
                all_accepted=True,
            )

    gate = ArtifactTrustGate(
        store, type("S", (), {"current": staticmethod(_champions())})(), _AcceptAll()
    )
    decision = gate.decide({f: tf.field_features(True, random.Random(1)) for f in HEADER})
    assert decision.refusal is not None and decision.refusal.startswith("calibrator_champion_only")
    assert not decision.all_accepted and not any(f.accepted for f in decision.fields)


def test_cli_extractors_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = [
        "calibrate",
        "--cal-a-rows",
        str(tmp_path / "cal_a.jsonl"),
        "--trust-dir",
        str(tmp_path / "trust"),
        "--report",
        str(tmp_path / "report.md"),
        "--evidence",
        str(tmp_path / "evidence.json"),
    ]
    with pytest.raises(SystemExit) as exc:
        main([*base, "--extractors", "b"])
    assert exc.value.code == 2
    capsys.readouterr()
    assert main([*base, "--extractors", "a"]) == 2
    err = capsys.readouterr().err
    assert "TrustRowsError" in err and "cal A" in err and "cal B" not in err
    assert TrustExtractors.parse(["a", "b"]) is TrustExtractors.BOTH
    assert TrustExtractors.parse(["b", "a", "a"]) is TrustExtractors.BOTH
    assert TrustExtractors.parse(["a"]) is TrustExtractors.CHAMPION_ONLY
