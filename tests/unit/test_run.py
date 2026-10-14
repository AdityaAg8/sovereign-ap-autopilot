"""Unit tests for evals/run.py: the eval runner CLI, its per-set gold loaders, the deferred-not-
error handling for docile/uae, and the NO_COLOR-respecting terminal summary.
No network for fatura/extractbench/docile/uae (all data is injected via tmp_path fixtures); the
pint_ae case exercises the real pinned artifacts under data/pint_ae/, exactly as
tests/unit/test_pint_ae_validator.py does, so it is marked `artifacts`
(unit tests must never touch the network) and excluded from the default `pytest` selection. No
GPU."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evals.adapters.extractbench_adapter import framework_available
from evals.datasets.manifest import DatasetEntry, DatasetStatus, Manifest, ManifestFile
from evals.predictors import get_predictor
from evals.predictors.dummy import DummyPredictor
from evals.run import (
    RUNS_DIR,
    RunReport,
    _gold_extractbench_examples,
    _load_fatura_gold_fields,
    _score_docile_set,
    _score_extractbench_set,
    _score_fatura_set,
    _score_pint_ae_set,
    _score_uae_set,
    _use_color,
    gate_status,
    main,
    render_summary_table,
    run_eval,
    summarize_outcome,
    write_run_report,
)
from evals.scorers.extractbench import score_extractbench
from evals.splits import SplitsFile, empty_splits

MISSING_CLONE = (
    "data/external/ExtractBench missing: run "
    "`uv run python -m evals.datasets fetch --set extractbench`"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_EXTRACTBENCH_REPO = REPO_ROOT / "data" / "external" / "ExtractBench"
REAL_EXTRACTBENCH_JSONL = REPO_ROOT / "data" / "extractbench" / "short.jsonl"

SHA_FATURA_TEST = "a" * 64
SHA_EXTRACTBENCH_EVAL = "b" * 64


def _splits_with(*, fatura_test: list[str], extractbench_eval: list[str]) -> SplitsFile:
    base = empty_splits()
    sets = dict(base.sets)
    sets["fatura"] = {"train": [], "test": fatura_test}
    sets["extractbench"] = {"eval": extractbench_eval}
    return SplitsFile(generated_at=base.generated_at, sets=sets)


def _fatura_manifest(doc_id: str, sha256: str) -> Manifest:
    return Manifest(
        generated_at=empty_splits().generated_at,
        datasets={
            "fatura": DatasetEntry(
                status=DatasetStatus.PRESENT,
                source="https://zenodo.org/records/8261508",
                version="8261508",
                license="CC-BY-4.0",
                files=[
                    ManifestFile(
                        path=f"data/fatura/invoices_dataset_final/images/{doc_id}.jpg",
                        sha256=sha256,
                        bytes=1,
                        split="test",
                        doc_id=doc_id,
                    )
                ],
            )
        },
    )


def _write_fatura_fixture(data_root: Path, doc_id: str) -> None:
    annotations_dir = data_root / "fatura/invoices_dataset_final/Annotations/Original_Format"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "SELLER_NAME": {"bbox": [[0, 0], [1, 1]], "text": "Acme Corp"},
        "TOTAL": {"bbox": [[0, 0], [1, 1]], "text": "TOTAL : 441.14 EUR"},
        "TABLE": [[{"bbox": [[0, 0], [1, 1]]}]],
        "LOGO": {"bbox": [[0, 0], [1, 1]]},
    }
    (annotations_dir / f"{doc_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_fatura_gold_fields_keeps_only_text_entries(tmp_path: Path) -> None:
    _write_fatura_fixture(tmp_path, "doc1")
    fields = _load_fatura_gold_fields(tmp_path, "doc1")
    assert fields == {"SELLER_NAME": "Acme Corp", "TOTAL": "TOTAL : 441.14 EUR"}


def test_score_fatura_set_scores_dummy_predictor_end_to_end(tmp_path: Path) -> None:
    _write_fatura_fixture(tmp_path, "doc1")
    manifest = _fatura_manifest("doc1", SHA_FATURA_TEST)
    splits = _splits_with(fatura_test=[SHA_FATURA_TEST], extractbench_eval=[])

    outcome = _score_fatura_set(manifest, splits, tmp_path, DummyPredictor(), None)

    assert outcome.status == "scored"
    assert outcome.split == "test"
    assert outcome.n == 1
    field_f1 = outcome.metrics["field_f1"]
    assert field_f1["precision"] == 1.0  # nothing predicted -> no false positives
    assert field_f1["recall"] == 0.0  # dummy predicts nothing -> every gold field missed
    assert field_f1["false_negatives"] == 2


def test_score_fatura_set_deferred_when_not_in_manifest() -> None:
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    outcome = _score_fatura_set(manifest, empty_splits(), Path("."), DummyPredictor(), None)
    assert outcome.status == "deferred"
    assert "not present" in (outcome.reason or "")


def _extractbench_manifest(doc_id: str, sha256: str) -> Manifest:
    return Manifest(
        generated_at=empty_splits().generated_at,
        datasets={
            "extractbench": DatasetEntry(
                status=DatasetStatus.PRESENT,
                source="https://huggingface.co/datasets/llamaindex/ExtractBench",
                version="short",
                license="Apache-2.0",
                files=[
                    ManifestFile(
                        path=f"data/extractbench/docs/short/{doc_id}.pdf",
                        sha256=sha256,
                        bytes=1,
                        split="eval",
                        doc_id=doc_id,
                    )
                ],
            )
        },
    )


def _write_extractbench_fixture(data_root: Path, record_id: str) -> None:
    (data_root / "extractbench").mkdir(parents=True, exist_ok=True)
    record = {
        "id": record_id,
        "category": "short",
        "pdf": f"docs/{record_id}.pdf",
        "expected_output": json.dumps(
            {
                "vendor_name": "Acme Corp",
                "amount": 100.5,
                "final_k1": False,
                "not_found": None,
                "coded_entries": [{"code": "A", "amount": 1.0}],
            }
        ),
        "field_rules": json.dumps(
            {
                "vendor_name": {
                    "evidence": [{"page": 1, "quote": "Acme Corp Ltd.", "value": "Acme Corp"}]
                },
                "amount": {"evidence": [{"page": 1, "quote": None, "value": 100.5}]},
            }
        ),
    }
    (data_root / "extractbench" / "short.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


def test_score_extractbench_set_flattens_scalars_and_grounding(tmp_path: Path) -> None:
    doc_id = "00581-2011-p0050"
    _write_extractbench_fixture(tmp_path, f"short/{doc_id}")
    manifest = _extractbench_manifest(doc_id, SHA_EXTRACTBENCH_EVAL)
    splits = _splits_with(fatura_test=[], extractbench_eval=[SHA_EXTRACTBENCH_EVAL])

    outcome = _score_extractbench_set(manifest, splits, tmp_path, DummyPredictor(), None)

    assert outcome.status == "scored"
    assert outcome.n == 1
    value_f1 = outcome.metrics["value_f1"]
    # 4 gold values survive (not_found=None is skipped): vendor_name, amount, final_k1,
    # coded_entries (serialized). Dummy predicts nothing, so all 4 are false negatives.
    assert value_f1["false_negatives"] == 4
    assert value_f1["true_positives"] == 0
    assert outcome.metrics["used_official_framework"] is False


def test_score_extractbench_set_deferred_without_split_members(tmp_path: Path) -> None:
    manifest = _extractbench_manifest("doc1", SHA_EXTRACTBENCH_EVAL)
    outcome = _score_extractbench_set(manifest, empty_splits(), tmp_path, DummyPredictor(), None)
    assert outcome.status == "deferred"


def test_gold_extractbench_examples_carries_page_and_bbox_from_field_rules() -> None:
    """The gold loader must populate page/bbox from `field_rules` evidence, not just the free-text
    `grounding` quote -- otherwise every gold example reaches `score_extractbench` with
    page=None, which the official-framework adapter treats as "no citation" and grounding F1 is
    graded on nothing."""
    record = {
        "id": "short/doc1",
        "expected_output": json.dumps(
            {"vendor_name": "Acme Corp", "final_k1": False, "no_evidence_field": "x"}
        ),
        "field_rules": json.dumps(
            {
                "vendor_name": {
                    "evidence": [
                        {"page": 1, "bbox": [0.1, 0.2, 0.3, 0.4], "quote": "Acme Corp Ltd."}
                    ]
                },
                # A coarse/checkbox field: no OCR quote, but still a real citable location.
                "final_k1": {
                    "evidence": [{"page": 2, "bbox": [0.5, 0.6, 0.05, 0.02], "quote": None}]
                },
                # No field_rules entry at all -> ungrounded, not an error.
            }
        ),
    }

    examples = {e.field: e for e in _gold_extractbench_examples(record)}

    assert examples["vendor_name"].page == 1
    assert examples["vendor_name"].bbox == [0.1, 0.2, 0.3, 0.4]
    assert examples["vendor_name"].grounding == "Acme Corp Ltd."

    assert examples["final_k1"].page == 2
    assert examples["final_k1"].bbox == [0.5, 0.6, 0.05, 0.02]
    assert examples["final_k1"].grounding is None  # quote was null, but bbox still carried

    assert examples["no_evidence_field"].page is None
    assert examples["no_evidence_field"].bbox is None


@pytest.mark.parametrize(
    ("page", "bbox", "message"),
    [
        (None, [0.1, 0.2, 0.3, 0.4], "1-indexed int page"),
        (0, [0.1, 0.2, 0.3, 0.4], "1-indexed int page"),
        (True, [0.1, 0.2, 0.3, 0.4], "1-indexed int page"),
        (1, [0.1, 0.2, 0.3], "4 numbers"),
        (1, [0.1, 0.2, 0.3, "0.4"], "4 numbers"),
        (1, [0.1, 0.2, 0.3, False], "4 numbers"),
    ],
    ids=["no-page", "page-0", "bool-page", "3-coords", "string-coord", "bool-coord"],
)
def test_gold_extractbench_examples_reject_malformed_bbox_evidence(
    page: object, bbox: list[object], message: str
) -> None:
    """Malformed bbox evidence fails loudly: a missing page would silently drop the field from
    grounding (the adapter skips page=None) and a bad bbox would reach the official framework."""
    record = {
        "id": "short/doc1",
        "expected_output": {"vendor_name": "Acme Corp"},
        "field_rules": {"vendor_name": {"evidence": [{"page": page, "bbox": bbox}]}},
    }
    with pytest.raises(ValueError, match=message):
        _gold_extractbench_examples(record)


@pytest.mark.external
def test_gold_extractbench_examples_grounding_is_non_vacuous_via_official_framework() -> None:
    """Proves the fix end to end against the real official framework
    (not the -spec fallback, which scores grounding on free text and would never notice a missing
    bbox): gold loaded
    through `_gold_extractbench_examples` and used as its own prediction must score grounding F1
    1.0, and corrupting one predicted citation's page must lower it -- i.e. the metric is actually
    being computed from real citations, not vacuously passing because none were ever supplied."""
    if not framework_available(REAL_EXTRACTBENCH_REPO):
        pytest.fail(MISSING_CLONE)
    with REAL_EXTRACTBENCH_JSONL.open(encoding="utf-8") as handle:
        record = json.loads(handle.readline())

    gold = _gold_extractbench_examples(record)
    groundable = [e for e in gold if e.page is not None]
    assert groundable, "fixture row has no bbox evidence to prove the fix against"

    predicted = list(gold)  # gold-as-prediction: identical values and citations
    baseline = score_extractbench(gold, predicted, external_repo=REAL_EXTRACTBENCH_REPO)
    assert baseline.used_official_framework is True
    assert baseline.grounding_f1.f1 == 1.0
    assert baseline.grounding_f1.true_positives > 0  # non-vacuous: real citations were graded

    target = groundable[0]
    wrong_page_target = replace(target, page=target.page + 1)  # type: ignore[operator]
    predicted_wrong = [wrong_page_target if e is target else e for e in predicted]
    perturbed = score_extractbench(gold, predicted_wrong, external_repo=REAL_EXTRACTBENCH_REPO)

    assert perturbed.grounding_f1.f1 < baseline.grounding_f1.f1


def test_score_docile_set_deferred_with_blocked_reason() -> None:
    manifest = Manifest(
        generated_at=empty_splits().generated_at,
        datasets={
            "docile": DatasetEntry(
                status=DatasetStatus.DEFERRED,
                source="https://docile.rossum.ai",
                version="",
                license="",
                blocked="[dataset:docile] needs DOCILE_TOKEN",
            )
        },
    )
    outcome = _score_docile_set(manifest)
    assert outcome.status == "deferred"
    assert outcome.reason == "[dataset:docile] needs DOCILE_TOKEN"


def test_score_docile_set_deferred_when_missing_from_manifest() -> None:
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    outcome = _score_docile_set(manifest)
    assert outcome.status == "deferred"
    assert "DOCILE_TOKEN" in (outcome.reason or "")


def test_score_uae_set_deferred_until_m2() -> None:
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    outcome = _score_uae_set(manifest)
    assert outcome.status == "deferred"
    assert "M2" in (outcome.reason or "")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.artifacts
def test_score_pint_ae_set_validates_official_samples_clean() -> None:
    """Exercises the real pinned PINT AE artifacts, exactly like
    tests/unit/test_pint_ae_validator.py: fetch them first with
    `uv run python -m evals.pint_ae.fetch` (or let this run once with network available) so
    repeat runs are offline."""
    outcome = _score_pint_ae_set(Path("data"))
    assert outcome.status == "scored"
    assert outcome.split == "official_samples"
    assert outcome.n > 0
    assert outcome.metrics["schematron_pass_rate"] == 1.0
    assert outcome.metrics["failed"] == []
    assert outcome.gate is not None
    assert outcome.gate.passed is True


def test_run_eval_dispatches_every_requested_set_in_order(tmp_path: Path) -> None:
    _write_fatura_fixture(tmp_path, "doc1")
    manifest = _fatura_manifest("doc1", SHA_FATURA_TEST)
    splits = _splits_with(fatura_test=[SHA_FATURA_TEST], extractbench_eval=[])

    report = run_eval(
        predictor=DummyPredictor(),
        sets=["fatura", "docile", "uae"],
        manifest=manifest,
        splits=splits,
        data_root=tmp_path,
    )

    assert [r.set_name for r in report.results] == ["fatura", "docile", "uae"]
    assert report.results[0].status == "scored"
    assert report.results[1].status == "deferred"
    assert report.results[2].status == "deferred"
    assert report.predictor == "dummy"
    assert report.git_sha  # non-empty, "unknown" is an acceptable fallback


def test_write_run_report_round_trips(tmp_path: Path) -> None:
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    report = run_eval(
        predictor=DummyPredictor(),
        sets=["uae"],
        manifest=manifest,
        splits=empty_splits(),
        data_root=tmp_path,
    )

    path = write_run_report(report, tmp_path / "runs")

    assert path.exists()
    loaded = RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.run_id == report.run_id
    assert loaded.results[0].set_name == "uae"


def test_main_cli_writes_a_run_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    splits_path = tmp_path / "splits.json"
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    splits_path.write_text(
        json.dumps(json.loads(empty_splits().model_dump_json())), encoding="utf-8"
    )
    out_dir = tmp_path / "runs"

    exit_code = main(
        [
            "--predictor",
            "dummy",
            "--sets",
            "uae",
            "--manifest-path",
            str(manifest_path),
            "--splits-path",
            str(splits_path),
            "--data-root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--no-summary",
        ]
    )

    assert exit_code == 0
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1


def test_get_predictor_dummy_and_unknown() -> None:
    assert get_predictor("dummy").name == "dummy"
    with pytest.raises(KeyError):
        get_predictor("no-such-predictor")


def test_dummy_predictor_returns_empty_mapping() -> None:
    predictor = DummyPredictor()
    assert predictor.predict_fields("any-doc-id") == {}


def test_summarize_and_gate_status_for_deferred_outcome() -> None:
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    outcome = _score_uae_set(manifest)
    assert summarize_outcome(outcome).startswith("deferred:")
    assert gate_status(outcome) == "PENDING"


def test_render_summary_table_respects_explicit_color_flag(tmp_path: Path) -> None:
    manifest = Manifest(generated_at=empty_splits().generated_at, datasets={})
    report = run_eval(
        predictor=DummyPredictor(),
        sets=["uae"],
        manifest=manifest,
        splits=empty_splits(),
        data_root=tmp_path,
    )

    plain = render_summary_table(report, color=False)
    colored = render_summary_table(report, color=True)

    assert "\x1b[" not in plain
    assert "\x1b[" in colored
    assert "PENDING" in plain


def test_use_color_respects_no_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTtyStream:
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            return None

    import evals.run as run_module

    monkeypatch.setattr(run_module.sys, "stdout", _FakeTtyStream())
    monkeypatch.setenv("NO_COLOR", "")
    assert _use_color() is False
    monkeypatch.delenv("NO_COLOR")
    assert _use_color() is True


def test_runs_dir_constant_points_under_reports() -> None:
    assert RUNS_DIR.parts[-2:] == ("reports", "runs")
