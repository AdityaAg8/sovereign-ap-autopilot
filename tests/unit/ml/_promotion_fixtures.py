"""Shared builders for the M10.2 promotion tests (ADR-0015): `ShadowEval`s whose F1 numbers come
from the real fatura and UAE scorers run on a small gold set and a predictor, with docile and uae
set outcomes produced by `evals.run` itself, so a degraded challenger is degraded by its
predictions rather than by a hand-typed number."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from evals.datasets.manifest import DatasetEntry, DatasetStatus, Manifest
from evals.run import (
    GateKind,
    GateResult,
    RunReport,
    SetName,
    SetOutcome,
    _score_docile_set,
    _score_uae_set,
)
from evals.scorers.docile import DocileTaskScore
from evals.scorers.fatura import score_fatura
from evals.scorers.uae import UaeDocument, score_uae
from ml.promotion.evidence import ExtractorGateReport, ShadowEval

Fields = Mapping[str, Mapping[str, str]]
Predictor = Callable[[Fields], dict[str, dict[str, str]]]

GIT_SHA = "a" * 40
SESSION = "shadow-2026-09-24"

FATURA_GOLD: dict[str, dict[str, str]] = {
    f"fatura{i:02d}": {"invoice_number": f"F-{i}", "total": f"{10 + i}.00"} for i in range(25)
}
EXTRACTBENCH_GOLD: dict[str, dict[str, str]] = {
    f"eb{i:02d}": {"vendor": f"Vendor {i}", "amount": f"{i}.50"} for i in range(25)
}
UAE_GOLD: list[UaeDocument] = [
    UaeDocument(
        doc_id=f"uae{i:02d}",
        lang="en" if i % 2 == 0 else "ar",
        fields={
            "invoice_number": f"INV-{i}",
            "invoice_date": f"2026-09-{1 + i:02d}",
            "seller_trn": f"100{i:012d}",
            "buyer_trn": f"200{i:012d}",
            "total_amount": f"{100 + i}.00",
        },
    )
    for i in range(20)
]
"""100 gold fields: each wrong field costs one point of pooled F1."""


def perfect(gold: Fields) -> dict[str, dict[str, str]]:
    return {doc: dict(fields) for doc, fields in gold.items()}


def corrupt(n_fields: int) -> Predictor:
    """Gets one field wrong in each of the first `n_fields` documents."""

    def predict(gold: Fields) -> dict[str, dict[str, str]]:
        out = perfect(gold)
        for doc in sorted(out)[:n_fields]:
            first = sorted(out[doc])[0]
            out[doc][first] = "WRONG"
        return out

    return predict


def manifest(*, docile_present: bool = False) -> Manifest:
    datasets = {}
    if docile_present:
        datasets["docile"] = DatasetEntry(
            status=DatasetStatus.PRESENT, source="docile", version="1", license="research"
        )
    return Manifest(generated_at=datetime(2026, 9, 24, tzinfo=UTC), datasets=datasets)


def f1_block(gold: Fields, predictor: Predictor) -> dict[str, Any]:
    return dict(asdict(score_fatura(gold, predictor(gold))))


def docile_scored(kile_f1: float, lir_f1: float) -> SetOutcome:
    """The shape a wired-in docile set will carry: `DocileTaskScore` blocks for KILE and LIR."""

    def block(f1: float) -> dict[str, Any]:
        return asdict(DocileTaskScore(f1, f1, f1, f1, 90, 10, 10))

    return SetOutcome(
        set_name="docile",
        status="scored",
        split="val",
        n=50,
        metrics={"kile": block(kile_f1), "lir": block(lir_f1)},
        gate=GateResult(kind=GateKind.REPORT),
    )


def pint_ae(pass_rate: float = 1.0) -> SetOutcome:
    return SetOutcome(
        set_name="pint_ae",
        status="scored",
        split="official_samples",
        n=4,
        metrics={"schematron_pass_rate": pass_rate, "failed": []},
        gate=GateResult(
            kind=GateKind.MIN, threshold=1.0, measured=pass_rate, passed=pass_rate >= 1.0
        ),
    )


def run_report(
    predictor: Predictor,
    *,
    version: str,
    per_set: Mapping[SetName, Predictor] | None = None,
    overrides: Mapping[SetName, SetOutcome] | None = None,
    drop: tuple[SetName, ...] = (),
    official: bool = True,
    docile_present: bool = False,
    git_sha: str = GIT_SHA,
) -> RunReport:
    """A report shaped like `evals.run.run_eval` output for `extractor_b@<version>`."""
    per_set = per_set or {}
    eb = per_set.get("extractbench", predictor)
    eb_block = f1_block(EXTRACTBENCH_GOLD, eb)
    built: dict[SetName, SetOutcome] = {
        "fatura": SetOutcome(
            set_name="fatura",
            status="scored",
            split="test",
            n=len(FATURA_GOLD),
            metrics={"field_f1": f1_block(FATURA_GOLD, per_set.get("fatura", predictor))},
            gate=GateResult(kind=GateKind.REPORT),
        ),
        "extractbench": SetOutcome(
            set_name="extractbench",
            status="scored",
            split="eval",
            n=len(EXTRACTBENCH_GOLD),
            metrics={
                "value_f1": eb_block,
                "grounding_f1": eb_block,
                "used_official_framework": official,
            },
            gate=GateResult(kind=GateKind.REPORT),
        ),
        "docile": _score_docile_set(manifest(docile_present=docile_present)),
        "uae": _score_uae_set(manifest()),
        "pint_ae": pint_ae(),
    }
    built.update(overrides or {})
    return RunReport(
        run_id=f"run-{version}",
        generated_at=datetime(2026, 9, 24, tzinfo=UTC),
        git_sha=git_sha,
        predictor=f"extractor_b@{version}",
        sets_requested=list(built),
        limit=None,
        results=[o for name, o in built.items() if name not in drop],
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def gate_report(predictor: Predictor, **overrides: Any) -> ExtractorGateReport:
    """P1-P5 numbers for one version; UAE F1s come from `score_uae` on `UAE_GOLD`."""
    gold = {doc.doc_id: doc.fields for doc in UAE_GOLD}
    uae = score_uae(UAE_GOLD, predictor(gold))
    values: dict[str, Any] = {
        "session_id": SESSION,
        "uae_test_digest": _digest("uae-test"),
        "latency_subset_digest": _digest("m3.4-200"),
        "uae_f1_overall": uae.overall.f1,
        "uae_f1_en": uae.en.f1,
        "uae_f1_ar": uae.ar.f1,
        "uae_n": len(UAE_GOLD),
        "valid_json_rate": 1.0,
        "latency_p50_s": 1.0,
        "latency_p95_s": 1.4,
        "latency_n": 200,
        "reference_a_f1_overall": 0.97,
        "reference_a_p50_s": 2.0,
        "extract_peak_vram_gb": 10.9,
        "sequential_fallback_recorded": False,
        "leakage_green": True,
        "base_revision": "rev-base",
        "lock_base_revision": "rev-base",
    }
    values.update(overrides)
    return ExtractorGateReport(**values)


def shadow(
    predictor: Predictor,
    *,
    version: str,
    report: RunReport | None = None,
    **gate_overrides: Any,
) -> ShadowEval:
    return ShadowEval(
        report=report or run_report(predictor, version=version),
        gates=gate_report(predictor, **gate_overrides),
    )


CHAMPION = corrupt(3)
"""The standing champion: three wrong fields per set (UAE pooled F1 0.97)."""
