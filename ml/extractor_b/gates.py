"""Extractor B promotion gates P1-P5 (ADR-0009 §1.8, measured on the served Q4_K_M + mmproj through
the llama-swap `extract` group): the sequential warm latency run for A and B on the 200-document
latency subset with whole-card nvidia-smi sampling, the P5 hygiene checks, the typed
`GateEvidence` file and `reports/extractor_b_gates.md`. Numbers only ever come from measurements."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scripts.smoke import VramSampler

from apx.common.types import Sha256Hex
from apx.extract.extractor import LlmExtractor
from apx.extract.interfaces import ExtractorId
from apx.llm.interfaces import LlmClient
from apx.parse.interfaces import ParsedDocument
from evals.bakeoff.subset import BAKEOFF_SUBSET_SIZE, SubsetError, uae_subset
from evals.extraction_a import prompt_builder_for
from evals.extraction_identity import RowIdentity
from evals.scorers.perf import LatencyStats
from ml.extractor_b.checkpoints import sha256_file
from ml.extractor_b.guards import check_leakage
from ml.extractor_b.interfaces import DataLeakError, RunRecord, TrainingError, TrainingSample
from ml.extractor_b.train import resolve_base, train_split_sha256
from ml.promotion.decide import (
    P1_MARGIN,
    P2_LATENCY_RATIO_MAX,
    P3_VALID_JSON_MIN,
    P4_VRAM_FALLBACK_MAX_GB,
    P4_VRAM_MAX_GB,
)
from ml.promotion.errors import PromotionError
from ml.promotion.evidence import ExtractorGateReport, Fraction

LATENCY_SUBSET_SIZE: Final[int] = BAKEOFF_SUBSET_SIZE
"""ADR-0009 P2: the 200-document M3.4 subset, defined once in `evals.bakeoff.subset` (first 200
UAE test sha256s in sorted order, i.e. a hash-random sample); the digest is recorded with every
run."""
A_ALIAS: Final[str] = "extractor-a"
CHALLENGER_ALIAS: Final[str] = "extractor-b-challenger"
EVIDENCE_SCHEMA_VERSION: Final[int] = 1

GateId = Literal["P1", "P2", "P3", "P4", "P5"]


class GateEvidenceError(PromotionError):
    """Gate evidence is missing, inconsistent or unmeasurable; no gate can be claimed."""


class VramUnavailableError(GateEvidenceError):
    """nvidia-smi returned no reading (or missed one) during the P4 window."""


def latency_subset(test_ids: Sequence[str]) -> list[str]:
    try:
        return uae_subset(test_ids, LATENCY_SUBSET_SIZE)
    except SubsetError as exc:
        raise GateEvidenceError(
            f"UAE test has {len(set(test_ids))} documents, P2 needs {LATENCY_SUBSET_SIZE}"
        ) from exc


# --- P4: whole-card VRAM ---------------------------------------------------------------------

SampleFn = Callable[[], float | None]


class StrictVramSampler:
    """`scripts.smoke.VramSampler` (nvidia-smi `memory.used`, max across GPUs, polled on a
    thread) that fails closed: a window with no reading or any missed reading raises
    `VramUnavailableError` instead of reporting a peak that may have missed the true one."""

    def __init__(self, sample_fn: SampleFn, *, interval_s: float = 0.2) -> None:
        self._sample_fn = sample_fn
        self._interval_s = interval_s
        self.samples = 0
        self.misses = 0
        self._inner: VramSampler | None = None

    def _sample(self) -> float | None:
        value = self._sample_fn()
        if value is None:
            self.misses += 1
        else:
            self.samples += 1
        return value

    def __enter__(self) -> StrictVramSampler:
        self._inner = VramSampler(interval_s=self._interval_s, sample_fn=self._sample)
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._inner is not None:
            self._inner.__exit__(*exc)

    @property
    def peak_gb(self) -> float:
        peak = None if self._inner is None else self._inner.peak_gb
        if peak is None or self.misses:
            raise VramUnavailableError(
                f"nvidia-smi gave {self.samples} readings and missed {self.misses}; "
                "P4 needs an unbroken whole-card sample of the extract stage"
            )
        return peak


# --- P2: sequential warm latency run ---------------------------------------------------------

ExtractFn = Callable[[str, ParsedDocument], Awaitable[float]]
"""Runs one extract call of `model` on `document` and returns that call's latency in seconds."""


@dataclass(frozen=True, slots=True)
class LatencyRun:
    a: LatencyStats
    b: LatencyStats
    peak_vram_gb: float
    """GiB: nvidia-smi MiB / 1024, the unit ADR-0013's budget was measured in."""
    vram_samples: int
    vram_misses: int = 0


def latency_stats(latencies_s: Sequence[float]) -> LatencyStats:
    """p50/p95 exactly as `evals.scorers.perf.PerfCollector.latency_stats` computes them."""
    if not latencies_s:
        raise GateEvidenceError("no timed extract calls recorded")
    samples = np.array(latencies_s, dtype=float)
    return LatencyStats(
        p50_s=float(np.percentile(samples, 50)),
        p95_s=float(np.percentile(samples, 95)),
        n=len(samples),
    )


async def run_latency(
    documents: Sequence[ParsedDocument],
    extract: ExtractFn,
    sampler: StrictVramSampler,
    *,
    a_model: str = A_ALIAS,
    b_model: str = CHALLENGER_ALIAS,
) -> LatencyRun:
    """One session: A then B, each warmed by one untimed call (which also pays llama-swap's
    A↔B swap), then every document once, strictly sequentially. The VRAM window spans both
    models and the swap, i.e. the whole `extract` stage with `swap: true` (ADR-0013)."""
    if not documents:
        raise GateEvidenceError("latency run has no documents")

    async def timed(model: str) -> LatencyStats:
        await extract(model, documents[0])  # warm-up: loads the model, not timed
        return latency_stats([await extract(model, doc) for doc in documents])

    with sampler:
        a = await timed(a_model)
        b = await timed(b_model)
    peak = sampler.peak_gb  # raises on any missed reading: P4 fails closed
    return LatencyRun(
        a=a, b=b, peak_vram_gb=peak, vram_samples=sampler.samples, vram_misses=sampler.misses
    )


def llm_extract_fn(client: LlmClient, *, a_model: str, b_model: str) -> ExtractFn:
    """`ExtractFn` over the live extractors: A with its text + image prompt, B image-only, the
    same builders as `evals.extraction_a`. The latency is the extract call's HTTP round trip
    (`LlmExtractor.last_latency_ms`), the number the A harness reports."""
    extractors = {
        a_model: LlmExtractor(
            client, prompt_builder_for(ExtractorId.A), model=a_model, extractor=ExtractorId.A
        ),
        b_model: LlmExtractor(
            client, prompt_builder_for(ExtractorId.B), model=b_model, extractor=ExtractorId.B
        ),
    }

    async def extract(model: str, document: ParsedDocument) -> float:
        extractor = extractors[model]
        await extractor.extract(document)
        if extractor.last_latency_ms is None:
            raise GateEvidenceError(f"{model} returned no latency")
        return extractor.last_latency_ms / 1000

    return extract


def extract_group_swaps(config_path: Path, members: Sequence[str]) -> bool:
    """True when the llama-swap `extract` group has `swap: true` and holds every alias in
    `members`: llama-swap then keeps only one of them loaded (the ADR-0002 sequential fallback,
    recorded in ADR-0013), so P4's ceiling is the 11.8 GB whole-card budget."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        group = config["routing"]["router"]["settings"]["groups"]["extract"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise GateEvidenceError(f"cannot read the extract group from {config_path}: {exc}") from exc
    return group.get("swap") is True and set(members) <= set(group.get("members", []))


# --- P5: hygiene -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hygiene:
    leakage_green: bool
    leakage_detail: str
    base_revision: str
    lock_base_revision: str


def check_hygiene(
    run_dir: Path, data_dir: Path, *, lock_path: Path, splits_path: Path | None = None
) -> Hygiene:
    """Leakage test on the run's own manifest (its sha256 must equal the run fingerprint's) plus
    the dev rows, against the train pool of evals/splits.json; and the trained base revision
    against the `extractor_b_train_base` pin in models.lock."""
    try:
        record = RunRecord.model_validate_json((run_dir / "run.json").read_bytes())
    except (OSError, ValidationError) as exc:
        raise GateEvidenceError(f"cannot read {run_dir / 'run.json'}: {exc}") from exc
    manifest = data_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise GateEvidenceError(f"run manifest {manifest} is missing")
    if sha256_file(manifest) != record.fingerprint.manifest_sha256:
        raise GateEvidenceError(
            f"{manifest} is not the manifest run {record.run_id} trained on (sha256 differs)"
        )
    paths = [manifest] + [p for p in (data_dir / "dev.jsonl",) if p.is_file()]
    try:
        shas = [
            TrainingSample.model_validate_json(line).document_sha256
            for path in paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except ValidationError as exc:
        raise GateEvidenceError(f"run manifest rows are invalid: {exc}") from exc
    try:
        check_leakage(shas, train_split_sha256(splits_path))
        green, detail = True, f"{len(shas)} rows, all on the train pool"
    except DataLeakError as exc:
        green, detail = False, str(exc)
    try:
        lock_revision = resolve_base({}, lock_path).revision
    except TrainingError as exc:
        raise GateEvidenceError(f"no base pin in {lock_path}: {exc}") from exc
    return Hygiene(green, detail, record.fingerprint.base_revision, lock_revision)


# --- Verdict and evidence --------------------------------------------------------------------


def _dec(value: float) -> Decimal:
    return Decimal(repr(value))  # same rule as ml.promotion.decide: compare the reported digits


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: GateId
    name: str
    measured: str
    rule: str
    passed: bool


def evaluate_gates(report: ExtractorGateReport) -> tuple[GateResult, ...]:
    """P1-P5 with the ADR-0009 §1.8 thresholds (shared with the M10.2 decider, never moved)."""
    f1, f1_a = _dec(report.uae_f1_overall), _dec(report.reference_a_f1_overall)
    p50, p50_a = _dec(report.latency_p50_s), _dec(report.reference_a_p50_s)
    valid, vram = _dec(report.valid_json_rate), _dec(report.extract_peak_vram_gb)
    vram_max = P4_VRAM_FALLBACK_MAX_GB if report.sequential_fallback_recorded else P4_VRAM_MAX_GB
    same_base = report.base_revision == report.lock_base_revision
    return (
        GateResult(
            gate="P1",
            name="UAE test micro field F1 (14 header fields, EN+AR pooled)",
            measured=f"B {f1:.4f} (EN {report.uae_f1_en:.4f}, AR {report.uae_f1_ar:.4f}); "
            f"A {f1_a:.4f}",
            rule=f"B >= A - {P1_MARGIN} ({f1_a - P1_MARGIN:.4f})",
            passed=f1 >= f1_a - P1_MARGIN,
        ),
        GateResult(
            gate="P2",
            name=f"p50 extract latency, warm, sequential, {report.latency_n} documents",
            measured=f"B {p50:.2f} s (p95 {report.latency_p95_s:.2f} s); A {p50_a:.2f} s",
            rule=f"B <= {P2_LATENCY_RATIO_MAX} x A ({P2_LATENCY_RATIO_MAX * p50_a:.2f} s)",
            passed=p50 <= P2_LATENCY_RATIO_MAX * p50_a,
        ),
        GateResult(
            gate="P3",
            name=f"valid JSON on UAE test ({report.uae_n} documents)",
            measured=f"{100 * report.valid_json_rate:.2f}%",
            rule=f">= {100 * P3_VALID_JSON_MIN:.0f}%",
            passed=valid >= P3_VALID_JSON_MIN,
        ),
        GateResult(
            gate="P4",
            name="extract-stage peak VRAM, whole card (GiB = nvidia-smi MiB / 1024)",
            measured=f"{report.extract_peak_vram_gb:.2f} GiB",
            rule=f"<= {vram_max} (ADR-0013 budget, same sampler and unit; llama-swap extract "
            f"group swap: {report.sequential_fallback_recorded})",
            passed=vram <= vram_max,
        ),
        GateResult(
            gate="P5",
            name="hygiene: leakage test on the run manifest, base revision == models.lock",
            measured=f"leakage green: {report.leakage_green}; base {report.base_revision}",
            rule=f"leakage green and base == {report.lock_base_revision}",
            passed=report.leakage_green and same_base,
        ),
    )


class GateEvidence(BaseModel):
    """`reports/extractor_b_gates.json`: what `promote` re-reads. Its sha256 is also stored as a
    tag on the MLflow model version, so an edited file no longer matches the registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    registered_model: str
    model_version: str = Field(min_length=1)
    run_id: str
    mlflow_run_id: str
    gguf_sha256: Sha256Hex
    mmproj_sha256: Sha256Hex
    champion_at_measurement: str | None
    """The `champion` alias when the gates were measured; `promote` compares-and-sets on it."""
    challenger_alias: str
    reference_alias: str
    measured_at: datetime
    git_sha: str
    report: ExtractorGateReport
    reference_a_f1_en: Fraction
    reference_a_f1_ar: Fraction
    reference_a_p95_s: float = Field(ge=0.0)
    uae_n_en: int = Field(ge=0)
    uae_n_ar: int = Field(ge=0)
    vram_samples: int = Field(ge=1)
    vram_missed_readings: int = Field(ge=0)
    """Always 0 in a written file: any missed nvidia-smi reading stops the gate run."""
    a_rows_sha256: Sha256Hex
    b_rows_sha256: Sha256Hex
    a_rows_identity: RowIdentity
    b_rows_identity: RowIdentity
    leakage_detail: str
    results: tuple[GateResult, ...]

    @property
    def all_passed(self) -> bool:
        return len(self.results) == 5 and all(r.passed for r in self.results)


def evidence_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_evidence(path: Path, evidence: GateEvidence) -> str:
    """Write the evidence JSON and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return evidence_sha256(path)


def load_evidence(path: Path) -> GateEvidence:
    """Strict load; the stored verdict must equal a fresh `evaluate_gates` on the stored
    measurements, so a hand-edited PASS is rejected."""
    try:
        evidence = GateEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise GateEvidenceError(f"cannot read gate evidence {path}: {exc}") from exc
    if evidence.results != evaluate_gates(evidence.report):
        raise GateEvidenceError(f"{path}: stored gate results differ from its measurements")
    return evidence


def render_markdown(evidence: GateEvidence) -> str:
    r = evidence.report
    verdict = "PASS: all five gates hold" if evidence.all_passed else "FAIL"
    lines = [
        "# Extractor B promotion gates (M7.1, ADR-0009 §1.8)",
        "",
        f"Registered model `{evidence.registered_model}` version {evidence.model_version} "
        f"(run `{evidence.run_id}`, MLflow run `{evidence.mlflow_run_id}`), served as "
        f"`{evidence.challenger_alias}`; reference `{evidence.reference_alias}`. Session "
        f"`{r.session_id}`, measured {evidence.measured_at.isoformat()} at `{evidence.git_sha}`. "
        "Generated by `python -m ml.extractor_b.promote gates`; the numbers below are read from "
        "`reports/extractor_b_gates.json`.",
        "",
        f"**Verdict: {verdict}.**",
        "",
        "| Gate | What | Measured | Rule | Result |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {g.gate} | {g.name} | {g.measured} | {g.rule} | {'PASS' if g.passed else 'FAIL'} |"
        for g in evidence.results
    ]
    lines += [
        "",
        "## Detail",
        "",
        "| | Extractor A | Extractor B |",
        "|---|---:|---:|",
        f"| Field F1, pooled ({r.uae_n} docs) | {r.reference_a_f1_overall:.4f} | "
        f"{r.uae_f1_overall:.4f} |",
        f"| Field F1, EN ({evidence.uae_n_en} docs) | {evidence.reference_a_f1_en:.4f} | "
        f"{r.uae_f1_en:.4f} |",
        f"| Field F1, AR ({evidence.uae_n_ar} docs) | {evidence.reference_a_f1_ar:.4f} | "
        f"{r.uae_f1_ar:.4f} |",
        f"| Latency p50 / p95 ({r.latency_n} docs) | {r.reference_a_p50_s:.2f} s / "
        f"{evidence.reference_a_p95_s:.2f} s | {r.latency_p50_s:.2f} s / {r.latency_p95_s:.2f} s |",
        "",
        f"- UAE test digest `{r.uae_test_digest}`; latency subset digest "
        f"`{r.latency_subset_digest}`.",
        f"- VRAM: {evidence.vram_samples} nvidia-smi readings ({evidence.vram_missed_readings} "
        "missed) over the extract stage (A and B with the llama-swap swap), whole card including "
        "the desktop, in GiB (MiB / 1024).",
        f"- Rows: A `{evidence.a_rows_sha256}` ({evidence.a_rows_identity.model}, GGUF "
        f"`{evidence.a_rows_identity.gguf_sha256}`, prompt "
        f"`{evidence.a_rows_identity.prompt_version}`); B `{evidence.b_rows_sha256}` "
        f"({evidence.b_rows_identity.model}, GGUF `{evidence.b_rows_identity.gguf_sha256}`, "
        f"prompt `{evidence.b_rows_identity.prompt_version}`).",
        f"- Leakage: {evidence.leakage_detail}.",
        f"- Export: Q4_K_M `{evidence.gguf_sha256}`, mmproj `{evidence.mmproj_sha256}`.",
        "",
    ]
    return "\n".join(lines)


def gate_tags(evidence: GateEvidence, sha256: str) -> dict[str, str]:
    """MLflow model-version tags recording the verdict and binding the evidence file."""
    tags = {"gates.evidence_sha256": sha256, "gates.all_passed": str(evidence.all_passed)}
    tags.update({f"gates.{g.gate}": "PASS" if g.passed else "FAIL" for g in evidence.results})
    tags["gates.session_id"] = evidence.report.session_id
    return tags
