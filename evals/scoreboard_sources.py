"""Strict readers for the measured reports the PRD §5 scoreboard draws on (ADR-0005: numbers are
never hand-written): the chat_evals, attacks_e2e, trust_gate, extractor_b_gates and bakeoff JSON
evidence, the real-data track's JSON reports (ADR-0019), plus the generated extraction_a,
gold_xml, fraud, guard, smoke, e2e_pipeline, parse, factory_selfcheck and extractor_b markdown
reports. Every
parser is pure text-in, typed-values-out and raises `ReportFormatError` on any format drift
rather than guessing; evals/scoreboard.py turns these values into scoreboard rows."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError as PydanticValidationError

from evals.real.models import BeforeAfterReport, RealEvalReport

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

StampKind = Literal["generated", "committed", "file mtime"]


class ReportFormatError(ValueError):
    """A report exists but no longer matches the format its generator writes. The scoreboard
    fails loudly on this instead of guessing a number (ADR-0005)."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


@dataclass(frozen=True)
class SourceStamp:
    """Where a scoreboard number came from and when that source was produced: the timestamp
    printed inside the report when it has one, else its last git commit, else its file mtime."""

    path: str
    at: datetime
    kind: StampKind

    def render(self) -> str:
        when = self.at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return f"`{self.path}` ({self.kind} {when})"


# --- small strict helpers ---------------------------------------------------------------------


def _match(source: str, pattern: str, text: str, what: str) -> re.Match[str]:
    """The one required regex match for `what`, or ReportFormatError naming what drifted."""
    found = re.search(pattern, text, flags=re.MULTILINE)
    if found is None:
        raise ReportFormatError(source, f"missing {what} (expected /{pattern}/)")
    return found


def _table_rows(source: str, text: str, header: str) -> list[list[str]]:
    """Cells of the GFM table whose header line is exactly `header`; raises if the header is
    absent, the separator line is missing or the table has no data rows."""
    lines = text.splitlines()
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise ReportFormatError(source, f"missing table header {header!r}") from exc
    if start + 1 >= len(lines) or not lines[start + 1].startswith("|---"):
        raise ReportFormatError(source, f"table {header!r} has no separator line")
    n_cols = header.count("|") - 1
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != n_cols:
            raise ReportFormatError(source, f"row {line!r} has {len(cells)} cells, want {n_cols}")
        rows.append(cells)
    if not rows:
        raise ReportFormatError(source, f"table {header!r} has no data rows")
    return rows


def _pct_cell(source: str, cell: str, what: str) -> float:
    found = re.fullmatch(r"(\d+(?:\.\d+)?)%", cell)
    if found is None:
        raise ReportFormatError(source, f"{what}: {cell!r} is not a percentage")
    return float(found.group(1))


def _float_cell(source: str, cell: str, what: str) -> float:
    try:
        return float(cell)
    except ValueError as exc:
        raise ReportFormatError(source, f"{what}: {cell!r} is not a number") from exc


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _iso(source: str, value: str, what: str) -> datetime:
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ReportFormatError(source, f"{what}: {value!r} is not an ISO timestamp") from exc


# --- reports/chat_evals.json (M9.5) -----------------------------------------------------------


@dataclass(frozen=True)
class ChatModelScores:
    label: str
    alias: str
    n_tasks: int
    tool_call_accuracy: float
    citation_faithfulness: float
    banking_user_tasks: int
    banking_injection_pairs: int
    utility: float
    utility_under_attack: float
    attack_success_rate: float


@dataclass(frozen=True)
class ChatEvals:
    generated_at: datetime
    dry_run: bool
    models: tuple[ChatModelScores, ...]

    def model(self, label: str) -> ChatModelScores | None:
        return next((m for m in self.models if m.label == label), None)


def _get(source: str, obj: Any, key: str, kind: type[Any] | tuple[type[Any], ...]) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise ReportFormatError(source, f"missing key {key!r}")
    value = obj[key]
    if isinstance(value, bool) and kind is not bool:
        raise ReportFormatError(source, f"key {key!r} is a bool, want {kind}")
    if not isinstance(value, kind):
        raise ReportFormatError(source, f"key {key!r} is {type(value).__name__}, want {kind}")
    return value


def parse_chat_evals(text: str, source: str = "reports/chat_evals.json") -> ChatEvals:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportFormatError(source, f"not valid JSON ({exc})") from exc
    num = (int, float)
    models: list[ChatModelScores] = []
    for entry in _get(source, data, "models", list):
        chat = _get(source, entry, "chat", dict)
        banking = _get(source, entry, "banking", dict)
        models.append(
            ChatModelScores(
                label=_get(source, entry, "label", str),
                alias=_get(source, entry, "alias", str),
                n_tasks=_get(source, chat, "n_tasks", int),
                tool_call_accuracy=float(_get(source, chat, "tool_call_accuracy", num)),
                citation_faithfulness=float(_get(source, chat, "citation_faithfulness", num)),
                banking_user_tasks=_get(source, banking, "user_tasks", int),
                banking_injection_pairs=_get(source, banking, "injection_pairs", int),
                utility=float(_get(source, banking, "utility", num)),
                utility_under_attack=float(_get(source, banking, "utility_under_attack", num)),
                attack_success_rate=float(_get(source, banking, "attack_success_rate", num)),
            )
        )
    if not models:
        raise ReportFormatError(source, "no models")
    return ChatEvals(
        generated_at=_iso(source, _get(source, data, "generated_at", str), "generated_at"),
        dry_run=_get(source, data, "dry_run", bool),
        models=tuple(models),
    )


# --- reports/extraction_a.md (M3.3) -----------------------------------------------------------


@dataclass(frozen=True)
class ExtractionAReport:
    model: str
    n_docs: int
    n_arabic: int
    valid_json_pct: float
    valid_json_verdict: str
    overall_field_accuracy_pct: float


def parse_extraction_a(text: str, source: str = "reports/extraction_a.md") -> ExtractionAReport:
    head = _match(
        source,
        r"^Model `(?P<model>[^`]+)` on the UAE factory test split, (?P<n>\d+) documents "
        r"\((?P<ar>\d+) with Arabic text\)\.",
        text,
        "model/document-count line",
    )
    valid = _match(
        source,
        r"^\| Valid JSON \| (?P<pct>\d+\.\d{2})% \| ≥ 99% (?P<verdict>PASS|FAIL) \|$",
        text,
        "Valid JSON row",
    )
    overall = _match(
        source, r"^Overall field accuracy: (?P<pct>\d+\.\d{2})%\.$", text, "overall accuracy"
    )
    return ExtractionAReport(
        model=head.group("model"),
        n_docs=int(head.group("n")),
        n_arabic=int(head.group("ar")),
        valid_json_pct=float(valid.group("pct")),
        valid_json_verdict=valid.group("verdict"),
        overall_field_accuracy_pct=float(overall.group("pct")),
    )


# --- reports/gold_xml.md (M2.5) ---------------------------------------------------------------


@dataclass(frozen=True)
class GoldXmlReport:
    passed: int
    total: int


def parse_gold_xml(text: str, source: str = "reports/gold_xml.md") -> GoldXmlReport:
    found = _match(
        source,
        r"Schematron pass rate: \*\*(?P<passed>\d+)/(?P<total>\d+) \(\d+\.\d{2}%\)\*\*",
        text,
        "schematron pass-rate line",
    )
    passed, total = int(found.group("passed")), int(found.group("total"))
    if total == 0 or passed > total:
        raise ReportFormatError(source, f"impossible pass count {passed}/{total}")
    return GoldXmlReport(passed=passed, total=total)


# --- reports/fraud.md (M8.2) ------------------------------------------------------------------

FRAUD_HEADER = (
    "| Typology | Recall @ 1% FPR (min) | Gate ≥ 90% | Deployed recall (min) | Rules only (min) |"
)


@dataclass(frozen=True)
class FraudReport:
    recall_pct: dict[str, float]
    verdicts: dict[str, str]
    overall: str

    @property
    def weakest(self) -> tuple[str, float]:
        name = min(self.recall_pct, key=lambda k: self.recall_pct[k])
        return name, self.recall_pct[name]


def parse_fraud(text: str, source: str = "reports/fraud.md") -> FraudReport:
    recall: dict[str, float] = {}
    verdicts: dict[str, str] = {}
    for cells in _table_rows(source, text, FRAUD_HEADER):
        typology, pct, verdict = cells[0], cells[1], cells[2]
        if verdict not in ("PASS", "FAIL"):
            raise ReportFormatError(source, f"typology {typology}: verdict {verdict!r}")
        recall[typology] = _pct_cell(source, pct, f"typology {typology} recall")
        verdicts[typology] = verdict
    overall = _match(source, r"^Overall: (?P<v>PASS|FAIL)\.$", text, "Overall line").group("v")
    expected = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"
    if overall != expected:
        raise ReportFormatError(source, f"Overall {overall} contradicts per-typology verdicts")
    return FraudReport(recall_pct=recall, verdicts=verdicts, overall=overall)


# --- reports/guard.md (M6.4) ------------------------------------------------------------------

GUARD_HEADER = (
    "| model | operating point | threshold | precision | recall | F1 | FPR | hard-neg FPR |"
)
OURS_MARKER = "(ours)"


@dataclass(frozen=True)
class GuardPoint:
    model: str
    operating_point: str
    precision: float | None
    recall: float | None
    f1: float | None

    @property
    def measured(self) -> bool:
        return self.f1 is not None


@dataclass(frozen=True)
class GuardReport:
    points: tuple[GuardPoint, ...]

    @property
    def ours(self) -> list[GuardPoint]:
        return [p for p in self.points if OURS_MARKER in p.model]

    @property
    def best_baseline(self) -> GuardPoint | None:
        measured = [p for p in self.points if OURS_MARKER not in p.model and p.measured]
        return max(measured, key=lambda p: p.f1 or 0.0) if measured else None


def parse_guard(text: str, source: str = "reports/guard.md") -> GuardReport:
    points: list[GuardPoint] = []
    for cells in _table_rows(source, text, GUARD_HEADER):
        model, op, _threshold, precision, recall, f1 = cells[:6]
        numbers = (precision, recall, f1)
        if all(n == "" for n in numbers):
            points.append(GuardPoint(model, op, None, None, None))
            continue
        if any(n == "" for n in numbers):
            raise ReportFormatError(source, f"{model} / {op}: partially filled P/R/F1")
        points.append(
            GuardPoint(
                model,
                op,
                _float_cell(source, precision, "precision"),
                _float_cell(source, recall, "recall"),
                _float_cell(source, f1, "F1"),
            )
        )
    if not any(OURS_MARKER in p.model for p in points):
        raise ReportFormatError(
            source, f"no row for our fine-tune (model containing {OURS_MARKER})"
        )
    return GuardReport(points=tuple(points))


# --- reports/smoke.md (M0.5) ------------------------------------------------------------------

SMOKE_STAGE_HEADER = "| Stage | Peak VRAM (GB) | Within budget |"


@dataclass(frozen=True)
class SmokeReport:
    generated_at: datetime
    stage_peak_gb: dict[str, float]

    @property
    def max_stage(self) -> tuple[str, float]:
        name = max(self.stage_peak_gb, key=lambda k: self.stage_peak_gb[k])
        return name, self.stage_peak_gb[name]


def parse_smoke(text: str, source: str = "reports/smoke.md") -> SmokeReport:
    stamp = _match(source, r"^Generated: (?P<at>\S+)$", text, "Generated line").group("at")
    peaks = {
        cells[0]: _float_cell(source, cells[1], f"stage {cells[0]} peak VRAM")
        for cells in _table_rows(source, text, SMOKE_STAGE_HEADER)
    }
    return SmokeReport(generated_at=_iso(source, stamp, "Generated"), stage_peak_gb=peaks)


# --- reports/e2e_pipeline*.md (M5.6) ----------------------------------------------------------


@dataclass(frozen=True)
class E2eReport:
    n_docs: int
    attacks_stopped: int
    attacks_total: int
    median_s: int | None
    """None when no document reached a routing decision (the writer then prints no timing)."""
    max_s: int | None


def parse_e2e(text: str, source: str = "reports/e2e_pipeline.md") -> E2eReport:
    docs = _match(source, r"^(?P<n>\d+) documents dropped into the inbox", text, "document count")
    attacks = _match(
        source,
        r"^- Attack PDFs kept out of payment: \*\*(?P<ok>\d+)/(?P<n>\d+)\*\*"
        r"(?: \(\d+ of them only because the workflow failed before routing\))?$",
        text,
        "attack line",
    )
    timing = _match(
        source,
        r"^(?:- Seconds from drop to routing decision: median (?P<med>\d+), max (?P<max>\d+)"
        r"|- No document reached a routing decision)$",
        text,
        "timing line",
    )
    routed = timing.group("med") is not None
    return E2eReport(
        n_docs=int(docs.group("n")),
        attacks_stopped=int(attacks.group("ok")),
        attacks_total=int(attacks.group("n")),
        median_s=int(timing.group("med")) if routed else None,
        max_s=int(timing.group("max")) if routed else None,
    )


# --- reports/parse.md (M3.1) ------------------------------------------------------------------


@dataclass(frozen=True)
class ParseReport:
    generated_at: datetime
    fields: int
    iou_pass_rate: float
    gate_rate: float


def parse_parse(text: str, source: str = "reports/parse.md") -> ParseReport:
    stamp = _match(source, r"^Generated: (?P<at>\S+)$", text, "Generated line").group("at")
    gate = _match(
        source, r"^- Gate: IoU >= 0\.5 on >= (?P<pct>\d+)% of labeled fields$", text, "gate line"
    )
    overall = _match(
        source, r"^\| Overall \| (?P<n>\d+) \| (?P<rate>[01]\.\d+) \|$", text, "Overall row"
    )
    return ParseReport(
        generated_at=_iso(source, stamp, "Generated"),
        fields=int(overall.group("n")),
        iou_pass_rate=float(overall.group("rate")),
        gate_rate=int(gate.group("pct")) / 100,
    )


# --- reports/factory_selfcheck.md (M2.4) ------------------------------------------------------


@dataclass(frozen=True)
class SelfcheckReport:
    matched: int
    total: int
    verdict: str
    threshold_pct: float


def parse_selfcheck(text: str, source: str = "reports/factory_selfcheck.md") -> SelfcheckReport:
    found = _match(
        source,
        r"^\*\*Overall exact match: (?P<ok>\d+)/(?P<n>\d+) \(\d+\.\d{2}%\) -- (?P<v>PASS|FAIL)"
        r"\*\* \(threshold (?P<th>\d+(?:\.\d+)?)%\)$",
        text,
        "overall exact-match line",
    )
    return SelfcheckReport(
        matched=int(found.group("ok")),
        total=int(found.group("n")),
        verdict=found.group("v"),
        threshold_pct=float(found.group("th")),
    )


# --- reports/extractor_b.md (M7.1) ------------------------------------------------------------


@dataclass(frozen=True)
class ExtractorBReport:
    generated_at: datetime
    run_id: str
    dev_docs: int
    selected_step: int
    dev_f1: float
    dev_valid_json_pct: float


def parse_extractor_b(text: str, source: str = "reports/extractor_b.md") -> ExtractorBReport:
    head = _match(
        source,
        r"^Run `(?P<run>[^`]+)` .*Generated by `python -m ml\.extractor_b\.report` on "
        r"(?P<at>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC .*Dev set: (?P<n>\d+) UAE documents\.$",
        text,
        "run/generated/dev-set line",
    )
    step = _match(
        source, r"^\| Selected step \(`best\.json`\) \| \*\*(?P<s>\d+)\*\* \|$", text, "step"
    )
    f1 = _match(
        source, r"^\| Dev micro field F1 at the selected step \| (?P<f1>\d+\.\d+) \|$", text, "F1"
    )
    valid = _match(
        source,
        r"^\| Valid JSON at the selected step \| (?P<pct>\d+(?:\.\d+)?)% \|$",
        text,
        "valid JSON row",
    )
    return ExtractorBReport(
        generated_at=_iso(source, head.group("at").replace(" ", "T") + "+00:00", "generated"),
        run_id=head.group("run"),
        dev_docs=int(head.group("n")),
        selected_step=int(step.group("s")),
        dev_f1=float(f1.group("f1")),
        dev_valid_json_pct=float(valid.group("pct")),
    )


# --- JSON helpers -----------------------------------------------------------------------------


def _load_json(source: str, text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportFormatError(source, f"not valid JSON ({exc})") from exc


def _num(source: str, obj: Any, key: str) -> float:
    return float(_get(source, obj, key, (int, float)))


def _opt_num(source: str, obj: Any, key: str) -> float | None:
    if not isinstance(obj, dict) or key not in obj:
        raise ReportFormatError(source, f"missing key {key!r}")
    return None if obj[key] is None else _num(source, obj, key)


def _rate(source: str, obj: Any, key: str) -> float:
    value = _num(source, obj, key)
    if not 0.0 <= value <= 1.0:
        raise ReportFormatError(source, f"key {key!r} = {value} is not a rate in [0, 1]")
    return value


def _opt_rate(source: str, obj: Any, key: str) -> float | None:
    if not isinstance(obj, dict) or key not in obj:
        raise ReportFormatError(source, f"missing key {key!r}")
    return None if obj[key] is None else _rate(source, obj, key)


def _count(source: str, obj: Any, key: str) -> int:
    value: int = _get(source, obj, key, int)
    if value < 0:
        raise ReportFormatError(source, f"key {key!r} = {value} is negative")
    return value


# --- reports/attacks_e2e.json (M6.6) ----------------------------------------------------------

ATTACK_GATE_STATUSES: frozenset[str] = frozenset(
    {"PASS", "PARTIAL", "FAIL_PAYMENT_CHANGED", "FAIL_MASTER_CHANGED", "FAIL_INCOMPLETE"}
)


@dataclass(frozen=True)
class AttacksE2eReport:
    """The M6.6 end-to-end attack run as `evals/attacks_e2e/report.py::to_json` writes it."""

    gate_status: str
    gate_detail: str
    planned: int
    suite_size: int
    cases: int
    conclusive: int
    payment_changed: int
    asr: float | None
    timed_out: int
    missing: int
    stale: int


def parse_attacks_e2e(text: str, source: str = "reports/attacks_e2e.json") -> AttacksE2eReport:
    data = _load_json(source, text)
    gate = _get(source, data, "gate", dict)
    status = _get(source, gate, "status", str)
    if status not in ATTACK_GATE_STATUSES:
        raise ReportFormatError(source, f"unknown gate status {status!r}")
    overall = _get(source, data, "overall", dict)
    cases, conclusive = _count(source, overall, "cases"), _count(source, overall, "conclusive")
    changed = _count(source, overall, "payment_changed")
    if conclusive > cases or changed > cases:
        raise ReportFormatError(source, f"impossible counts {changed}/{conclusive}/{cases}")
    return AttacksE2eReport(
        gate_status=status,
        gate_detail=_get(source, gate, "detail", str),
        planned=_count(source, data, "planned"),
        suite_size=_count(source, data, "suite_size"),
        cases=cases,
        conclusive=conclusive,
        payment_changed=changed,
        asr=_opt_rate(source, overall, "asr"),
        timed_out=_count(source, overall, "timed_out"),
        missing=len(_get(source, data, "missing", list)),
        stale=len(_get(source, data, "stale", list)),
    )


# --- reports/trust_gate.json (M7.5) -----------------------------------------------------------


@dataclass(frozen=True)
class TrustAlpha:
    """UAE test numbers at one alpha's calibrated threshold (None lambda: nothing accepted)."""

    alpha: float
    lambda_hat: float | None
    accepted: int
    wrong: int
    n_fields: int
    error: float | None
    stp: float


@dataclass(frozen=True)
class TrustGateReport:
    """`evals/trust_gate.py::evidence`: the calibration's gate verdict and its test outcomes."""

    generated_at: datetime
    gate_alpha: float
    gate_passed: bool
    gate_detail: str
    test: tuple[TrustAlpha, ...] | None

    def at(self, alpha: float) -> TrustAlpha | None:
        return next((t for t in self.test or () if abs(t.alpha - alpha) < 1e-12), None)


def parse_trust_gate(text: str, source: str = "reports/trust_gate.json") -> TrustGateReport:
    data = _load_json(source, text)
    gate = _get(source, data, "gate", dict)
    if "test" not in data:
        raise ReportFormatError(source, "missing key 'test'")
    raw_test = data["test"]
    test: tuple[TrustAlpha, ...] | None = None
    if raw_test is not None:
        if not isinstance(raw_test, dict) or not raw_test:
            raise ReportFormatError(source, "key 'test' is not a non-empty object")
        outcomes: list[TrustAlpha] = []
        for key, entry in raw_test.items():
            try:
                alpha = float(key)
            except ValueError as exc:
                raise ReportFormatError(source, f"test alpha {key!r} is not a number") from exc
            accepted, wrong = _count(source, entry, "accepted"), _count(source, entry, "wrong")
            if wrong > accepted:
                raise ReportFormatError(source, f"alpha {key}: {wrong} wrong of {accepted}")
            outcomes.append(
                TrustAlpha(
                    alpha=alpha,
                    lambda_hat=_opt_num(source, entry, "lambda_hat"),
                    accepted=accepted,
                    wrong=wrong,
                    n_fields=_count(source, entry, "n_fields"),
                    error=_opt_rate(source, entry, "error"),
                    stp=_rate(source, entry, "stp"),
                )
            )
        test = tuple(sorted(outcomes, key=lambda t: t.alpha))
    return TrustGateReport(
        generated_at=_iso(source, _get(source, data, "generated_at", str), "generated_at"),
        gate_alpha=_rate(source, gate, "alpha"),
        gate_passed=_get(source, gate, "passed", bool),
        gate_detail=_get(source, gate, "detail", str),
        test=test,
    )


# --- reports/extractor_b_gates.json (M7.1 P1-P5) ----------------------------------------------

EXTRACTOR_B_GATE_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class ExtractorBGates:
    """The fields of `ml/extractor_b/gates.py::GateEvidence` the scoreboard shows."""

    measured_at: datetime
    model_version: str
    uae_f1_overall: float
    uae_f1_en: float
    uae_f1_ar: float
    uae_n: int
    uae_n_en: int
    uae_n_ar: int
    reference_a_f1_en: float
    reference_a_f1_ar: float
    valid_json_rate: float
    latency_p50_s: float
    latency_p95_s: float
    latency_n: int
    extract_peak_vram_gb: float
    results: tuple[tuple[str, bool], ...]

    @property
    def failed(self) -> list[str]:
        return [gate for gate, passed in self.results if not passed]

    @property
    def all_passed(self) -> bool:
        return len(self.results) == len(EXTRACTOR_B_GATE_IDS) and not self.failed


def parse_extractor_b_gates(
    text: str, source: str = "reports/extractor_b_gates.json"
) -> ExtractorBGates:
    data = _load_json(source, text)
    if _get(source, data, "schema_version", int) != 1:
        raise ReportFormatError(source, f"schema_version {data['schema_version']} is not 1")
    report = _get(source, data, "report", dict)
    results: list[tuple[str, bool]] = []
    for entry in _get(source, data, "results", list):
        gate = _get(source, entry, "gate", str)
        if gate not in EXTRACTOR_B_GATE_IDS:
            raise ReportFormatError(source, f"unknown gate {gate!r}")
        results.append((gate, _get(source, entry, "passed", bool)))
    if sorted(g for g, _ in results) != list(EXTRACTOR_B_GATE_IDS):
        raise ReportFormatError(source, f"gates {[g for g, _ in results]} are not P1-P5 once each")
    return ExtractorBGates(
        measured_at=_iso(source, _get(source, data, "measured_at", str), "measured_at"),
        model_version=_get(source, data, "model_version", str),
        uae_f1_overall=_rate(source, report, "uae_f1_overall"),
        uae_f1_en=_rate(source, report, "uae_f1_en"),
        uae_f1_ar=_rate(source, report, "uae_f1_ar"),
        uae_n=_count(source, report, "uae_n"),
        uae_n_en=_count(source, data, "uae_n_en"),
        uae_n_ar=_count(source, data, "uae_n_ar"),
        reference_a_f1_en=_rate(source, data, "reference_a_f1_en"),
        reference_a_f1_ar=_rate(source, data, "reference_a_f1_ar"),
        valid_json_rate=_rate(source, report, "valid_json_rate"),
        latency_p50_s=_num(source, report, "latency_p50_s"),
        latency_p95_s=_num(source, report, "latency_p95_s"),
        latency_n=_count(source, report, "latency_n"),
        extract_peak_vram_gb=_num(source, report, "extract_peak_vram_gb"),
        results=tuple(results),
    )


# --- reports/bakeoff.json (M3.4) --------------------------------------------------------------
# The machine-readable twin of reports/bakeoff.md, written by evals/bakeoff/report.py::to_json
# (extra keys are ignored); without the file the bake-off shows as pending. Schema version 1:
#   {"schema_version": 1, "generated_at": ISO-8601,
#    "decision": {"status": "PENDING" | "KEEP" | "REPLACE", "champion": str},
#    "arms": [{"arm": str, "verdict": str, "complete": bool, "uae_rows": int, "eb_rows": int,
#              "stopped_at_look": int or null (futility stop, ADR-0016 amendment 1),
#              "uae_f1_en" | "uae_f1_ar" | "uae_f1_pooled" | "uae_valid_json" |
#              "eb_value_f1" | "eb_valid_json": rate in [0, 1] or null,
#              "p50_s" | "p95_s": seconds or null}, ...]}

BAKEOFF_DECISIONS: frozenset[str] = frozenset({"PENDING", "KEEP", "REPLACE"})


@dataclass(frozen=True)
class BakeoffArm:
    arm: str
    verdict: str
    complete: bool
    uae_rows: int
    eb_rows: int
    uae_f1_en: float | None
    uae_f1_ar: float | None
    uae_f1_pooled: float | None
    uae_valid_json: float | None
    eb_value_f1: float | None
    eb_valid_json: float | None
    p50_s: float | None
    p95_s: float | None
    stopped_at_look: int | None


@dataclass(frozen=True)
class BakeoffReport:
    generated_at: datetime
    decision: str
    champion: str
    arms: tuple[BakeoffArm, ...]

    @property
    def champion_arm(self) -> BakeoffArm | None:
        return next((a for a in self.arms if a.arm == self.champion), None)


def _opt_look(source: str, entry: Any) -> int | None:
    if not isinstance(entry, dict) or "stopped_at_look" not in entry:
        raise ReportFormatError(source, "missing key 'stopped_at_look'")
    return None if entry["stopped_at_look"] is None else _count(source, entry, "stopped_at_look")


def parse_bakeoff(text: str, source: str = "reports/bakeoff.json") -> BakeoffReport:
    data = _load_json(source, text)
    if _get(source, data, "schema_version", int) != 1:
        raise ReportFormatError(source, f"schema_version {data['schema_version']} is not 1")
    decision = _get(source, data, "decision", dict)
    status = _get(source, decision, "status", str)
    if status not in BAKEOFF_DECISIONS:
        raise ReportFormatError(source, f"unknown decision status {status!r}")
    arms = tuple(
        BakeoffArm(
            arm=_get(source, entry, "arm", str),
            verdict=_get(source, entry, "verdict", str),
            complete=_get(source, entry, "complete", bool),
            uae_rows=_count(source, entry, "uae_rows"),
            eb_rows=_count(source, entry, "eb_rows"),
            uae_f1_en=_opt_rate(source, entry, "uae_f1_en"),
            uae_f1_ar=_opt_rate(source, entry, "uae_f1_ar"),
            uae_f1_pooled=_opt_rate(source, entry, "uae_f1_pooled"),
            uae_valid_json=_opt_rate(source, entry, "uae_valid_json"),
            eb_value_f1=_opt_rate(source, entry, "eb_value_f1"),
            eb_valid_json=_opt_rate(source, entry, "eb_valid_json"),
            p50_s=_opt_num(source, entry, "p50_s"),
            p95_s=_opt_num(source, entry, "p95_s"),
            stopped_at_look=_opt_look(source, entry),
        )
        for entry in _get(source, data, "arms", list)
    )
    if not arms:
        raise ReportFormatError(source, "no arms")
    report = BakeoffReport(
        generated_at=_iso(source, _get(source, data, "generated_at", str), "generated_at"),
        decision=status,
        champion=_get(source, decision, "champion", str),
        arms=arms,
    )
    if status != "PENDING" and report.champion_arm is None:
        raise ReportFormatError(source, f"champion {report.champion!r} is not an arm")
    return report


# --- real-data track (ADR-0019 §6) --------------------------------------------------------------


def parse_real_eval(text: str, source: str) -> RealEvalReport:
    """One `reports/real_<set>.<extractor>-<version>.json` (`evals.real.run.report_all`)."""
    try:
        return RealEvalReport.model_validate_json(text)
    except PydanticValidationError as exc:
        raise ReportFormatError(
            source, f"not a RealEvalReport: {exc.error_count()} error(s)"
        ) from exc


def parse_real_before_after(text: str, source: str) -> tuple[BeforeAfterReport, ...]:
    """`reports/real_before_after.json`: the paired B comparisons, one per set and split."""
    data = _load_json(source, text)
    entries = _get(source, data, "comparisons", list)
    try:
        return tuple(BeforeAfterReport.model_validate(entry) for entry in entries)
    except PydanticValidationError as exc:
        raise ReportFormatError(
            source, f"not a BeforeAfterReport list: {exc.error_count()}"
        ) from exc


# --- source loading ---------------------------------------------------------------------------


def git_commit_time(path: Path) -> datetime | None:
    """Last commit time of `path`, or None when git is unavailable or the file is untracked."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return _utc(datetime.fromisoformat(out)) if out else None


@dataclass(frozen=True)
class LoadedSource:
    text: str
    stamp: SourceStamp


class SourceReader:
    """Reads report files under `reports_dir` and stamps each one. `commit_time` is injectable so
    unit tests never shell out to git."""

    def __init__(
        self,
        reports_dir: Path = REPORTS_DIR,
        *,
        commit_time: Callable[[Path], datetime | None] = git_commit_time,
    ) -> None:
        self.reports_dir = reports_dir
        self._commit_time = commit_time

    def load(self, name: str, *, generated_at: datetime | None = None) -> LoadedSource | None:
        """The report `name` (relative to reports_dir) or None when it was never generated."""
        path = self.reports_dir / name
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        return LoadedSource(text=text, stamp=self.stamp(name, generated_at))

    def stamp(self, name: str, generated_at: datetime | None) -> SourceStamp:
        display = f"reports/{name}"
        if generated_at is not None:
            return SourceStamp(display, _utc(generated_at), "generated")
        path = self.reports_dir / name
        committed = self._commit_time(path)
        if committed is not None:
            return SourceStamp(display, committed, "committed")
        return SourceStamp(display, datetime.fromtimestamp(path.stat().st_mtime, UTC), "file mtime")
