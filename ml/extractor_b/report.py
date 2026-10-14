"""Extractor B training report (docs/design/training-loop.md step 13): reads one run directory,
the guard outputs and, when reachable, its MLflow run, and writes `reports/extractor_b.md` with
the selected step and why, the dev F1 / valid-JSON curve, quarantine, skipped steps, resumes, the
loss summary and the export. CPU only, host-side; an unreachable source is a stated gap."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from apx.common.errors import ApxError
from ml.extractor_b.checkpoints import FsCheckpointStore
from ml.extractor_b.config import DEFAULT_CONFIG_PATH, load_config
from ml.extractor_b.evaluator import (
    DEV_OUTPUTS_DIR,
    FinishReason,
    dev_outputs_path,
    parse_prediction,
)
from ml.extractor_b.export import (
    Q4_NAME,
    ExportManifest,
    latest_run_dir,
    load_run_record,
    verify_export,
)
from ml.extractor_b.interfaces import (
    CheckpointMeta,
    EvalResult,
    GuardReport,
    LoopState,
    RunRecord,
    TrainingError,
)
from ml.extractor_b.selection import select_best
from ml.extractor_b.smoke import SMOKE_NAME, SmokeResult

REPORT_PATH: Final[Path] = Path("reports/extractor_b.md")
DEFAULT_MLFLOW_URI: Final[str] = "http://127.0.0.1:5000"
"""The host reaches MLflow through the compose `gateway` (infra/compose.yaml)."""
_LAST_N: Final[int] = 50
_LOOP_RE: Final[re.Pattern[str]] = re.compile(r"(.{8,200}?)\1{3,}", re.S)
"""A unit of 8-200 characters repeated back to back at least 4 times."""


class ReportError(TrainingError):
    """The run directory is incomplete or inconsistent, so no honest report can be written."""


class MlflowUnavailableError(ReportError):
    """MLflow could not be read; the report states the gap instead of guessing."""


# ---------------------------------------------------------------------------------------------
# MLflow (REST, stdlib only: the host venv has no MLflow client).
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MlflowRun:
    status: str
    start_ms: int | None
    end_ms: int | None
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]


Fetch = Callable[[str, float], bytes]


def _http_get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body: bytes = response.read()
        return body


def fetch_mlflow_run(
    uri: str, run_id: str, *, fetch: Fetch = _http_get, timeout: float = 5.0
) -> MlflowRun:
    url = f"{uri.rstrip('/')}/api/2.0/mlflow/runs/get?" + urllib.parse.urlencode({"run_id": run_id})
    try:
        payload = json.loads(fetch(url, timeout))
        run = payload["run"]
        info, data = run["info"], run.get("data", {})
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
        raise MlflowUnavailableError(f"MLflow run {run_id} not readable at {uri}: {exc}") from exc

    def pairs(key: str) -> list[dict[str, Any]]:
        value = data.get(key, [])
        return value if isinstance(value, list) else []

    def ms(key: str) -> int | None:
        value = info.get(key)
        return int(value) if value is not None else None

    return MlflowRun(
        status=str(info.get("status", "UNKNOWN")),
        start_ms=ms("start_time"),
        end_ms=ms("end_time"),
        params={str(p["key"]): str(p["value"]) for p in pairs("params")},
        tags={str(t["key"]): str(t["value"]) for t in pairs("tags")},
        metrics={str(m["key"]): float(m["value"]) for m in pairs("metrics")},
    )


# ---------------------------------------------------------------------------------------------
# Loss and invalid-JSON summaries (pure).
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LossSummary:
    steps: int
    first: float
    last: float
    minimum: float
    minimum_step: int
    mean: float
    mean_last_n: float
    epoch_means: dict[int, float]
    max_grad_norm: float
    non_finite: int


def summarize_loss(
    log_history: Sequence[Mapping[str, Any]], *, last_n: int = _LAST_N
) -> LossSummary:
    """Per-step training loss from HF `trainer_state.json` `log_history` (entries with `loss`)."""
    entries = sorted(
        (e for e in log_history if "loss" in e and "step" in e), key=lambda e: int(e["step"])
    )
    if not entries:
        raise ReportError("trainer_state.json has no per-step loss entries")
    losses = [float(e["loss"]) for e in entries]
    finite = [(int(e["step"]), x) for e, x in zip(entries, losses, strict=True) if math.isfinite(x)]
    if not finite:
        raise ReportError("every logged training loss is non-finite")
    values = [x for _, x in finite]
    min_step, min_loss = min(finite, key=lambda p: p[1])
    by_epoch: dict[int, list[float]] = {}
    for e, x in zip(entries, losses, strict=True):
        if math.isfinite(x):
            epoch = max(0, math.ceil(float(e.get("epoch", 0.0))) - 1)
            by_epoch.setdefault(epoch + 1, []).append(x)
    grads = [float(e["grad_norm"]) for e in entries if "grad_norm" in e]
    tail = values[-last_n:]
    return LossSummary(
        steps=len(entries),
        first=values[0],
        last=values[-1],
        minimum=min_loss,
        minimum_step=min_step,
        mean=sum(values) / len(values),
        mean_last_n=sum(tail) / len(tail),
        epoch_means={k: sum(v) / len(v) for k, v in sorted(by_epoch.items())},
        max_grad_norm=max((g for g in grads if math.isfinite(g)), default=math.nan),
        non_finite=len(entries) - len(finite),
    )


class InvalidKind(StrEnum):
    LOOPING = "looping"
    TRUNCATED = "truncated"
    MALFORMED = "malformed"


def _looping(text: str) -> bool:
    return _LOOP_RE.search(text[-2000:]) is not None


def _unterminated(text: str) -> bool:
    """True when `text` is JSON cut off mid-structure: every bracket seen closes correctly and
    nothing follows the top-level value, but the text ends inside a string or an open bracket."""
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and (not stack or stack.pop() != ch or not stack):
            return False  # a mismatched closer, or the top-level value closed: not a cut-off
    return in_string or bool(stack)


def classify_invalid(text: str, finish_reason: FinishReason | None = None) -> InvalidKind | None:
    """None for a valid prediction (same rule as the dev metric: a JSON object). Otherwise
    looping (a repeated unit, the usual cause of running out of tokens) first. With a recorded
    `finish_reason` the rest is decided by it: out of budget is truncated, a stop token is
    malformed (the model ended on its own with broken JSON). Without one (older files), JSON that
    is fine up to an abrupt end is truncated and anything else malformed."""
    if parse_prediction(text) is not None:
        return None
    if _looping(text):
        return InvalidKind.LOOPING
    if finish_reason is FinishReason.MAX_NEW_TOKENS:
        return InvalidKind.TRUNCATED
    if finish_reason is FinishReason.EOS:
        return InvalidKind.MALFORMED
    stripped = text.strip()
    if stripped.startswith("{") and _unterminated(stripped):
        return InvalidKind.TRUNCATED
    return InvalidKind.MALFORMED


@dataclass(frozen=True, slots=True)
class DevOutputs:
    step: int
    path: Path
    rows: int
    invalid: dict[InvalidKind, list[str]]


def read_dev_outputs(run_dir: Path, step: int) -> DevOutputs | None:
    """Classify a persisted step's rows (`evaluator.DevOutputRow`: `sample_id`, `text`, and
    `finish_reason` when recorded; the other fields are informational)."""
    path = dev_outputs_path(run_dir, step)
    if not path.is_file():
        return None
    invalid: dict[InvalidKind, list[str]] = {}
    rows = 0
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            sample_id, text = str(row["sample_id"]), str(row["text"])
            raw_reason = row.get("finish_reason")
            reason = None if raw_reason is None else FinishReason(raw_reason)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ReportError(f"{path}:{n} is not a {{sample_id, text}} row: {exc}") from exc
        rows += 1
        kind = classify_invalid(text, reason)
        if kind is not None:
            invalid.setdefault(kind, []).append(sample_id)
    return DevOutputs(step=step, path=path, rows=rows, invalid=invalid)


# ---------------------------------------------------------------------------------------------
# Evidence collection.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointRow:
    step: int
    verified: bool
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class Evidence:
    run_dir: Path
    record: RunRecord
    best_step: int
    reselected_step: int | None
    tie_epsilon: float
    checkpoints: list[CheckpointRow]
    state_step: int
    state: LoopState
    loss: LossSummary
    guard: GuardReport | None
    quarantine_rows: list[dict[str, Any]]
    spikes: list[dict[str, Any]]
    dev_rows: int | None
    dev_outputs: DevOutputs | None
    mlflow: MlflowRun | None
    gaps: list[str] = field(default_factory=list)
    export: ExportManifest | None = None
    smoke: SmokeResult | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def collect(
    run_dir: Path,
    data_dir: Path,
    *,
    tie_epsilon: float,
    keep_last: int,
    mlflow: Callable[[str], MlflowRun] | None,
) -> Evidence:
    record = load_run_record(run_dir)
    store = FsCheckpointStore(run_dir, fingerprint=record.fingerprint, keep_last=keep_last)
    best = store.best_step()
    if best is None:
        raise ReportError(f"{run_dir / 'best.json'} is missing")
    dirs = sorted(p for p in store.root.glob("checkpoint-*") if p.is_dir())
    rows: list[CheckpointRow] = []
    metas: dict[int, CheckpointMeta] = {}
    for path in dirs:
        meta = store.verify(path)
        step = int(path.name.split("-", 1)[1].split(".", 1)[0])
        if meta is not None:
            metas[step] = meta
        rows.append(CheckpointRow(step, meta is not None, meta.created_at if meta else None))
    if not metas:
        raise ReportError(f"no verified checkpoint under {store.root}")
    state_step = max(metas)
    try:
        state = LoopState.model_validate_json(
            (store.path_for(state_step) / "loop_state.json").read_bytes()
        )
    except (OSError, ValidationError) as exc:
        raise ReportError(f"loop_state.json of step {state_step} unreadable: {exc}") from exc
    trainer_state = _read_json(store.path_for(state_step) / "trainer_state.json")
    loss = summarize_loss(trainer_state.get("log_history", []))

    gaps: list[str] = []
    guard_path = data_dir / "guard_report.json"
    guard: GuardReport | None = None
    if guard_path.is_file():
        guard = GuardReport.model_validate_json(guard_path.read_bytes())
        if guard.quarantine_sha256 != record.fingerprint.quarantine_sha256:
            gaps.append(
                f"{guard_path} is from another data prep (quarantine sha256 "
                f"{guard.quarantine_sha256[:12]} vs run "
                f"{record.fingerprint.quarantine_sha256[:12]})"
            )
    else:
        gaps.append(f"{guard_path} missing: quarantine counts come from the fingerprint only")
    dev_path = data_dir / "dev.jsonl"
    dev_rows = len(_read_jsonl(dev_path)) if dev_path.is_file() else None

    dev_outputs = read_dev_outputs(run_dir, best)
    if dev_outputs is None:
        gaps.append(
            f"per-row dev generations were not persisted for step {best} "
            f"(`<run>/{DEV_OUTPUTS_DIR}/step-{best:06d}.jsonl` is missing): the run predates "
            "issue #53 or its evaluator had no run_dir, so the invalid rows cannot be split "
            "into looping / truncated / malformed"
        )

    mlflow_run: MlflowRun | None = None
    if mlflow is None:
        gaps.append("MLflow not queried (--no-mlflow)")
    else:
        try:
            mlflow_run = mlflow(record.mlflow_run_id)
        except MlflowUnavailableError as exc:
            gaps.append(f"MLflow unavailable, so run status and logged metrics are absent: {exc}")

    export: ExportManifest | None = None
    export_dir = run_dir / "export"
    if export_dir.exists():
        export = verify_export(export_dir)
        if export is None:
            gaps.append(f"{export_dir} exists but does not verify against its export.json")
    smoke: SmokeResult | None = None
    smoke_path = export_dir / SMOKE_NAME
    if export is not None and smoke_path.is_file():
        try:
            smoke = SmokeResult.model_validate_json(smoke_path.read_bytes())
        except ValidationError as exc:
            raise ReportError(f"{smoke_path} is invalid: {exc}") from exc
        if smoke.q4_sha256 != export.artifact(Q4_NAME).sha256:
            gaps.append(f"{smoke_path} was run against another Q4_K_M; re-run `make smoke-b`")
            smoke = None

    return Evidence(
        run_dir=run_dir,
        record=record,
        best_step=best,
        reselected_step=select_best(state.evals, tie_epsilon=tie_epsilon),
        tie_epsilon=tie_epsilon,
        checkpoints=rows,
        state_step=state_step,
        state=state,
        loss=loss,
        guard=guard,
        quarantine_rows=_read_jsonl(data_dir / "quarantine.jsonl"),
        spikes=_read_jsonl(run_dir / "spikes.jsonl"),
        dev_rows=dev_rows,
        dev_outputs=dev_outputs,
        mlflow=mlflow_run,
        gaps=gaps,
        export=export,
        smoke=smoke,
    )


# ---------------------------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------------------------


def selection_reason(evals: Sequence[EvalResult], best: int, tie_epsilon: float) -> str:
    generative = [e for e in evals if e.kind == "generative" and e.dev_f1 is not None]
    if not generative:
        return "no generative eval exists"
    top = max(e.dev_f1 for e in generative if e.dev_f1 is not None)
    tied = sorted(
        e.step for e in generative if e.dev_f1 is not None and e.dev_f1 >= top - tie_epsilon
    )
    if len(tied) == 1:
        return f"highest dev F1 ({top:.2f})"
    losses = {e.step: e.dev_loss for e in evals if e.dev_loss is not None}
    with_loss = [s for s in tied if s in losses]
    rule = (
        "lower dev loss wins, then the earlier step"
        if with_loss
        else "none of them has a dev loss at its own step, so the earlier step wins"
    )
    joined = ", ".join(str(s) for s in tied)
    return f"steps {joined} tie within {tie_epsilon} F1 points of the top ({top:.2f}); {rule}"


def _fmt(value: float | None, spec: str) -> str:
    return "" if value is None or not math.isfinite(value) else format(value, spec)


def _pct(rate: float | None) -> str:
    return "" if rate is None or not math.isfinite(rate) else f"{rate * 100:.1f}%"


def _ts(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M UTC")


def render(ev: Evidence, *, generated_at: datetime) -> str:
    fp = ev.record.fingerprint
    best_eval = next(
        (e for e in ev.state.evals if e.step == ev.best_step and e.kind == "generative"), None
    )
    best_f1 = best_eval.dev_f1 if best_eval else None
    best_valid = best_eval.valid_json_rate if best_eval else None
    out: list[str] = [
        "# Extractor B training report",
        "",
        f"Run `{ev.record.run_id}` (MLflow `{ev.record.mlflow_run_id}`). Generated by "
        f"`python -m ml.extractor_b.report` on {generated_at:%Y-%m-%d %H:%M} UTC from the run "
        "directory, the guard outputs and MLflow. Dev set: "
        f"{ev.dev_rows if ev.dev_rows is not None else 'unknown'} UAE documents.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Selected step (`best.json`) | **{ev.best_step}** |",
        f"| Why | {selection_reason(ev.state.evals, ev.best_step, ev.tie_epsilon)} |",
        f"| Dev micro field F1 at the selected step | {_fmt(best_f1, '.2f')} |",
        f"| Valid JSON at the selected step | {_pct(best_valid)} |",
        f"| `select_best` re-run on the logged evals | {ev.reselected_step} "
        f"({'matches' if ev.reselected_step == ev.best_step else 'DIFFERS from'} best.json) |",
        f"| Steps trained | {ev.state_step} (last verified checkpoint) |",
        f"| Resumes | {ev.state.resume_count} |",
        f"| Skipped optimizer steps (spike guard) | {ev.state.spike.total_skipped_steps} |",
        "",
        "## Fingerprint",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| config_sha256 | `{fp.config_sha256}` |",
        f"| manifest_sha256 | `{fp.manifest_sha256}` |",
        f"| quarantine_sha256 | `{fp.quarantine_sha256}` |",
        f"| base_revision | `{fp.base_revision}` |",
        f"| image_digest | `{fp.image_digest}` |",
        f"| rung | {fp.rung.value} |",
        f"| git_sha (run start; recorded, not compared) | `{fp.git_sha}` |",
        "",
        "## Dev curve",
        "",
        "Generative evals are greedy decoding scored by `evals/scorers` (micro field F1 over the "
        "14 header fields, in points). Dev-loss evals are teacher-forced.",
        "",
        "| Step | Kind | Dev F1 | Valid JSON | Dev loss |",
        "|---:|---|---:|---:|---:|",
    ]
    for e in ev.state.evals:
        mark = " (selected)" if e.step == ev.best_step and e.kind == "generative" else ""
        out.append(
            f"| {e.step}{mark} | {e.kind} | {_fmt(e.dev_f1, '.2f')} | "
            f"{_pct(e.valid_json_rate)} | {_fmt(e.dev_loss, '.5f')} |"
        )
    out += ["", "## Checkpoints", "", "| Step | Verified | Saved |", "|---:|---|---|"]
    for c in ev.checkpoints:
        saved = c.created_at.strftime("%Y-%m-%d %H:%M UTC") if c.created_at else ""
        out.append(f"| {c.step} | {'yes' if c.verified else 'NO'} | {saved} |")

    loss = ev.loss
    out += [
        "",
        "## Training loss",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Logged steps | {loss.steps} |",
        f"| First / last step loss | {loss.first:.4f} / {loss.last:.4f} |",
        f"| Minimum (step {loss.minimum_step}) | {loss.minimum:.5f} |",
        f"| Mean, all steps | {loss.mean:.5f} |",
        f"| Mean, last {min(_LAST_N, loss.steps)} steps | {loss.mean_last_n:.5f} |",
    ]
    out += [f"| Mean, epoch {k} | {v:.5f} |" for k, v in loss.epoch_means.items()]
    out += [
        f"| Max grad norm | {_fmt(loss.max_grad_norm, '.3f')} |",
        f"| Non-finite losses | {loss.non_finite} |",
    ]

    out += ["", "## Quarantine and skipped steps", ""]
    if ev.guard is not None:
        out.append(
            f"Guards kept {ev.guard.kept} rows and quarantined {ev.guard.quarantined} "
            f"(`quarantine.jsonl`: {len(ev.quarantine_rows)} rows)."
        )
        if ev.guard.by_source_reason:
            out += ["", "| Source | Reason | Rows |", "|---|---|---:|"]
            for key, n in sorted(ev.guard.by_source_reason.items()):
                source, _, reason = key.partition(":")
                out.append(f"| {source} | {reason} | {n} |")
    elif ev.quarantine_rows:
        counts = Counter((r.get("source"), r.get("reason")) for r in ev.quarantine_rows)
        out += ["| Source | Reason | Rows |", "|---|---|---:|"]
        out += [f"| {s} | {r} | {n} |" for (s, r), n in sorted(counts.items())]
    out.append("")
    if ev.spikes:
        actions = Counter(str(s.get("action")) for s in ev.spikes)
        out.append(
            f"The spike guard logged {len(ev.spikes)} events ("
            + ", ".join(f"{a}: {n}" for a, n in sorted(actions.items()))
            + f"); {ev.state.spike.total_skipped_steps} optimizer steps were skipped in total."
        )
    else:
        out.append(
            f"No spike-guard events (`spikes.jsonl` absent or empty); "
            f"{ev.state.spike.total_skipped_steps} optimizer steps skipped."
        )

    out += ["", "## Invalid JSON on dev", ""]
    invalid_rate = None if best_valid is None else 1 - best_valid
    if invalid_rate is not None and ev.dev_rows:
        out.append(
            f"At step {ev.best_step}, {invalid_rate * 100:.1f}% of dev outputs "
            f"(about {round(invalid_rate * ev.dev_rows)} of {ev.dev_rows}) were not a JSON object."
        )
    if ev.dev_outputs is not None:
        out += ["", "| Failure kind | Rows | Sample ids |", "|---|---:|---|"]
        for kind in InvalidKind:
            ids = ev.dev_outputs.invalid.get(kind, [])
            out.append(f"| {kind.value} | {len(ids)} | {', '.join(ids[:5])} |")
    else:
        out.append(
            "The per-row generations were not persisted, so the failure kinds (looping / "
            "truncated / malformed) are unknown for this run; see Gaps."
        )

    out += ["", "## MLflow", ""]
    if ev.mlflow is not None:
        m = ev.mlflow
        out += [
            f"Status {m.status}, {_ts(m.start_ms)} to {_ts(m.end_ms)}.",
            "",
            "| Metric (latest) | Value |",
            "|---|---:|",
        ]
        out += [f"| {k} | {v:.6g} |" for k, v in sorted(m.metrics.items())]
    else:
        out.append("Not available; see Gaps.")

    out += ["", "## Export (ADR-0009 §1.7)", ""]
    if ev.export is not None:
        x = ev.export
        out += [
            f"From step {x.step} (adapter sha256 `{x.adapter_sha256[:16]}…`), base "
            f"`{x.base_path}` @ `{x.base_revision[:12]}`, llama.cpp `{x.llama_cpp_commit[:12]}`. "
            f"The mmproj is tensor-equal to `{x.mmproj_upstream_path}` "
            f"({x.mmproj_tensors_compared} tensors).",
            "",
            "| Artifact | Size | sha256 |",
            "|---|---:|---|",
        ]
        out += [f"| `{a.name}` | {a.size_bytes / 1e9:.2f} GB | `{a.sha256}` |" for a in x.artifacts]
    else:
        out.append("No verified export yet (`make export-b`).")
    if ev.smoke is not None:
        s = ev.smoke
        out += [
            "",
            f"llama-server smoke (d), {s.created_at:%Y-%m-%d %H:%M} UTC: {s.valid}/{s.documents} "
            f"UAE dev documents gave valid JSON under schema-constrained decoding through "
            f"llama-swap model `{s.model}` ({'PASS' if s.passed else 'FAIL'}).",
            "",
            "| Sample | Pages | Valid JSON | Finish | Latency |",
            "|---|---:|---|---|---:|",
        ]
        out += [
            f"| {d.sample_id} | {d.pages} | {'yes' if d.valid_json else 'NO'} | "
            f"{d.finish_reason} | {d.latency_ms / 1000:.1f} s |"
            for d in s.docs
        ]
    elif ev.export is not None:
        out += ["", "llama-server smoke (d): not run yet (`make smoke-b`)."]

    out += ["", "## Gaps", ""]
    out += [f"- {g}" for g in ev.gaps] if ev.gaps else ["None."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.report", description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="default: the newest models/extractor_b/b-*")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI),
        help=f"default: $MLFLOW_TRACKING_URI or {DEFAULT_MLFLOW_URI}",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        run_dir = args.run_dir or latest_run_dir(cfg.runs_dir)
        uri: str = args.mlflow_uri

        def mlflow(run_id: str) -> MlflowRun:
            return fetch_mlflow_run(uri, run_id)

        evidence = collect(
            run_dir,
            cfg.data_dir,
            tie_epsilon=cfg.cadence.f1_tie_epsilon,
            keep_last=cfg.retention.keep_last_verified,
            mlflow=None if args.no_mlflow else mlflow,
        )
    except ApxError as exc:
        print(f"report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render(evidence, generated_at=datetime.now(UTC)), encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out} (run {evidence.record.run_id}, {len(evidence.gaps)} gap(s))")
    for gap in evidence.gaps:
        print(f"  gap: {gap}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
