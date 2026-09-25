"""Smoke-tests every runtime model pinned in infra/models.lock.json (PRD §4, §10 M0.5): the
extract/arbitrate LLMs through llama-swap's OpenAI-compatible API, the guard/parse models
directly through transformers, and the two embedding models through sentence-transformers
(both ship in that format, not a plain `AutoModel` checkpoint). For each it loads the model,
samples peak GPU memory every 200 ms, times one request, and checks the PRD §4 extraction-
call rules (JSON schema output, one tool call, logprobs, thinking disabled) or embedding
dimensions. Writes reports/smoke.md and exits nonzero if any runtime model fails; an
unreachable model or a missing optional dependency (torch/transformers/sentence-transformers
are not pinned in pyproject.toml -- they belong to a later milestone's `ml/` work, and are
installed instead in infra/docker/toolbox/Dockerfile's image) is reported as a FAIL with a
reason, never skipped. Intended to run inside that `toolbox` container (`make smoke`), which
has a real GPU and those packages; running it directly on the host is also safe, it just
reports every transformers-backed model as FAIL with "not installed".
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "infra" / "models.lock.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REPORT = REPO_ROOT / "reports" / "smoke.md"
DEFAULT_BASE_URL = os.environ.get("LLAMA_SWAP_BASE_URL", "http://localhost:8080")

VRAM_BUDGET_GB = 11.5
SAMPLE_INTERVAL_S = 0.2

# infra/models.lock.json `intent` -> llama-swap served model id (infra/llama-swap.yaml).
LLAMA_SWAP_SERVED_NAME = {
    "extractor_a": "extractor-a",
    "extractor_b_base": "extractor-b",
    "arbiter": "arbiter",
}
# infra/models.lock.json `intent` -> apx.common.types.Stage value, for the per-stage VRAM
# budget table. extractor_a and extractor_b_base share the "extract" llama-swap group
# (swap: false, exclusive: true, PRD §4): llama-swap keeps both resident once either is
# called, so the max of their two independently-sampled peaks approximates the group's
# combined peak without needing a dedicated "call both at once" harness.
MODEL_STAGE = {
    "parse": "parse",
    "injection_guard_base": "scan",
    "extractor_a": "extract",
    "extractor_b_base": "extract",
    "arbiter": "arbitrate",
    "text_embeddings": "fraud",
    "page_similarity": "fraud_page",
}

INVOICE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
    },
    "required": ["invoice_number", "total"],
    "additionalProperties": False,
}

TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_po",
        "description": "Look up a purchase order by number.",
        "parameters": {
            "type": "object",
            "properties": {"po_number": {"type": "string"}},
            "required": ["po_number"],
        },
    },
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ModelResult:
    intent: str
    served_name: str
    stage: str
    reachable: bool
    checks: tuple[CheckResult, ...]
    peak_vram_gb: float | None
    latency_ms: float | None

    @property
    def passed(self) -> bool:
        return self.reachable and all(c.passed for c in self.checks)


# --- VRAM sampling (nvidia-smi, sampled every 200 ms in a background thread) -------------


def parse_nvidia_smi_memory_used(output: str) -> list[float]:
    """Parses `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` stdout
    (one MiB value per line, one line per GPU) into GB values."""
    values: list[float] = []
    for raw_line in output.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values.append(float(line) / 1024.0)
    return values


def _run_nvidia_smi() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=5,
        check=True,
    )
    return result.stdout


def sample_vram_gb(runner: Callable[[], str] = _run_nvidia_smi) -> float | None:
    """One VRAM sample in GB (max across GPUs), or None if nvidia-smi is unavailable, not on
    PATH, or returns something unparseable -- never raises."""
    try:
        output = runner()
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        values = parse_nvidia_smi_memory_used(output)
    except ValueError:
        return None
    return max(values) if values else None


class VramSampler:
    """Background-thread sampler: while the `with` block is open, polls `sample_fn` every
    `interval_s` seconds and tracks the maximum value seen as `peak_gb`."""

    def __init__(
        self,
        interval_s: float = SAMPLE_INTERVAL_S,
        sample_fn: Callable[[], float | None] = sample_vram_gb,
    ) -> None:
        self._interval_s = interval_s
        self._sample_fn = sample_fn
        self._peak_gb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> VramSampler:
        self._stop.clear()
        self._poll_once()  # at least one sample even for a near-instant block
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._interval_s + 2)

    def _poll_once(self) -> None:
        value = self._sample_fn()
        if value is not None:
            self._peak_gb = value if self._peak_gb is None else max(self._peak_gb, value)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._poll_once()

    @property
    def peak_gb(self) -> float | None:
        return self._peak_gb


# --- synthetic invoice image (PIL, in-script -- no fixture files) -----------------------


def build_invoice_image() -> Image.Image:
    """Renders a tiny synthetic invoice as a PIL image -- no fixture files. Shared by the
    llama-swap VLM checks (as a data URL) and the parse/page-similarity transformers checks
    (as a PIL image), so every VLM sees the same input."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "INVOICE", fill="black")
    draw.text((10, 40), "Invoice No: INV-0001", fill="black")
    draw.text((10, 60), "Total: AED 105.00", fill="black")
    return image


def build_invoice_png_data_url(image: Image.Image | None = None) -> str:
    """Encodes `build_invoice_image()` (or a given image) as a base64 PNG data URL, matching
    apx.llm.interfaces.ImagePart's `data:image/(png|jpeg);base64,...` pattern."""
    image = image if image is not None else build_invoice_image()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --- llama-swap checks (pure evaluation of an already-parsed response body) --------------


def evaluate_structured_response(data: dict[str, Any]) -> list[CheckResult]:
    """Checks a /v1/chat/completions body against the PRD §4 extraction-call rules: JSON-
    schema-shaped content, logprobs present, and thinking disabled (no <think> leakage)."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        reason = f"malformed response: {exc!r}"
        return [
            CheckResult("json_schema_output", False, reason),
            CheckResult("logprobs", False, reason),
            CheckResult("thinking_disabled", False, reason),
        ]

    checks = []
    try:
        parsed = json.loads(content)
        schema_ok = isinstance(parsed, dict) and "invoice_number" in parsed and "total" in parsed
        checks.append(
            CheckResult(
                "json_schema_output",
                schema_ok,
                "" if schema_ok else f"unexpected shape: {content!r}",
            )
        )
    except (json.JSONDecodeError, TypeError) as exc:
        checks.append(CheckResult("json_schema_output", False, f"not valid JSON: {exc}"))

    think_leaked = isinstance(content, str) and "<think>" in content
    checks.append(
        CheckResult(
            "thinking_disabled",
            not think_leaked,
            "found <think> in content" if think_leaked else "",
        )
    )

    logprobs = choice.get("logprobs")
    has_logprobs = bool(logprobs and logprobs.get("content"))
    checks.append(
        CheckResult(
            "logprobs", has_logprobs, "" if has_logprobs else "no logprobs.content in response"
        )
    )
    return checks


def evaluate_tool_call_response(data: dict[str, Any]) -> CheckResult:
    """Checks that a tool-calling request produced the expected `lookup_po` tool call."""
    try:
        tool_calls = data["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError) as exc:
        return CheckResult("tool_call", False, f"malformed response: {exc!r}")
    ok = bool(tool_calls) and tool_calls[0].get("function", {}).get("name") == "lookup_po"
    return CheckResult("tool_call", ok, "" if ok else f"no matching tool call: {tool_calls!r}")


def run_llama_swap_model(
    client: httpx.Client,
    base_url: str,
    *,
    intent: str,
    served_name: str,
    stage: str,
    include_tool_call: bool,
    invoice_image_data_url: str,
) -> ModelResult:
    checks: list[CheckResult] = []
    latency_ms: float | None = None
    reachable = True

    structured_payload = {
        "model": served_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract invoice_number and total as JSON."},
                    {"type": "image_url", "image_url": {"url": invoice_image_data_url}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "invoice", "schema": INVOICE_JSON_SCHEMA, "strict": True},
        },
        "temperature": 0.0,
        "seed": 42,
        "logprobs": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    with VramSampler() as sampler:
        start = time.monotonic()
        try:
            # 320s, not httpx's default: comfortably above infra/llama-swap.yaml's own
            # healthCheckTimeout (300s) -- a cold model swap loads a ~7 GB GGUF + mmproj from
            # `models/`, a bind mount over NTFS (ADR-0006's addendum), which measured 60-120s+
            # run to run.
            response = client.post(
                f"{base_url}/v1/chat/completions", json=structured_payload, timeout=320.0
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            reachable = False
            checks.append(CheckResult("reachable", False, f"{type(exc).__name__}: {exc}"))
        else:
            latency_ms = (time.monotonic() - start) * 1000
            checks.append(CheckResult("reachable", True))
            checks.extend(evaluate_structured_response(response.json()))

        if reachable and include_tool_call:
            tool_payload = {
                "model": served_name,
                "messages": [{"role": "user", "content": "Look up PO number PO-42."}],
                "tools": [TOOL_SPEC],
                "temperature": 0.2,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            try:
                tool_response = client.post(
                    f"{base_url}/v1/chat/completions", json=tool_payload, timeout=320.0
                )
                tool_response.raise_for_status()
                checks.append(evaluate_tool_call_response(tool_response.json()))
            except httpx.HTTPError as exc:
                checks.append(CheckResult("tool_call", False, f"{type(exc).__name__}: {exc}"))

    _unload_llama_swap_model(client, base_url, served_name)
    return ModelResult(
        intent=intent,
        served_name=served_name,
        stage=stage,
        reachable=reachable,
        checks=tuple(checks),
        peak_vram_gb=sampler.peak_gb,
        latency_ms=latency_ms,
    )


def _unload_llama_swap_model(client: httpx.Client, base_url: str, served_name: str) -> None:
    """Explicitly unloads `served_name` after its check (llama-swap's admin API, `POST
    /api/models/unload/{id}`) instead of waiting out infra/llama-swap.yaml's 600s `ttl`. Two
    models in the same exclusive group (`extractor-a`/`extractor-b`) get reloaded together, so
    unloading one costs the next check a redundant group reload -- acceptable here, since a
    smoke run isn't latency-sensitive. Without this, a later stage's VRAM sample would include
    this model's memory too (PyTorch's cache isn't the only thing that lingers -- llama-swap's
    own subprocess does too), which is exactly the PRD C2 per-stage budget this script proves.
    Best-effort: a failed unload just means the next check's peak might run a bit high, not a
    reason to fail this one."""
    with contextlib.suppress(httpx.HTTPError):
        client.post(f"{base_url}/api/models/unload/{served_name}", timeout=30.0)


# --- transformers checks (guard/parse/embedding models, loaded directly) ----------------


def _empty_cuda_cache() -> None:
    """Returns PyTorch's cached-but-unused GPU memory to the driver once a check's model has
    gone out of scope. Without this, nvidia-smi keeps counting a freed model against this
    process for the rest of the run (PyTorch's caching allocator never gives memory back on
    its own) -- every later check's peak would include every earlier one's model too, and
    this toolbox process would keep competing with llama-swap for the one shared GPU (PRD C2)
    long after each check is actually done with it."""
    try:
        import torch
    except ImportError:
        return
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_transformers_load_only_model(*, intent: str, stage: str, repo_dir: Path) -> ModelResult:
    """Load + one forward pass for the injection guard base checkpoint: a plain transformers
    encoder (`AutoModel`/`AutoTokenizer`), `local_files_only=True` always so this makes no
    network call."""
    with VramSampler() as sampler:
        start = time.monotonic()
        reachable, checks = _load_and_probe_generic(repo_dir)
        latency_ms = (time.monotonic() - start) * 1000
    _empty_cuda_cache()
    return ModelResult(
        intent=intent,
        served_name=intent,
        stage=stage,
        reachable=reachable,
        checks=tuple(checks),
        peak_vram_gb=sampler.peak_gb,
        latency_ms=latency_ms,
    )


def _load_and_probe_generic(repo_dir: Path) -> tuple[bool, list[CheckResult]]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        return False, [CheckResult("load", False, f"transformers not installed: {exc}")]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(repo_dir), local_files_only=True, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            str(repo_dir), local_files_only=True, trust_remote_code=True, dtype="auto"
        )
        with torch.no_grad():
            inputs = tokenizer("Sample invoice text.", return_tensors="pt")
            model(**inputs)
    except (OSError, ValueError) as exc:
        return False, [CheckResult("load", False, f"could not load model from {repo_dir}: {exc}")]
    return True, [CheckResult("load", True)]


def run_transformers_vlm_model(
    *, intent: str, stage: str, repo_dir: Path, invoice_image: Image.Image
) -> ModelResult:
    """Loads the parse VLM (PaddleOCR-VL) and runs one short OCR generation on the same
    synthetic invoice used for the llama-swap VLM checks. No `trust_remote_code`: transformers
    now has `paddleocr_vl` as a native model type, and its own implementation is what actually
    gets used regardless -- passing the checkpoint's local `configuration_paddleocr_vl.py`/
    `modeling_paddleocr_vl.py` as well produces a config/model class mismatch (`PaddleOCRVLConfig
    object has no attribute 'text_config'` or `...'rope_parameters'`, depending on transformers
    version). `repo_dir` is always a local path, never a hub id, so this makes no network call
    either way."""
    with VramSampler() as sampler:
        start = time.monotonic()
        reachable, checks = _load_and_probe_vlm(repo_dir, invoice_image)
        latency_ms = (time.monotonic() - start) * 1000
    _empty_cuda_cache()
    return ModelResult(
        intent=intent,
        served_name=intent,
        stage=stage,
        reachable=reachable,
        checks=tuple(checks),
        peak_vram_gb=sampler.peak_gb,
        latency_ms=latency_ms,
    )


def _load_and_probe_vlm(
    repo_dir: Path, invoice_image: Image.Image
) -> tuple[bool, list[CheckResult]]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        return False, [CheckResult("load", False, f"transformers not installed: {exc}")]

    try:
        processor = AutoProcessor.from_pretrained(str(repo_dir), local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(repo_dir), local_files_only=True, dtype="auto"
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
    except (OSError, ValueError, AttributeError) as exc:
        return False, [CheckResult("load", False, f"could not load model from {repo_dir}: {exc}")]

    checks = [CheckResult("load", True)]
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": invoice_image},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        text = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        ocr_ok = bool(text.strip())
        checks.append(CheckResult("ocr_output", ocr_ok, "" if ocr_ok else "empty OCR output"))
    except (RuntimeError, ValueError, KeyError, IndexError) as exc:
        checks.append(CheckResult("ocr_output", False, f"{type(exc).__name__}: {exc}"))

    return True, checks


def run_sentence_transformer_model(
    *, intent: str, stage: str, repo_dir: Path, inputs: Sequence[str | Image.Image]
) -> ModelResult:
    """Loads a sentence-transformers-formatted embedding model (Qwen3-Embedding for
    `text_embeddings`, Qwen3-VL-Embedding for `page_similarity` -- both ship `modules.json` +
    `1_Pooling/`, i.e. the SentenceTransformer format, not a plain `AutoModel` checkpoint) and
    checks that `.encode()` returns vectors of the model's own declared dimension, never a
    hardcoded number. `local_files_only=True` always, so a nonexistent `repo_dir` fails
    locally instead of falling back to a Hub lookup."""
    with VramSampler() as sampler:
        start = time.monotonic()
        reachable, checks = _load_and_probe_sentence_transformer(repo_dir, inputs)
        latency_ms = (time.monotonic() - start) * 1000
    _empty_cuda_cache()
    return ModelResult(
        intent=intent,
        served_name=intent,
        stage=stage,
        reachable=reachable,
        checks=tuple(checks),
        peak_vram_gb=sampler.peak_gb,
        latency_ms=latency_ms,
    )


def _load_and_probe_sentence_transformer(
    repo_dir: Path, inputs: Sequence[str | Image.Image]
) -> tuple[bool, list[CheckResult]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        return False, [CheckResult("load", False, f"sentence-transformers not installed: {exc}")]

    try:
        model = SentenceTransformer(
            str(repo_dir),
            local_files_only=True,
            trust_remote_code=True,
            model_kwargs={"dtype": "auto"},
        )
    except (OSError, ValueError) as exc:
        return False, [CheckResult("load", False, f"could not load model from {repo_dir}: {exc}")]

    checks = [CheckResult("load", True)]
    try:
        embeddings = model.encode(list(inputs))
        # sentence-transformers renamed this method (`get_embedding_dimension`); fall back to
        # the older name so this keeps working on any >=3.3 install, not just the newest.
        get_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
        expected_dim = get_dim()
        actual_dim = int(embeddings.shape[-1])
        dims_ok = expected_dim is not None and actual_dim == expected_dim
        checks.append(
            CheckResult(
                "embeddings",
                dims_ok,
                "" if dims_ok else f"got dim {actual_dim}, expected {expected_dim}",
            )
        )
    except (RuntimeError, ValueError) as exc:
        checks.append(CheckResult("embeddings", False, f"{type(exc).__name__}: {exc}"))

    return True, checks


# --- report -------------------------------------------------------------------------------


def render_report(
    results: Sequence[ModelResult],
    budget_gb: float = VRAM_BUDGET_GB,
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now(UTC).isoformat()
    lines = [
        "# Smoke test report (PRD §10 M0.5)",
        "",
        f"Generated: {generated_at}",
        f"VRAM budget per stage: {budget_gb} GB (PRD C2)",
        "",
        "| Model | Stage | Reachable | Peak VRAM (GB) | Latency (ms) | Checks | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        checks_str = (
            "; ".join(
                f"{c.name}={'PASS' if c.passed else 'FAIL'}"
                + (f" ({c.detail})" if not c.passed and c.detail else "")
                for c in r.checks
            )
            or "n/a"
        )
        vram_str = f"{r.peak_vram_gb:.2f}" if r.peak_vram_gb is not None else "n/a"
        latency_str = f"{r.latency_ms:.0f}" if r.latency_ms is not None else "n/a"
        lines.append(
            f"| {r.intent} | {r.stage} | {'yes' if r.reachable else 'no'} | {vram_str} | "
            f"{latency_str} | {checks_str} | {'PASS' if r.passed else 'FAIL'} |"
        )

    lines += ["", "## Per-stage VRAM budget", ""]
    stage_peaks: dict[str, float] = {}
    for r in results:
        if r.peak_vram_gb is not None:
            stage_peaks[r.stage] = max(stage_peaks.get(r.stage, 0.0), r.peak_vram_gb)
    if stage_peaks:
        lines += ["| Stage | Peak VRAM (GB) | Within budget |", "|---|---|---|"]
        for stage, peak in sorted(stage_peaks.items()):
            within = peak <= budget_gb
            verdict = "yes" if within else "no -- switch to sequential groups (PRD §4)"
            lines.append(f"| {stage} | {peak:.2f} | {verdict} |")
    else:
        lines.append("No VRAM samples collected (nvidia-smi unavailable or no model reachable).")
    lines.append("")
    return "\n".join(lines)


# --- orchestration ------------------------------------------------------------------------


def load_runtime_models(lock_path: Path) -> list[dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return [m for m in lock["models"] if m.get("runtime", False)]


def run_all(lock_path: Path, models_dir: Path, base_url: str) -> list[ModelResult]:
    invoice_image = build_invoice_image()
    invoice_image_data_url = build_invoice_png_data_url(invoice_image)
    results: list[ModelResult] = []
    with httpx.Client() as client:
        for model in load_runtime_models(lock_path):
            intent = model["intent"]
            stage = MODEL_STAGE.get(intent, model.get("stage", intent))
            repo_dir = models_dir / model["repo_id"]
            if intent in LLAMA_SWAP_SERVED_NAME:
                result = run_llama_swap_model(
                    client,
                    base_url,
                    intent=intent,
                    served_name=LLAMA_SWAP_SERVED_NAME[intent],
                    stage=stage,
                    include_tool_call=(intent == "extractor_a"),
                    invoice_image_data_url=invoice_image_data_url,
                )
            elif intent == "parse":
                result = run_transformers_vlm_model(
                    intent=intent, stage=stage, repo_dir=repo_dir, invoice_image=invoice_image
                )
            elif intent == "text_embeddings":
                result = run_sentence_transformer_model(
                    intent=intent,
                    stage=stage,
                    repo_dir=repo_dir,
                    inputs=["Invoice total AED 105.00, invoice number INV-0001."],
                )
            elif intent == "page_similarity":
                result = run_sentence_transformer_model(
                    intent=intent, stage=stage, repo_dir=repo_dir, inputs=[invoice_image]
                )
            else:
                result = run_transformers_load_only_model(
                    intent=intent, stage=stage, repo_dir=repo_dir
                )
            print(f"{'PASS' if result.passed else 'FAIL'} {intent} ({stage})", flush=True)
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    results = run_all(args.lock, args.models_dir, args.base_url)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(results).rstrip("\n") + "\n"
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(f"wrote {args.report}")

    failures = [r.intent for r in results if not r.passed]
    if failures:
        print(
            f"error: {len(failures)} of {len(results)} runtime model(s) failed: {failures}",
            file=sys.stderr,
        )
        return 1
    print(f"all {len(results)} runtime model(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
