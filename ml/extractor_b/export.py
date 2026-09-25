"""Extractor B export (ADR-0009 §1.7 a-c, e): merge the `best.json` LoRA checkpoint to 16-bit,
convert it with the pinned llama.cpp to a bf16 GGUF, quantize to Q4_K_M, convert the mmproj and
prove it tensor-equal to the pinned upstream one, then log sha256 and size to MLflow. The pure
planning and checking logic is torch-free; the merge and conversion run in the exporter image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from apx.common.errors import ApxError
from ml.extractor_b.checkpoints import FsCheckpointStore, atomic_write_json, sha256_file
from ml.extractor_b.interfaces import RunRecord, TrainingError

EXPORT_DIR_NAME: Final[str] = "export"
STAGING_DIR_NAME: Final[str] = "export.partial"
MANIFEST_NAME: Final[str] = "export.json"
MERGED_DIR_NAME: Final[str] = "merged_16bit"
Q4_NAME: Final[str] = "extractor-b-Q4_K_M.gguf"
BF16_NAME: Final[str] = "extractor-b-BF16.gguf"
MMPROJ_NAME: Final[str] = "mmproj-F16.gguf"
QUANT_TYPE: Final[str] = "Q4_K_M"
UPSTREAM_MMPROJ: Final[Path] = Path("models/unsloth/Qwen3.5-4B-GGUF/mmproj-F16.gguf")
GPU_LOCK_PATH: Final[Path] = Path("data/ml/gpu.lock")
_MISMATCH_SHOWN: Final[int] = 10
_STDERR_TAIL: Final[int] = 4000


class ExportError(TrainingError):
    """Base for every typed stop of the Extractor B export. Never falls back silently."""


class BestPointerError(ExportError):
    """`best.json` is missing or unreadable, or names a checkpoint that fails verification."""


class BaseMismatchError(ExportError):
    """The run trained on a different base path or revision than the one resolved now."""


class ExportToolError(ExportError):
    """A pinned llama.cpp tool (converter or `llama-quantize`) failed or is missing."""


class MmprojMismatchError(ExportError):
    """The exported mmproj is not tensor-equal to the pinned upstream one (ADR-0009 §1.7 c): a
    conversion or base-revision error, since the vision tower was frozen during training."""


# ---------------------------------------------------------------------------------------------
# Plan: which checkpoint, where the files go.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExportPlan:
    run_id: str
    mlflow_run_id: str
    run_dir: Path
    step: int
    checkpoint_dir: Path
    adapter_sha256: str
    trained_revision: str

    @property
    def export_dir(self) -> Path:
        return self.run_dir / EXPORT_DIR_NAME

    @property
    def staging_dir(self) -> Path:
        return self.run_dir / STAGING_DIR_NAME

    def staged(self, name: str) -> Path:
        return self.staging_dir / name


def load_run_record(run_dir: Path) -> RunRecord:
    path = run_dir / "run.json"
    try:
        return RunRecord.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ExportError(f"cannot read run record {path}: {exc}") from exc


def plan_export(run_dir: Path, *, keep_last: int = 2) -> ExportPlan:
    """Resolve the checkpoint named by `best.json` and verify it (COMPLETE.json, fingerprint and
    every sha256). The last step is never used as a stand-in when `best.json` is absent."""
    record = load_run_record(run_dir)
    store = FsCheckpointStore(run_dir, fingerprint=record.fingerprint, keep_last=keep_last)
    try:
        step = store.best_step()
    except TrainingError as exc:
        raise BestPointerError(f"{run_dir / 'best.json'} is unreadable: {exc}") from exc
    if step is None:
        raise BestPointerError(f"{run_dir / 'best.json'} is missing; export reads only best.json")
    path = store.path_for(step)
    meta = store.verify(path)
    if meta is None:
        raise BestPointerError(f"best.json names step {step}, but {path} fails verification")
    adapter = meta.files.get("adapter_model.safetensors")
    if adapter is None:
        raise BestPointerError(f"{path} has no adapter_model.safetensors")
    return ExportPlan(
        run_id=record.run_id,
        mlflow_run_id=record.mlflow_run_id,
        run_dir=run_dir,
        step=step,
        checkpoint_dir=path,
        adapter_sha256=adapter,
        trained_revision=record.fingerprint.base_revision,
    )


def adapter_base_path(checkpoint_dir: Path) -> str:
    path = checkpoint_dir / "adapter_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, str) or not value:
        raise ExportError(f"{path} names no base_model_name_or_path")
    return value


def check_base(plan: ExportPlan, resolved_base: Path, resolved_revision: str) -> None:
    """The adapter must name the base path the resolver picks now, and the run must have trained
    on the same base revision; merging onto another base is wrong, never approximately right."""
    if resolved_revision != plan.trained_revision:
        raise BaseMismatchError(
            f"run trained on base revision {plan.trained_revision}, resolved base is "
            f"{resolved_revision}"
        )
    trained_on = Path(adapter_base_path(plan.checkpoint_dir))
    if trained_on.as_posix().rstrip("/") != resolved_base.as_posix().rstrip("/"):
        raise BaseMismatchError(
            f"adapter was trained on {trained_on.as_posix()}, resolved base is "
            f"{resolved_base.as_posix()}"
        )


# ---------------------------------------------------------------------------------------------
# Pinned llama.cpp commands.
# ---------------------------------------------------------------------------------------------


def convert_command(
    llama_cpp_dir: Path, model_dir: Path, outfile: Path, *, outtype: str, mmproj: bool = False
) -> list[str]:
    cmd = [
        sys.executable,
        str(llama_cpp_dir / "convert_hf_to_gguf.py"),
        str(model_dir),
        "--outtype",
        outtype,
        "--outfile",
        str(outfile),
    ]
    return [*cmd, "--mmproj"] if mmproj else cmd


def quantize_command(source: Path, target: Path, *, quant: str = QUANT_TYPE) -> list[str]:
    return ["llama-quantize", str(source), str(target), quant]


Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def tool_name(cmd: Sequence[str]) -> str:
    """`convert_hf_to_gguf.py` for a `python <script>` command, else the binary's name."""
    if len(cmd) > 1 and cmd[0] == sys.executable:
        return Path(cmd[1]).name
    return Path(cmd[0]).name


def run_tool(cmd: list[str], *, expect: Path, runner: Runner = _default_runner) -> None:
    """Run one pinned tool; a non-zero exit, a missing binary or a missing output file is an
    `ExportToolError` carrying the tail of stderr."""
    print("export: " + " ".join(cmd), flush=True)
    try:
        done = runner(cmd)
    except OSError as exc:
        raise ExportToolError(f"cannot run {cmd[0]}: {exc}") from exc
    if done.returncode != 0:
        tail = (done.stderr or "")[-_STDERR_TAIL:]
        raise ExportToolError(f"{tool_name(cmd)} exited {done.returncode}:\n{tail}")
    if not expect.is_file():
        raise ExportToolError(f"{tool_name(cmd)} exited 0 but wrote no {expect}")


# ---------------------------------------------------------------------------------------------
# mmproj tensor equality and artifact digests.
# ---------------------------------------------------------------------------------------------


class TensorDigest(BaseModel):
    """One GGUF tensor as compared by the mmproj check: GGML type, shape and data sha256."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    dtype: str
    shape: tuple[int, ...]
    sha256: str


def compare_tensors(
    exported: Mapping[str, TensorDigest], upstream: Mapping[str, TensorDigest]
) -> None:
    """Raise `MmprojMismatchError` unless both files hold the same tensor names with the same
    type, shape and bytes. Metadata keys are not compared: only the weights must be identical."""
    missing = sorted(set(upstream) - set(exported))
    extra = sorted(set(exported) - set(upstream))
    differ = sorted(n for n in set(exported) & set(upstream) if exported[n] != upstream[n])
    if not (missing or extra or differ):
        return
    parts = []
    for label, names in (("missing", missing), ("extra", extra), ("different", differ)):
        if names:
            shown = ", ".join(names[:_MISMATCH_SHOWN])
            more = (
                f" (+{len(names) - _MISMATCH_SHOWN} more)" if len(names) > _MISMATCH_SHOWN else ""
            )
            parts.append(f"{len(names)} {label}: {shown}{more}")
    if differ:
        first = differ[0]
        parts.append(f"e.g. {first}: exported {exported[first]} vs upstream {upstream[first]}")
    raise MmprojMismatchError(
        "exported mmproj is not tensor-equal to upstream; " + "; ".join(parts)
    )


def gguf_tensor_digests(path: Path) -> dict[str, TensorDigest]:
    """Tensor digests of a GGUF file via the pinned llama.cpp `gguf` reader (exporter image)."""
    gguf = importlib.import_module("gguf")
    reader: Any = gguf.GGUFReader(str(path))
    out: dict[str, TensorDigest] = {}
    for tensor in reader.tensors:
        data = tensor.data
        out[str(tensor.name)] = TensorDigest(
            dtype=str(tensor.tensor_type.name),
            shape=tuple(int(x) for x in tensor.shape),
            sha256=hashlib.sha256(data.tobytes()).hexdigest(),
        )
    return out


class ArtifactDigest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    """Path relative to the export directory."""
    sha256: str
    size_bytes: int


def digest_file(root: Path, name: str) -> ArtifactDigest:
    path = root / name
    return ArtifactDigest(name=name, sha256=sha256_file(path), size_bytes=path.stat().st_size)


def digest_tree(root: Path, name: str) -> ArtifactDigest:
    """One digest for a directory: sha256 over sorted `relative-path\\0sha256\\n` lines."""
    base = root / name
    lines, size = [], 0
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        lines.append(f"{path.relative_to(base).as_posix()}\0{sha256_file(path)}\n")
        size += path.stat().st_size
    if not lines:
        raise ExportError(f"{base} is empty")
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return ArtifactDigest(name=name, sha256=digest, size_bytes=size)


class ExportManifest(BaseModel):
    """`export/export.json`: what was exported, from which checkpoint, with which tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    mlflow_run_id: str
    step: int
    adapter_sha256: str
    base_path: str
    base_revision: str
    llama_cpp_commit: str
    quant: str
    artifacts: tuple[ArtifactDigest, ...]
    mmproj_upstream_path: str
    mmproj_upstream_sha256: str
    mmproj_tensors_compared: int
    created_at: datetime

    def artifact(self, name: str) -> ArtifactDigest:
        for item in self.artifacts:
            if item.name == name:
                return item
        raise ExportError(f"export manifest has no artifact {name!r}")


def read_manifest(export_dir: Path) -> ExportManifest | None:
    path = export_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return ExportManifest.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise ExportError(f"{path} is invalid: {exc}") from exc


def verify_export(export_dir: Path) -> ExportManifest | None:
    """The manifest if every file artifact in it still matches its sha256 and size, else None."""
    manifest = read_manifest(export_dir)
    if manifest is None:
        return None
    for item in manifest.artifacts:
        path = export_dir / item.name
        if path.is_dir():
            actual = digest_tree(export_dir, item.name)
        elif path.is_file():
            actual = digest_file(export_dir, item.name)
        else:
            return None
        if actual != item:
            return None
    return manifest


def mlflow_record(manifest: ExportManifest) -> tuple[dict[str, str], dict[str, float]]:
    """(tags, metrics) for the training run. sha256 values are tags, since MLflow params are
    immutable and a re-export must be able to overwrite them; sizes are metrics."""
    tags = {
        "export.step": str(manifest.step),
        "export.quant": manifest.quant,
        "export.llama_cpp_commit": manifest.llama_cpp_commit,
        "export.adapter_sha256": manifest.adapter_sha256,
        "export.mmproj_tensor_equal": "true",
        "export.mmproj_upstream_sha256": manifest.mmproj_upstream_sha256,
    }
    metrics: dict[str, float] = {}
    for item in manifest.artifacts:
        key = item.name.replace("/", "_")
        tags[f"export.{key}.sha256"] = item.sha256
        metrics[f"export.{key}.size_bytes"] = float(item.size_bytes)
    return tags, metrics


# ---------------------------------------------------------------------------------------------
# Container-only steps (exporter image: torch, Unsloth, llama.cpp, gguf).
# ---------------------------------------------------------------------------------------------


def _merge(plan: ExportPlan, out: Path) -> None:  # pragma: no cover - exporter image, GPU
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        str(plan.checkpoint_dir), load_in_4bit=False, local_files_only=True
    )
    model.save_pretrained_merged(str(out), processor, save_method="merged_16bit")
    if not (out / "config.json").is_file():
        raise ExportError(f"merge wrote no config.json to {out}")


def _log_mlflow(uri: str, manifest: ExportManifest) -> None:  # pragma: no cover - live MLflow
    mlflow: Any = importlib.import_module("mlflow")
    mlflow.set_tracking_uri(uri)
    tags, metrics = mlflow_record(manifest)
    try:
        with mlflow.start_run(run_id=manifest.mlflow_run_id):
            mlflow.set_tags(tags)
            mlflow.log_metrics(metrics)
    except Exception as exc:  # the MLflow client raises many unrelated types
        raise ExportError(
            f"export finished but MLflow logging to {uri} failed: {exc!r}; "
            "re-log with `python -m ml.extractor_b.export --log-only`"
        ) from exc


def recover_interrupted_swap(plan: ExportPlan) -> bool:
    """Restore `export/` from `export.replaced/` when a previous run died between the two renames
    of `_promote_staging`. Returns True when it restored something."""
    aside = plan.run_dir / f"{EXPORT_DIR_NAME}.replaced"
    if aside.exists() and not plan.export_dir.exists():
        aside.rename(plan.export_dir)
        return True
    return False


def reusable_export(
    existing: ExportManifest | None, plan: ExportPlan, *, llama_cpp_commit: str
) -> bool:
    """A verified export is reused only for the same step, the same trained base revision and the
    same pinned llama.cpp commit; after a pin bump it is rebuilt, never kept silently."""
    return (
        existing is not None
        and existing.step == plan.step
        and existing.base_revision == plan.trained_revision
        and existing.llama_cpp_commit == llama_cpp_commit
    )


def _promote_staging(plan: ExportPlan) -> None:
    aside = plan.run_dir / f"{EXPORT_DIR_NAME}.replaced"
    if aside.exists():
        shutil.rmtree(aside)
    if plan.export_dir.exists():
        plan.export_dir.rename(aside)
    plan.staging_dir.rename(plan.export_dir)
    if aside.exists():
        shutil.rmtree(aside)


def build_export(
    plan: ExportPlan,
    *,
    base_path: Path,
    base_revision: str,
    llama_cpp_dir: Path,
    llama_cpp_commit: str,
    upstream_mmproj: Path,
    runner: Runner = _default_runner,
    merge: Callable[[ExportPlan, Path], None] = _merge,
    digests: Callable[[Path], dict[str, TensorDigest]] = gguf_tensor_digests,
) -> ExportManifest:
    """Steps (a)-(c) into `export.partial/`, then the manifest, then an atomic swap into
    `export/`. Any failure leaves `export/` untouched and `export.partial/` for inspection."""
    check_base(plan, base_path, base_revision)
    if not upstream_mmproj.is_file():
        raise ExportError(f"pinned upstream mmproj {upstream_mmproj} is missing (make models)")
    if plan.staging_dir.exists():
        shutil.rmtree(plan.staging_dir)
    plan.staging_dir.mkdir(parents=True)

    merged, bf16 = plan.staged(MERGED_DIR_NAME), plan.staged(BF16_NAME)
    q4, mmproj = plan.staged(Q4_NAME), plan.staged(MMPROJ_NAME)
    merge(plan, merged)
    run_tool(
        convert_command(llama_cpp_dir, merged, bf16, outtype="bf16"), expect=bf16, runner=runner
    )
    run_tool(quantize_command(bf16, q4), expect=q4, runner=runner)
    bf16.unlink()  # intermediate only; the merged checkpoint is the kept 16-bit copy
    run_tool(
        convert_command(llama_cpp_dir, merged, mmproj, outtype="f16", mmproj=True),
        expect=mmproj,
        runner=runner,
    )
    upstream = digests(upstream_mmproj)
    compare_tensors(digests(mmproj), upstream)

    root = plan.staging_dir
    manifest = ExportManifest(
        run_id=plan.run_id,
        mlflow_run_id=plan.mlflow_run_id,
        step=plan.step,
        adapter_sha256=plan.adapter_sha256,
        base_path=base_path.as_posix(),
        base_revision=base_revision,
        llama_cpp_commit=llama_cpp_commit,
        quant=QUANT_TYPE,
        artifacts=(
            digest_file(root, Q4_NAME),
            digest_file(root, MMPROJ_NAME),
            digest_tree(root, MERGED_DIR_NAME),
        ),
        mmproj_upstream_path=upstream_mmproj.as_posix(),
        mmproj_upstream_sha256=sha256_file(upstream_mmproj),
        mmproj_tensors_compared=len(upstream),
        created_at=datetime.now(UTC),
    )
    atomic_write_json(root / MANIFEST_NAME, manifest)
    _promote_staging(plan)
    return manifest


# ---------------------------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------------------------


def latest_run_dir(runs_dir: Path) -> Path:
    runs = sorted(p for p in runs_dir.glob("b-*") if (p / "run.json").is_file())
    if not runs:
        raise ExportError(f"no run with run.json under {runs_dir}")
    return runs[-1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.export", description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="default: the newest models/extractor_b/b-*")
    parser.add_argument("--force", action="store_true", help="re-export over a verified export")
    parser.add_argument(
        "--log-only", action="store_true", help="only (re-)log a verified export to MLflow"
    )
    return parser.parse_args(argv)


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ExportError(f"{name} must be set (the exporter image sets it)")
    return value


def run(args: argparse.Namespace, env: Mapping[str, str]) -> int:  # pragma: no cover - image
    from apx.llm.preflight import GpuWorkload, check_workload_headroom
    from ml.extractor_b.config import DEFAULT_CONFIG_PATH, load_config
    from ml.extractor_b.gpu import gpu_lock
    from ml.extractor_b.loop import assert_offline_env
    from ml.extractor_b.train import resolve_base

    cfg = load_config(DEFAULT_CONFIG_PATH)
    run_dir = args.run_dir or latest_run_dir(cfg.runs_dir)
    plan = plan_export(run_dir, keep_last=cfg.retention.keep_last_verified)
    print(f"export: run {plan.run_id}, best.json step {plan.step}", flush=True)
    if recover_interrupted_swap(plan):
        print(f"export: restored {plan.export_dir} from an interrupted replace", flush=True)
    existing = verify_export(plan.export_dir)
    if args.log_only:
        if existing is None:
            raise ExportError(f"--log-only needs a verified export in {plan.export_dir}")
        _log_mlflow(cfg.mlflow_uri, existing)
        return 0
    llama_cpp_commit = _required(env, "LLAMA_CPP_COMMIT")
    if existing is not None and not args.force:
        if reusable_export(existing, plan, llama_cpp_commit=llama_cpp_commit):
            print(
                f"export: {plan.export_dir} already verified for step {plan.step}; "
                "--force redoes it"
            )
            _log_mlflow(cfg.mlflow_uri, existing)
            return 0
        print(
            f"export: {plan.export_dir} was built for step {existing.step}, base "
            f"{existing.base_revision}, llama.cpp {existing.llama_cpp_commit}; rebuilding",
            flush=True,
        )

    assert_offline_env(env)
    check_workload_headroom(GpuWorkload.TRAINER)
    base = resolve_base(env)
    with gpu_lock(cfg.vram, lock_path=GPU_LOCK_PATH, job="extractor_b_export"):
        manifest = build_export(
            plan,
            base_path=base.path,
            base_revision=base.revision,
            llama_cpp_dir=Path(_required(env, "LLAMA_CPP_DIR")),
            llama_cpp_commit=llama_cpp_commit,
            upstream_mmproj=UPSTREAM_MMPROJ,
        )
    _log_mlflow(cfg.mlflow_uri, manifest)
    for item in manifest.artifacts:
        print(f"export: {item.name} {item.size_bytes} bytes sha256 {item.sha256}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI shell
    try:
        return run(parse_args(argv), dict(os.environ))
    except ApxError as exc:
        print(f"export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
