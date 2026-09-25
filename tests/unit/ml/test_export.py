"""Unit tests for the Extractor B export's pure logic (ADR-0009 §1.7): best.json-only checkpoint
selection, base checks, pinned tool commands, typed tool failures, mmproj tensor equality and the
staged, atomically swapped export directory. Tiny fakes stand in for the merge and llama.cpp."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.extractor_b.checkpoints import FsCheckpointStore, atomic_write_json
from ml.extractor_b.export import (
    BF16_NAME,
    MANIFEST_NAME,
    MERGED_DIR_NAME,
    MMPROJ_NAME,
    Q4_NAME,
    BaseMismatchError,
    BestPointerError,
    ExportError,
    ExportManifest,
    ExportPlan,
    ExportToolError,
    MmprojMismatchError,
    TensorDigest,
    build_export,
    check_base,
    compare_tensors,
    convert_command,
    digest_file,
    digest_tree,
    latest_run_dir,
    mlflow_record,
    plan_export,
    quantize_command,
    recover_interrupted_swap,
    reusable_export,
    run_tool,
    tool_name,
    verify_export,
)
from ml.extractor_b.interfaces import CheckpointMeta, RunRecord
from tests.unit.ml.conftest import make_fingerprint

NOW = datetime(2026, 9, 25, tzinfo=UTC)
BASE = Path("models/Qwen/Qwen3.5-4B")


def _run_dir(
    tmp_path: Path, steps: tuple[int, ...] = (383, 500, 510), best: int | None = 383
) -> Path:
    run_dir = tmp_path / "runs" / "b-20260923T2025-eb0aea7d"
    fp = make_fingerprint()
    atomic_write_json(
        run_dir / "run.json",
        RunRecord(run_id=run_dir.name, mlflow_run_id="mlf-1", fingerprint=fp, created_at=NOW),
    )
    store = FsCheckpointStore(run_dir, fingerprint=fp, keep_last=2)
    for step in steps:
        out = store.begin(step)
        (out / "adapter_model.safetensors").write_bytes(b"lora" + str(step).encode())
        (out / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": BASE.as_posix()}), encoding="utf-8"
        )
        meta = CheckpointMeta(step=step, epoch=1.0, created_at=NOW, fingerprint=fp, files={})
        store.commit(step, meta)
        if step == best:  # as in training: best is set when its eval lands, so it survives pruning
            store.set_best(best)
    return run_dir


# --- plan_export: best.json only ------------------------------------------------------------


def test_plan_uses_best_json_not_the_last_step(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    assert plan.step == 383
    assert plan.checkpoint_dir.name == "checkpoint-000383"
    assert plan.mlflow_run_id == "mlf-1"
    assert plan.trained_revision == "rev-1"
    assert plan.export_dir == plan.run_dir / "export"
    assert len(plan.adapter_sha256) == 64


def test_missing_best_json_never_falls_back_to_latest(tmp_path: Path) -> None:
    with pytest.raises(BestPointerError, match="missing"):
        plan_export(_run_dir(tmp_path, best=None))


def test_malformed_best_json_is_typed(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir / "best.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BestPointerError, match="unreadable"):
        plan_export(run_dir)


def test_best_checkpoint_failing_verification_is_refused(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    adapter = run_dir / "checkpoints" / "checkpoint-000383" / "adapter_model.safetensors"
    adapter.write_bytes(b"flipped")
    with pytest.raises(BestPointerError, match="fails verification"):
        plan_export(run_dir)


def test_best_pointing_at_a_pruned_step_is_refused(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir / "best.json").write_text('{"step": 250}', encoding="utf-8")
    with pytest.raises(BestPointerError, match="step 250"):
        plan_export(run_dir)


def test_missing_run_record_is_typed(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match=r"run\.json"):
        plan_export(tmp_path / "nowhere")


def test_latest_run_dir_picks_newest_with_run_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    (run_dir.parent / "b-20260924T0000-deadbeef").mkdir()  # no run.json: not a run
    assert latest_run_dir(run_dir.parent) == run_dir
    with pytest.raises(ExportError):
        latest_run_dir(tmp_path / "empty")


# --- base checks ----------------------------------------------------------------------------


def test_check_base_accepts_the_trained_base(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    check_base(plan, Path("models/Qwen/Qwen3.5-4B/"), "rev-1")


@pytest.mark.parametrize(
    ("path", "revision", "match"),
    [(Path("models/Qwen/Other"), "rev-1", "trained on models"), (BASE, "rev-2", "revision")],
)
def test_check_base_refuses_another_base(
    tmp_path: Path, path: Path, revision: str, match: str
) -> None:
    plan = plan_export(_run_dir(tmp_path))
    with pytest.raises(BaseMismatchError, match=match):
        check_base(plan, path, revision)


# --- pinned tool commands and typed failures ------------------------------------------------


def test_commands_use_the_pinned_converter_and_quantizer() -> None:
    llama = Path("/opt/llama.cpp")
    text = convert_command(llama, Path("m"), Path("o.gguf"), outtype="bf16")
    assert text[1:] == [
        str(llama / "convert_hf_to_gguf.py"),
        "m",
        "--outtype",
        "bf16",
        "--outfile",
        "o.gguf",
    ]
    mmproj = convert_command(llama, Path("m"), Path("p.gguf"), outtype="f16", mmproj=True)
    assert mmproj[-1] == "--mmproj" and "f16" in mmproj
    assert quantize_command(Path("a"), Path("b")) == ["llama-quantize", "a", "b", "Q4_K_M"]
    assert tool_name(text) == "convert_hf_to_gguf.py"
    assert tool_name(quantize_command(Path("a"), Path("b"))) == "llama-quantize"


def _done(code: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=code, stdout="", stderr=stderr)


def test_run_tool_nonzero_exit_carries_stderr(tmp_path: Path) -> None:
    with pytest.raises(ExportToolError, match=r"llama-quantize exited 3:\n.*bad tensor"):
        run_tool(
            ["llama-quantize", "a", "b", "Q4_K_M"],
            expect=tmp_path / "b",
            runner=lambda cmd: _done(3, "x" * 5000 + "bad tensor"),
        )


def test_run_tool_missing_binary_and_missing_output(tmp_path: Path) -> None:
    def missing(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(cmd[0])

    with pytest.raises(ExportToolError, match="cannot run"):
        run_tool(["llama-quantize"], expect=tmp_path / "x", runner=missing)
    with pytest.raises(ExportToolError, match="wrote no"):
        run_tool(["llama-quantize"], expect=tmp_path / "x", runner=lambda cmd: _done(0))


# --- mmproj tensor equality -----------------------------------------------------------------


def _digests(**changes: TensorDigest | None) -> dict[str, TensorDigest]:
    out = {
        "v.blk.0.attn_qkv.weight": TensorDigest(dtype="F16", shape=(1024, 3072), sha256="a" * 64),
        "v.blk.0.attn_qkv.bias": TensorDigest(dtype="F32", shape=(3072,), sha256="b" * 64),
        "mm.0.weight": TensorDigest(dtype="F16", shape=(4096, 2560), sha256="c" * 64),
    }
    for name, value in changes.items():
        if value is None:
            out.pop(name.replace("__", "."), None)
        else:
            out[name.replace("__", ".")] = value
    return out


def test_equal_tensors_pass() -> None:
    compare_tensors(_digests(), _digests())


@pytest.mark.parametrize(
    ("exported", "match"),
    [
        (_digests(mm__0__weight=None), "1 missing: mm.0.weight"),
        (_digests(mm__1__weight=TensorDigest(dtype="F16", shape=(1,), sha256="d" * 64)), "1 extra"),
        (
            _digests(mm__0__weight=TensorDigest(dtype="F16", shape=(4096, 2560), sha256="e" * 64)),
            "1 different: mm.0.weight",
        ),
        (
            _digests(mm__0__weight=TensorDigest(dtype="F32", shape=(4096, 2560), sha256="c" * 64)),
            "dtype='F32'",
        ),
    ],
)
def test_any_tensor_difference_fails_the_export(
    exported: dict[str, TensorDigest], match: str
) -> None:
    with pytest.raises(MmprojMismatchError, match=match):
        compare_tensors(exported, _digests())


# --- digests and the staged export ----------------------------------------------------------


def test_digest_tree_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    for root, order in ((tmp_path / "a", ("x", "y")), (tmp_path / "b", ("y", "x"))):
        (root / "m").mkdir(parents=True)
        for name in order:
            (root / "m" / name).write_bytes(name.encode())
    assert digest_tree(tmp_path / "a", "m") == digest_tree(tmp_path / "b", "m")
    (tmp_path / "b" / "m" / "x").write_bytes(b"changed")
    assert digest_tree(tmp_path / "a", "m").sha256 != digest_tree(tmp_path / "b", "m").sha256
    (tmp_path / "c" / "m").mkdir(parents=True)
    with pytest.raises(ExportError, match="empty"):
        digest_tree(tmp_path / "c", "m")


class FakeTools:
    """Runner, merge and GGUF digests standing in for Unsloth and the pinned llama.cpp."""

    def __init__(self, *, mmproj_sha: str = "a" * 64) -> None:
        self.calls: list[list[str]] = []
        self.mmproj_sha = mmproj_sha

    def merge(self, plan: ExportPlan, out: Path) -> None:
        out.mkdir()
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"merged" + str(plan.step).encode())

    def runner(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        target = Path(cmd[2] if cmd[0] == "llama-quantize" else cmd[cmd.index("--outfile") + 1])
        target.write_bytes(" ".join(cmd[1:]).encode())
        return _done(0)

    def digests(self, path: Path) -> dict[str, TensorDigest]:
        sha = (
            self.mmproj_sha
            if path.name == MMPROJ_NAME and "upstream" not in str(path)
            else "a" * 64
        )
        return {"v.patch_embd.weight": TensorDigest(dtype="F16", shape=(2, 2), sha256=sha)}


def _build(plan: ExportPlan, tmp_path: Path, tools: FakeTools) -> ExportManifest:
    upstream = tmp_path / "upstream" / MMPROJ_NAME
    upstream.parent.mkdir(exist_ok=True)
    upstream.write_bytes(b"upstream mmproj")
    return build_export(
        plan,
        base_path=BASE,
        base_revision="rev-1",
        llama_cpp_dir=Path("/opt/llama.cpp"),
        llama_cpp_commit="c550d2f",
        upstream_mmproj=upstream,
        runner=tools.runner,
        merge=tools.merge,
        digests=tools.digests,
    )


def test_build_export_writes_verified_artifacts_and_manifest(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    tools = FakeTools()
    manifest = _build(plan, tmp_path, tools)
    assert [c[0] for c in tools.calls][1] == "llama-quantize"
    assert "--mmproj" in tools.calls[2]
    assert not plan.staging_dir.exists()
    export = plan.export_dir
    assert sorted(p.name for p in export.iterdir()) == sorted(
        [MANIFEST_NAME, MERGED_DIR_NAME, MMPROJ_NAME, Q4_NAME]
    )  # the bf16 GGUF is an intermediate and is removed
    assert not (export / BF16_NAME).exists()
    verified = verify_export(export)
    assert verified == manifest
    assert verified is not None and verified.step == 383 and verified.quant == "Q4_K_M"
    assert verified.artifact(Q4_NAME) == digest_file(export, Q4_NAME)
    assert verified.mmproj_tensors_compared == 1


def test_mmproj_mismatch_leaves_the_old_export_untouched(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    first = _build(plan, tmp_path, FakeTools())
    with pytest.raises(MmprojMismatchError):
        _build(plan, tmp_path, FakeTools(mmproj_sha="f" * 64))
    assert verify_export(plan.export_dir) == first
    assert plan.staging_dir.is_dir()  # kept for inspection


def test_reexport_replaces_the_previous_one(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    _build(plan, tmp_path, FakeTools())
    second = _build(plan, tmp_path, FakeTools())
    assert verify_export(plan.export_dir) == second
    assert not (plan.run_dir / "export.replaced").exists()


def test_existing_export_is_reused_only_under_the_same_pins(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    manifest = _build(plan, tmp_path, FakeTools())
    assert reusable_export(manifest, plan, llama_cpp_commit="c550d2f")
    assert not reusable_export(manifest, plan, llama_cpp_commit="0000000")  # llama.cpp pin bumped
    assert not reusable_export(None, plan, llama_cpp_commit="c550d2f")
    other_base = dataclasses.replace(plan, trained_revision="rev-2")
    assert not reusable_export(manifest, other_base, llama_cpp_commit="c550d2f")
    other_step = dataclasses.replace(plan, step=plan.step + 1)
    assert not reusable_export(manifest, other_step, llama_cpp_commit="c550d2f")


def test_interrupted_replace_is_recovered(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    first = _build(plan, tmp_path, FakeTools())
    plan.export_dir.rename(plan.run_dir / "export.replaced")  # died between the two renames
    assert recover_interrupted_swap(plan)
    assert verify_export(plan.export_dir) == first
    assert not recover_interrupted_swap(plan)  # nothing left to restore


def test_build_refuses_a_missing_upstream_mmproj_and_another_base(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    kwargs: dict[str, object] = {
        "base_revision": "rev-1",
        "llama_cpp_dir": Path("/opt"),
        "llama_cpp_commit": "c",
        "upstream_mmproj": tmp_path / "absent.gguf",
    }
    with pytest.raises(ExportError, match="upstream mmproj"):
        build_export(plan, base_path=BASE, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(BaseMismatchError):
        build_export(plan, base_path=Path("models/x"), **kwargs)  # type: ignore[arg-type]
    assert not plan.export_dir.exists()


def test_tampered_export_does_not_verify(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    _build(plan, tmp_path, FakeTools())
    (plan.export_dir / Q4_NAME).write_bytes(b"tampered")
    assert verify_export(plan.export_dir) is None
    assert verify_export(tmp_path / "no-export") is None


def test_mlflow_record_has_sha256_tags_and_size_metrics(tmp_path: Path) -> None:
    plan = plan_export(_run_dir(tmp_path))
    manifest = _build(plan, tmp_path, FakeTools())
    tags, metrics = mlflow_record(manifest)
    assert tags["export.step"] == "383"
    assert tags[f"export.{Q4_NAME}.sha256"] == digest_file(plan.export_dir, Q4_NAME).sha256
    assert tags["export.mmproj_tensor_equal"] == "true"
    assert set(metrics) == {
        f"export.{Q4_NAME}.size_bytes",
        f"export.{MMPROJ_NAME}.size_bytes",
        f"export.{MERGED_DIR_NAME}.size_bytes",
    }


def test_challenger_entry_serves_the_export_and_leaves_extractor_b_alone() -> None:
    import yaml

    from apx.llm.client import RUNTIME_MODEL_ALIASES

    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "infra" / "llama-swap.yaml").read_text(encoding="utf-8"))
    models = config["models"]
    challenger = models["extractor-b-challenger"]["cmd"]
    assert f"/export/{Q4_NAME}" in challenger and f"/export/{MMPROJ_NAME}" in challenger
    served = models["extractor-b"]["cmd"]
    if "/models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" not in served:
        # Only `ml.extractor_b.promote` repoints extractor-b, and it records the champion in
        # models.lock `local_models` first (ADR-0009 §1.8).
        lock = json.loads((root / "infra" / "models.lock.json").read_text(encoding="utf-8"))
        assert f"/export/{Q4_NAME}" in served and f"/export/{MMPROJ_NAME}" in served
        entries = [
            e for e in lock.get("local_models", []) if e["registered_model"] == "extractor_b"
        ]
        # promote writes the served Q4_K_M sha256 into the entry's description; the lock entry
        # of that build must carry the same sha256, and so must the export manifest when the
        # (never committed) export is present on this machine.
        description = models["extractor-b"]["description"]
        served_entry = [e for e in entries if e["gguf_sha256"] in description]
        assert len(served_entry) == 1
        export_manifest = root / served.split("--model ")[1].split()[0].removeprefix("/")
        export_manifest = export_manifest.parent / "export.json"
        if export_manifest.is_file():
            manifest = json.loads(export_manifest.read_text(encoding="utf-8"))
            q4 = next(a for a in manifest["artifacts"] if a["name"] == Q4_NAME)
            assert served_entry[0]["gguf_sha256"] == q4["sha256"]
    assert "extractor-b-challenger" not in RUNTIME_MODEL_ALIASES
    groups = config["routing"]["router"]["settings"]["groups"]
    assert "extractor-b-challenger" in groups["extract"]["members"]  # swaps with A and B
