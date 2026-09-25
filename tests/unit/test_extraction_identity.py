"""Unit tests for extraction row identity (`evals.extraction_identity`, M7.1 review H2): the served
GGUF sha256 resolves from models.lock or an export manifest, the prompt version tracks the prompt
contract, and the harness refuses to resume a row file written by another build or prompt."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

from apx.extract.interfaces import ExtractorId
from apx.extract.prompt import InvoicePromptBuilder, PromptSettings
from evals.extraction_a import prompt_builder_for, run, run_identity
from evals.extraction_identity import (
    RowIdentity,
    RowIdentityError,
    ServedModelError,
    prompt_version,
    require_identity,
    served_gguf_sha256,
    served_paths,
)
from evals.splits import load_splits

REPO_ROOT = Path(__file__).resolve().parents[2]
SWAP = REPO_ROOT / "infra" / "llama-swap.yaml"
LOCK = REPO_ROOT / "infra" / "models.lock.json"
IDENTITY = RowIdentity(model="extractor-a", gguf_sha256="a" * 64, prompt_version="p1")


def test_served_paths_parse_the_llama_swap_cmd() -> None:
    model, mmproj = served_paths(SWAP, "extractor-a")
    assert model == "/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    assert mmproj == "/models/unsloth/Qwen3.5-9B-GGUF/mmproj-F16.gguf"
    with pytest.raises(ServedModelError):
        served_paths(SWAP, "no-such-alias")


def test_hub_gguf_sha_comes_from_models_lock(tmp_path: Path) -> None:
    sha = served_gguf_sha256(
        "extractor-a", llama_swap_config=SWAP, models_lock=LOCK, models_root=tmp_path
    )
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    entry = next(m for m in lock["models"] if m["intent"] == "extractor_a")
    assert sha == next(
        f["sha256"]
        for f in entry["files"]
        if f["filename"].endswith(".gguf") and "mmproj" not in f["filename"]
    )


def test_local_gguf_sha_comes_from_the_export_manifest(tmp_path: Path) -> None:
    model, _ = served_paths(SWAP, "extractor-b-challenger")
    export = tmp_path / model.removeprefix("/models/")
    export.parent.mkdir(parents=True)
    manifest = {"artifacts": [{"name": export.name, "sha256": "c" * 64, "size_bytes": 1}]}
    (export.parent / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
    kwargs = {"llama_swap_config": SWAP, "models_lock": LOCK, "models_root": tmp_path}
    assert served_gguf_sha256("extractor-b-challenger", **kwargs) == "c" * 64  # type: ignore[arg-type]
    (export.parent / "export.json").unlink()
    with pytest.raises(ServedModelError, match="neither pinned"):
        served_gguf_sha256("extractor-b-challenger", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "manifest",
    [
        "{not json",
        "[1, 2]",
        json.dumps({"artifacts": [{"name": "__NAME__"}]}),
        json.dumps({"artifacts": [{"name": "__NAME__", "sha256": "NOT-A-SHA"}]}),
    ],
    ids=["corrupt", "not-object", "no-sha", "malformed-sha"],
)
def test_broken_export_manifest_is_a_served_model_error(tmp_path: Path, manifest: str) -> None:
    """The runtime trust gate treats ServedModelError as a refusal; any other exception type
    would fail the activity and Temporal would retry it forever (M7.5 review F4)."""
    model, _ = served_paths(SWAP, "extractor-b-challenger")
    export = tmp_path / model.removeprefix("/models/")
    export.parent.mkdir(parents=True)
    (export.parent / "export.json").write_text(
        manifest.replace("__NAME__", export.name), encoding="utf-8"
    )
    kwargs = {"llama_swap_config": SWAP, "models_lock": LOCK, "models_root": tmp_path}
    with pytest.raises(ServedModelError):
        served_gguf_sha256("extractor-b-challenger", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "lock",
    ["{broken", "[]", json.dumps({"models": [None]})],
    ids=["corrupt", "not-object", "bad-entry"],
)
def test_broken_models_lock_is_a_served_model_error(tmp_path: Path, lock: str) -> None:
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(lock, encoding="utf-8")
    with pytest.raises(ServedModelError):
        served_gguf_sha256(
            "extractor-a", llama_swap_config=SWAP, models_lock=lock_path, models_root=tmp_path
        )


def test_prompt_version_tracks_the_prompt_contract() -> None:
    a, b = prompt_builder_for(ExtractorId.A), prompt_builder_for(ExtractorId.B)
    assert prompt_version(a) == prompt_version(prompt_builder_for(ExtractorId.A))
    assert prompt_version(a) != prompt_version(b)
    changed = InvoicePromptBuilder(PromptSettings(image_max_side=512))
    assert prompt_version(changed) != prompt_version(a)


def test_require_identity_rejects_foreign_and_legacy_rows(tmp_path: Path) -> None:
    good = {"s1": IDENTITY.as_row_fields()}
    require_identity(good, IDENTITY, tmp_path / "rows.jsonl")
    require_identity({}, IDENTITY, tmp_path / "rows.jsonl")
    other = {**good, "s2": {**IDENTITY.as_row_fields(), "gguf_sha256": "b" * 64}}
    with pytest.raises(RowIdentityError, match=r"rows\.jsonl: 1 of 2"):
        require_identity(other, IDENTITY, tmp_path / "rows.jsonl")
    with pytest.raises(RowIdentityError):
        require_identity({"s3": {"valid_json": True}}, IDENTITY, tmp_path / "rows.jsonl")


async def test_harness_refuses_to_resume_rows_of_another_run(tmp_path: Path) -> None:
    first = sorted(load_splits().sets["uae"]["test"])[0]
    (tmp_path / f"{first}.pdf").write_bytes(b"%PDF-1.4")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"sha": first, "valid_json": True}) + "\n", encoding="utf-8")
    swap = tmp_path / "llama-swap.yaml"
    shutil.copy(SWAP, swap)
    args = argparse.Namespace(
        data_dir=tmp_path,
        limit=0,
        rows=rows,
        extractor="a",
        split="test",
        model="",
        base_url="http://llm.test",
        concurrency=1,
        report=tmp_path / "report.md",
        llama_swap_config=swap,
        models_lock=LOCK,
        models_root=tmp_path,
    )
    with pytest.raises(RowIdentityError, match="from another run"):
        await run(args)
    expected = run_identity(
        ExtractorId.A, "extractor-a", llama_swap_config=swap, models_lock=LOCK, models_root=tmp_path
    )
    assert expected.model == "extractor-a" and not (tmp_path / "report.md").exists()
