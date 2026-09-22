"""Unit tests for scripts/resolve_models.py: manifest parsing, per-entry Hub resolution
(full snapshot vs pinned files, license lookup, missing-file failure) and the CLI, all
against fake Hub clients -- no network (PRD §8, §10 M0.4).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_SPEC_PATH = Path(__file__).resolve().parents[2] / "scripts" / "resolve_models.py"
_spec = importlib.util.spec_from_file_location("resolve_models", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
rm: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rm  # dataclasses needs the module registered before exec (py3.12)
_spec.loader.exec_module(rm)


def sibling(rfilename: str, size: int | None = None, sha256: str | None = None) -> Any:
    lfs = {"sha256": sha256, "size": size} if sha256 else None
    return SimpleNamespace(rfilename=rfilename, size=None if sha256 else size, lfs=lfs)


def model_info(
    sha: str = "deadbeef",
    siblings: list[Any] | None = None,
    card_data: Any = None,
    tags: list[str] | None = None,
) -> Any:
    return SimpleNamespace(sha=sha, siblings=siblings or [], card_data=card_data, tags=tags or [])


class FakeHub:
    def __init__(self, infos: dict[str, Any]) -> None:
        self._infos = infos
        self.calls: list[tuple[str, str, bool]] = []

    def model_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> Any:
        self.calls.append((repo_id, revision, files_metadata))
        if repo_id not in self._infos:
            import httpx
            from huggingface_hub.errors import HfHubHTTPError

            request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
            response = httpx.Response(404, request=request)
            raise HfHubHTTPError(f"404 for {repo_id}", response=response)
        return self._infos[repo_id]


MANIFEST_YAML = """
models:
  - intent: full_snapshot
    role: "Full repo snapshot model"
    stage: parse
    runtime: true
    repo_id: org/full-repo
    files: null
    vram_gb_est: 2.0
    notes: "example"
  - intent: pinned_files
    role: "Pinned GGUF + mmproj"
    stage: extract
    runtime: true
    repo_id: org/gguf-repo
    files: [model.gguf, mmproj.gguf]
    vram_gb_est: 7.0
    notes: "example"
  - intent: default_revision
    role: "No explicit revision"
    stage: fraud
    runtime: false
    repo_id: org/no-rev
    files: null
    notes: ""
"""


def test_load_manifest_parses_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(MANIFEST_YAML, encoding="utf-8")

    entries = rm.load_manifest(path)

    assert [e.intent for e in entries] == ["full_snapshot", "pinned_files", "default_revision"]
    assert entries[1].files == ["model.gguf", "mmproj.gguf"]
    assert entries[2].revision == "main"  # defaulted when absent
    assert entries[2].vram_gb_est is None


def test_resolve_entry_full_snapshot_collects_every_sibling() -> None:
    entry = rm.ManifestEntry(
        intent="full_snapshot",
        role="role",
        stage="parse",
        runtime=True,
        repo_id="org/full-repo",
        revision="main",
        files=None,
        vram_gb_est=2.0,
        notes="",
    )
    info = model_info(
        sha="abc123",
        siblings=[
            sibling("config.json", size=100),
            sibling("model.safetensors", sha256="f" * 64, size=999),
        ],
        tags=["license:apache-2.0"],
    )
    hub = FakeHub({"org/full-repo": info})

    result = rm.resolve_entry(hub, entry)

    assert result["revision_sha"] == "abc123"
    assert result["license"] == "apache-2.0"
    assert {f["filename"] for f in result["files"]} == {"config.json", "model.safetensors"}
    assert result["total_size_bytes"] == 1099
    assert hub.calls == [("org/full-repo", "main", True)]


def test_resolve_entry_pinned_files_only_selects_those_files() -> None:
    entry = rm.ManifestEntry(
        intent="pinned_files",
        role="role",
        stage="extract",
        runtime=True,
        repo_id="org/gguf-repo",
        revision="main",
        files=["model.gguf"],
        vram_gb_est=7.0,
        notes="",
    )
    info = model_info(
        siblings=[
            sibling("model.gguf", sha256="a" * 64, size=123),
            sibling("mmproj.gguf", sha256="b" * 64, size=456),
            sibling("README.md", size=10),
        ],
        card_data={"license": "apache-2.0"},
    )
    hub = FakeHub({"org/gguf-repo": info})

    result = rm.resolve_entry(hub, entry)

    assert [f["filename"] for f in result["files"]] == ["model.gguf"]
    assert result["files"][0]["sha256"] == "a" * 64
    assert result["license"] == "apache-2.0"


def test_resolve_entry_missing_file_raises() -> None:
    entry = rm.ManifestEntry(
        intent="pinned_files",
        role="role",
        stage="extract",
        runtime=True,
        repo_id="org/gguf-repo",
        revision="main",
        files=["missing.gguf"],
        vram_gb_est=7.0,
        notes="",
    )
    info = model_info(siblings=[sibling("model.gguf", sha256="a" * 64, size=123)])
    hub = FakeHub({"org/gguf-repo": info})

    with pytest.raises(rm.ModelResolutionError, match=r"missing\.gguf"):
        rm.resolve_entry(hub, entry)


def test_resolve_entry_unknown_repo_raises() -> None:
    entry = rm.ManifestEntry(
        intent="ghost",
        role="role",
        stage="extract",
        runtime=True,
        repo_id="org/does-not-exist",
        revision="main",
        files=None,
        vram_gb_est=None,
        notes="",
    )
    hub = FakeHub({})

    with pytest.raises(rm.ModelResolutionError, match="org/does-not-exist"):
        rm.resolve_entry(hub, entry)


def test_resolve_all_writes_expected_shape() -> None:
    entries = [
        rm.ManifestEntry(
            intent="a",
            role="r",
            stage="parse",
            runtime=True,
            repo_id="org/a",
            revision="main",
            files=None,
            vram_gb_est=1.0,
            notes="",
        )
    ]
    hub = FakeHub({"org/a": model_info(siblings=[sibling("f.bin", sha256="c" * 64, size=1)])})

    lock = rm.resolve_all(hub, entries)

    assert lock["source_manifest"] == "infra/models.yaml"
    assert len(lock["models"]) == 1
    assert "generated_at" in lock


def test_main_writes_lock_file_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - intent: only
    role: role
    stage: parse
    runtime: true
    repo_id: org/only
    files: null
    vram_gb_est: 1.0
    notes: ""
""",
        encoding="utf-8",
    )
    out = tmp_path / "models.lock.json"
    hub = FakeHub({"org/only": model_info(siblings=[sibling("f.bin", sha256="d" * 64, size=2)])})
    monkeypatch.setattr(rm, "HfApi", lambda: hub)

    exit_code = rm.main(["--manifest", str(manifest), "--out", str(out)])

    assert exit_code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["models"][0]["intent"] == "only"


def test_main_returns_one_on_resolution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - intent: ghost
    role: role
    stage: parse
    runtime: true
    repo_id: org/ghost
    files: null
    vram_gb_est: 1.0
    notes: ""
""",
        encoding="utf-8",
    )
    out = tmp_path / "models.lock.json"
    hub = FakeHub({})
    monkeypatch.setattr(rm, "HfApi", lambda: hub)

    exit_code = rm.main(["--manifest", str(manifest), "--out", str(out)])

    assert exit_code == 1
    assert not out.exists()
