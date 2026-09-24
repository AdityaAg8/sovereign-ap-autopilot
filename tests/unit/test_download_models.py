"""Unit tests for scripts/download_models.py: sha256/size verification, skip-if-already-
verified resume behaviour, and the multi-model CLI, all against a fake downloader --
no network (PRD §8, §10 M0.4).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SPEC_PATH = Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"
_spec = importlib.util.spec_from_file_location("download_models", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
dm: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")

    assert dm.sha256_of(path) == hashlib.sha256(b"hello world").hexdigest()


def make_model(repo_id: str, filename: str, content: bytes) -> dict:
    sha = hashlib.sha256(content).hexdigest()
    return {
        "intent": "example",
        "repo_id": repo_id,
        "revision_sha": "abc123",
        "revision_requested": "main",
        "runtime": True,
        "files": [{"filename": filename, "size_bytes": len(content), "sha256": sha}],
    }


def test_download_model_writes_file_and_verifies(tmp_path: Path) -> None:
    content = b"weights-bytes"
    model = make_model("org/repo", "model.bin", content)
    models_dir = tmp_path / "models"

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        assert repo_id == "org/repo"
        assert revision == "abc123"
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / filename
        dest.write_bytes(content)
        return dest

    written = dm.download_model(model, models_dir, downloader=fake_downloader)

    assert written == ["model.bin"]
    assert (models_dir / "org/repo" / "model.bin").read_bytes() == content


def test_download_model_skips_already_verified_file(tmp_path: Path) -> None:
    content = b"already-here"
    model = make_model("org/repo", "model.bin", content)
    models_dir = tmp_path / "models"
    dest_dir = models_dir / "org/repo"
    dest_dir.mkdir(parents=True)
    (dest_dir / "model.bin").write_bytes(content)

    calls = []

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        calls.append(filename)
        raise AssertionError("should not re-download an already-verified file")

    written = dm.download_model(model, models_dir, downloader=fake_downloader)

    assert written == ["model.bin"]
    assert calls == []


def test_download_model_redownloads_corrupt_file(tmp_path: Path) -> None:
    good_content = b"correct-bytes"
    model = make_model("org/repo", "model.bin", good_content)
    models_dir = tmp_path / "models"
    dest_dir = models_dir / "org/repo"
    dest_dir.mkdir(parents=True)
    (dest_dir / "model.bin").write_bytes(b"corrupted")

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        dest = local_dir / filename
        dest.write_bytes(good_content)
        return dest

    written = dm.download_model(model, models_dir, downloader=fake_downloader)

    assert written == ["model.bin"]
    assert (dest_dir / "model.bin").read_bytes() == good_content


def test_download_model_raises_on_sha_mismatch(tmp_path: Path) -> None:
    expected = b"expected-bytes"
    model = make_model("org/repo", "model.bin", expected)
    models_dir = tmp_path / "models"

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / filename
        dest.write_bytes(b"wrong-bytes!!!")  # same length as `expected`, different content
        return dest

    with pytest.raises(dm.DownloadError, match="sha256 mismatch"):
        dm.download_model(model, models_dir, downloader=fake_downloader)


def test_download_model_raises_on_size_mismatch(tmp_path: Path) -> None:
    model = make_model("org/repo", "model.bin", b"expected-bytes")
    model["files"][0]["sha256"] = None  # force the size check to be the one that fires
    models_dir = tmp_path / "models"

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / filename
        dest.write_bytes(b"short")
        return dest

    with pytest.raises(dm.DownloadError, match="size mismatch"):
        dm.download_model(model, models_dir, downloader=fake_downloader)


def test_main_runtime_only_skips_offline_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_content = b"runtime-bytes"
    offline_content = b"offline-bytes"
    lock = {
        "models": [
            make_model("org/runtime-repo", "runtime.bin", runtime_content),
            {**make_model("org/offline-repo", "offline.bin", offline_content), "runtime": False},
        ]
    }
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    models_dir = tmp_path / "models"

    calls: list[str] = []

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        calls.append(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        content = runtime_content if repo_id == "org/runtime-repo" else offline_content
        dest = local_dir / filename
        dest.write_bytes(content)
        return dest

    monkeypatch.setattr(dm, "_default_downloader", fake_downloader)

    exit_code = dm.main(
        ["--lock", str(lock_path), "--models-dir", str(models_dir), "--runtime-only"]
    )

    assert exit_code == 0
    assert calls == ["org/runtime-repo"]


def test_main_returns_one_when_a_model_fails_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model("org/repo", "model.bin", b"expected-bytes")
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(json.dumps({"models": [model]}), encoding="utf-8")
    models_dir = tmp_path / "models"

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / filename
        dest.write_bytes(b"wrong-bytes-entirely")
        return dest

    monkeypatch.setattr(dm, "_default_downloader", fake_downloader)

    exit_code = dm.main(["--lock", str(lock_path), "--models-dir", str(models_dir)])

    assert exit_code == 1


def test_main_intent_filter_downloads_only_that_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = {
        "models": [
            {**make_model("org/a", "a.bin", b"a"), "intent": "keep"},
            {**make_model("org/b", "b.bin", b"b"), "intent": "skip"},
        ]
    }
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    calls: list[str] = []

    def fake_downloader(*, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        calls.append(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / filename
        dest.write_bytes(b"a")
        return dest

    monkeypatch.setattr(dm, "_default_downloader", fake_downloader)
    args = ["--lock", str(lock_path), "--models-dir", str(tmp_path / "m")]
    assert dm.main([*args, "--intent", "keep"]) == 0
    assert calls == ["org/a"]
    assert dm.main([*args, "--intent", "nope"]) == 1
    assert calls == ["org/a"]
