"""Unit tests for the evals/datasets fetchers (docile, fatura, extractbench, agentdojo):
token/package presence branching, manifest construction and BLOCKED.md deferral.
All network and subprocess calls are stubbed; no real downloads,
no GPU."""

from __future__ import annotations

import importlib.metadata
import zipfile
from pathlib import Path

import pytest

from evals.datasets import (
    agentdojo_fetcher,
    common,
    docile_fetcher,
    extractbench_fetcher,
    fatura_fetcher,
)
from evals.datasets.manifest import DatasetStatus


def _write_zip(path: Path, name: str = "doc.txt", content: bytes = b"hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, content)


@pytest.fixture
def isolated_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points evals.datasets.common at a throwaway repo root and BLOCKED.md so fetcher tests
    never touch the real repository."""
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(common, "BLOCKED_MD", tmp_path / "BLOCKED.md")
    return tmp_path


# --- docile ----------------------------------------------------------------------------


def test_docile_fetch_defers_without_token(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCILE_TOKEN", raising=False)

    def fail_if_called(*args: object, **kwargs: object) -> Path:
        raise AssertionError("download_resumable must not be called without a token")

    monkeypatch.setattr(docile_fetcher, "download_resumable", fail_if_called)

    entry = docile_fetcher.fetch(isolated_repo_root / "data")

    assert entry.status is DatasetStatus.DEFERRED
    assert entry.files == []
    assert entry.blocked is not None
    blocked_text = (isolated_repo_root / "BLOCKED.md").read_text(encoding="utf-8")
    assert "[dataset:docile]" in blocked_text
    assert "DOCILE_TOKEN=" in blocked_text


def test_docile_fetch_downloads_both_splits_with_token(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCILE_TOKEN", "secret-token")

    def fake_download(url: str, dest: Path, **kwargs: object) -> Path:
        _write_zip(dest)
        return dest

    monkeypatch.setattr(docile_fetcher, "download_resumable", fake_download)

    entry = docile_fetcher.fetch(isolated_repo_root / "data")

    assert entry.status is DatasetStatus.PRESENT
    assert entry.blocked is None
    assert {f.path for f in entry.files} == {
        "data/docile/annotated-trainval.zip",
        "data/docile/test.zip",
    }
    splits = {f.path: f.split for f in entry.files}
    assert splits["data/docile/test.zip"] == "test"
    assert splits["data/docile/annotated-trainval.zip"] is None
    assert not (isolated_repo_root / "BLOCKED.md").exists()


# --- fatura ------------------------------------------------------------------------------


def _write_fatura_zip(path: Path) -> None:
    """Builds a fake FATURA.zip with FATURA's real internal layout: an images/ dir plus the
    official strat1_{train,test}.csv split files. Instance0/Instance1 are officially train,
    Instance2 is officially test, and Instance3/Instance4 are left out of both CSVs so the
    fetcher must fall back to its own seeded split for them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(5):
            archive.writestr(
                f"invoices_dataset_final/images/Template1_Instance{i}.jpg", f"fake-image-{i}"
            )
        archive.writestr(
            "invoices_dataset_final/strat1_train.csv",
            "img_path,annot_path\n"
            "Template1_Instance0.jpg,Template1_Instance0.json\n"
            "Template1_Instance1.jpg,Template1_Instance1.json\n",
        )
        archive.writestr(
            "invoices_dataset_final/strat1_test.csv",
            "img_path,annot_path\nTemplate1_Instance2.jpg,Template1_Instance2.json\n",
        )


def test_fatura_fetch_lists_one_entry_per_document_with_official_split(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(url: str, dest: Path, **kwargs: object) -> Path:
        assert "8261508" in url
        _write_fatura_zip(dest)
        return dest

    monkeypatch.setattr(fatura_fetcher, "download_resumable", fake_download)

    entry = fatura_fetcher.fetch(isolated_repo_root / "data")

    assert entry.status is DatasetStatus.PRESENT
    assert entry.license == "CC-BY-4.0"
    # 1 archive entry (doc_id=None) + 5 per-document image entries.
    assert len(entry.files) == 6
    doc_files = {f.doc_id: f for f in entry.files if f.doc_id is not None}
    assert set(doc_files) == {f"Template1_Instance{i}" for i in range(5)}
    assert doc_files["Template1_Instance0"].split == "train"
    assert doc_files["Template1_Instance1"].split == "train"
    assert doc_files["Template1_Instance2"].split == "test"
    # Not covered by either official CSV: falls back to the seeded deterministic split.
    for i in (3, 4):
        doc_id = f"Template1_Instance{i}"
        assert doc_files[doc_id].split == fatura_fetcher._fallback_split(doc_id)
    archive_entry = next(f for f in entry.files if f.doc_id is None)
    assert archive_entry.path == "data/fatura/FATURA.zip"


def test_fatura_fallback_split_is_deterministic_and_uses_both_buckets() -> None:
    results = {fatura_fetcher._fallback_split(f"doc-{i}") for i in range(50)}
    assert results == {"train", "test"}
    assert fatura_fetcher._fallback_split("stable-id") == fatura_fetcher._fallback_split(
        "stable-id"
    )


# --- extractbench ------------------------------------------------------------------------


def test_extractbench_fetch_builds_manifest_from_downloaded_files(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_snapshot_download(
        *, repo_id: str, repo_type: str, allow_patterns: list[str], local_dir: Path
    ) -> str:
        local_dir = Path(local_dir)
        (local_dir / "short.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (local_dir / "short.jsonl").write_text('{"id": "short/doc1"}\n', encoding="utf-8")
        pdf_dir = local_dir / "docs" / "short"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / "doc1.pdf").write_bytes(b"%PDF-1.4 fake")
        # huggingface_hub's local_dir downloads also leave bookkeeping metadata here; it must
        # not be treated as a dataset document.
        cache_dir = local_dir / ".cache" / "huggingface" / "download"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "short.jsonl.metadata").write_text("etag", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(extractbench_fetcher, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        extractbench_fetcher,
        "_clone_framework",
        lambda external_dir: extractbench_fetcher.FRAMEWORK_COMMIT,
    )

    entry = extractbench_fetcher.fetch(isolated_repo_root / "data")

    assert entry.status is DatasetStatus.PRESENT
    paths = {f.path for f in entry.files}
    assert paths == {
        "data/extractbench/short.jsonl",
        "data/extractbench/docs/short/doc1.pdf",
    }
    assert not any(".cache" in p for p in paths)
    doc_ids = {f.path: f.doc_id for f in entry.files}
    assert doc_ids["data/extractbench/docs/short/doc1.pdf"] == "doc1"
    assert doc_ids["data/extractbench/short.jsonl"] is None
    assert all(f.split == "eval" for f in entry.files)
    assert extractbench_fetcher.FRAMEWORK_COMMIT[:12] in entry.version


def test_clone_framework_checks_out_pinned_commit_when_head_differs(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        if (
            args[:2] == ("-C", str(isolated_repo_root / "external" / "ExtractBench"))
            and "rev-parse" in args
        ):
            # First rev-parse reports a stale commit; after checkout it reports the pin.
            return "deadbeef" if calls.count(args) == 1 else extractbench_fetcher.FRAMEWORK_COMMIT
        return ""

    monkeypatch.setattr(extractbench_fetcher, "_git", fake_git)

    result = extractbench_fetcher._clone_framework(isolated_repo_root / "external")

    assert result == extractbench_fetcher.FRAMEWORK_COMMIT
    joined = [" ".join(c) for c in calls]
    assert any("clone" in c for c in joined)
    assert any("checkout" in c and extractbench_fetcher.FRAMEWORK_COMMIT in c for c in joined)


# --- agentdojo ---------------------------------------------------------------------------


def test_agentdojo_fetch_defers_when_package_missing(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)

    entry = agentdojo_fetcher.fetch(isolated_repo_root / "data")

    assert entry.status is DatasetStatus.DEFERRED
    assert entry.files == []
    blocked_text = (isolated_repo_root / "BLOCKED.md").read_text(encoding="utf-8")
    assert "[dataset:agentdojo]" in blocked_text
    assert "uv add agentdojo" in blocked_text


def test_agentdojo_fetch_present_when_package_installed(
    isolated_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.9")

    entry = agentdojo_fetcher.fetch(isolated_repo_root / "data")

    assert entry.status is DatasetStatus.PRESENT
    assert entry.version == "0.1.9"
    assert entry.blocked is None
    assert not (isolated_repo_root / "BLOCKED.md").exists()
