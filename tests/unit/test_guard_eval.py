"""Unit tests for evals/guard.py and the prompt-injections fetcher (PRD §10 M6.4): test-split
metrics use a cal-chosen threshold, per-category recall and hard-negative FPR are computed from
the scores, blocked models are reported as such, and the fetcher rejects a bad checksum."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import guard
from evals.datasets import common, prompt_injections_fetcher
from evals.guard import ModelResult, ScoredRow, evaluate_model, load_scores, render_report


def _rows() -> list[ScoredRow]:
    cal = [
        ScoredRow("cal", 1, "attack", "tiny_font", "en", 0.8),
        ScoredRow("cal", 0, "public", "public:train", "", 0.3),
    ]
    test = [
        ScoredRow("test", 1, "attack", "tiny_font", "en", 0.9),
        ScoredRow("test", 1, "attack", "tiny_font", "ar", 0.2),
        ScoredRow("test", 1, "attack", "off_page", "en", 0.85),
        ScoredRow("test", 0, "hard_negative", "hardneg:en", "en", 0.95),
        ScoredRow("test", 0, "hard_negative", "hardneg:ar", "ar", 0.1),
        ScoredRow("test", 0, "public", "public:test", "", 0.05),
    ]
    return cal + test


def test_evaluate_model_uses_cal_threshold_and_scores_only_test() -> None:
    result = evaluate_model("m", _rows(), fixed=[("half", 0.5)])
    half, tuned = result.points
    assert tuned.threshold == 0.8  # max F1 on the two cal rows
    assert (tuned.overall.tp, tuned.overall.fn, tuned.overall.fp, tuned.overall.tn) == (2, 1, 1, 2)
    assert tuned.hard_negative_fpr == 0.5 and tuned.hard_negative_n == 2
    assert tuned.per_category_recall == {"off_page": (1.0, 1), "tiny_font": (0.5, 2)}
    assert half.threshold == 0.5


def test_report_lists_blocked_models_and_categories(tmp_path: Path) -> None:
    scores = tmp_path / "s.jsonl"
    scores.write_text("\n".join(json.dumps(r.__dict__) for r in _rows()) + "\n", encoding="utf-8")
    results = [
        ModelResult("Prompt Guard", "blocked"),
        evaluate_model("m", load_scores(scores)),
    ]
    text = render_report(results, {"test": {"pos": 3, "neg": 3, "hard": 2}})
    assert "| Prompt Guard | blocked |" in text
    assert "| tiny_font | 2 | 0.500 |" in text
    assert "| test | 3 | 3 | 2 |" in text


def test_report_cli_writes_file(tmp_path: Path) -> None:
    scores = tmp_path / "s.jsonl"
    scores.write_text("\n".join(json.dumps(r.__dict__) for r in _rows()), encoding="utf-8")
    out = tmp_path / "guard.md"
    code = guard.main(
        [
            "report",
            "--data-dir",
            str(tmp_path),
            "--scores",
            f"m={scores}",
            "--fixed",
            "m=argmax:0.5",
            "--status",
            "pg=blocked",
            "--out",
            str(out),
        ]
    )
    assert code == 0 and "argmax" in out.read_text(encoding="utf-8")


def test_fetcher_rejects_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)

    def fake_download(url: str, dest: Path, **_: object) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not the pinned parquet")
        return dest

    monkeypatch.setattr(prompt_injections_fetcher, "download_resumable", fake_download)
    with pytest.raises(prompt_injections_fetcher.ChecksumMismatchError):
        prompt_injections_fetcher.fetch(tmp_path / "data")
    assert not prompt_injections_fetcher.local_path(tmp_path / "data", "train").exists()


def test_fetcher_urls_are_pinned_to_a_revision_sha() -> None:
    assert len(prompt_injections_fetcher.REVISION) == 40
    assert all(len(sha) == 64 for _, _, sha, _ in prompt_injections_fetcher.PINNED_FILES)
