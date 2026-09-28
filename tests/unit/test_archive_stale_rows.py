"""Unit tests for `evals.archive_stale_rows` (ADR-0018 §5): row files from another parser build
are moved, with their parse-failure sibling, to the archive and never deleted; current, empty and
absent files stay; an unreadable file fails loudly instead of being kept or moved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.archive_stale_rows import archive_stale, main

CURRENT = "c" * 16
OLD = "440bcc9b05e6d980"


def _rows(path: Path, *identities: str | None) -> Path:
    lines = []
    for i, identity in enumerate(identities):
        row: dict[str, object] = {"sha": f"{i:064x}"}
        if identity is not None:
            row["parser_identity"] = identity
        lines.append(json.dumps(row) + "\n")
    path.write_text("".join(lines), "utf-8")
    return path


def test_stale_files_move_with_their_failures_and_current_ones_stay(tmp_path: Path) -> None:
    logs, archive = tmp_path / "logs", tmp_path / "logs" / "archive"
    logs.mkdir()
    stale = _rows(logs / "extractor_b_test.jsonl", OLD, OLD)
    failures = _rows(logs / "extractor_b_test.parse_failures.jsonl", None)
    current = _rows(logs / "cal_a.jsonl", CURRENT, CURRENT)
    empty = logs / "cal_b.jsonl"
    empty.write_text("", "utf-8")
    stale_bytes = stale.read_bytes()

    verdicts = archive_stale([stale, current, empty, logs / "absent.jsonl"], CURRENT, archive)

    by_path = {v.path: v for v in verdicts}
    assert set(by_path) == {stale, current, empty}
    assert by_path[stale].moved_to == archive / f"extractor_b_test.{OLD}.jsonl"
    assert (archive / f"extractor_b_test.{OLD}.jsonl").read_bytes() == stale_bytes
    assert (archive / f"extractor_b_test.parse_failures.{OLD}.jsonl").is_file()
    assert not stale.exists() and not failures.exists()
    assert current.is_file() and empty.is_file()
    assert not by_path[current].stale and not by_path[empty].stale


def test_mixed_or_unstamped_rows_are_stale_and_nothing_is_overwritten(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / f"a.{OLD}.jsonl").write_text("earlier\n", "utf-8")
    partly = _rows(tmp_path / "a.jsonl", CURRENT, OLD)  # one current row does not save the file
    mixed = _rows(tmp_path / "m.jsonl", OLD, "d" * 16)
    unstamped = _rows(tmp_path / "b.jsonl", None)
    verdicts = archive_stale([partly, mixed, unstamped], CURRENT, archive)
    assert [v.moved_to for v in verdicts] == [
        archive / f"a.{OLD}.1.jsonl",
        archive / "m.mixed.jsonl",
        archive / "b.none.jsonl",
    ]
    assert (archive / f"a.{OLD}.jsonl").read_text("utf-8") == "earlier\n"


def test_dry_run_moves_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stale = _rows(tmp_path / "a.jsonl", OLD)
    code = main(
        [
            str(stale),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--parser-identity",
            CURRENT,
            "--dry-run",
        ]
    )
    assert code == 0 and stale.is_file() and not (tmp_path / "archive").exists()
    assert "would move" in capsys.readouterr().out


def test_unreadable_file_fails_loudly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "a.jsonl"
    broken.write_text("{not json\n", "utf-8")
    code = main([str(broken), "--archive-dir", str(tmp_path / "archive"), "--parser-identity", "x"])
    assert code == 2 and broken.is_file()
    assert "not JSON" in capsys.readouterr().err
