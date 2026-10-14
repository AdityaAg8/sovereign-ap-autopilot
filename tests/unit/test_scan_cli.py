"""Unit tests for `apx.scan.cli`: scans a directory of PDFs and
reports flag counts per technique, and never counts a `ScanError` file as clean."""

from __future__ import annotations

from pathlib import Path

import pytest

from apx.scan.cli import format_details, format_report, main, scan_directory
from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_scan_directory_counts_flags_by_kind(tmp_path: Path) -> None:
    (tmp_path / "clean.pdf").write_bytes(pb.clean_invoice_pdf())
    (tmp_path / "invisible.pdf").write_bytes(pb.invisible_text_pdf())
    (tmp_path / "attachment.pdf").write_bytes(pb.attachment_pdf())

    report = scan_directory(tmp_path, PdfHiddenContentDetector())

    assert report.file_count == 3
    assert report.errors == {}
    assert report.counts[ScanFlagKind.INVISIBLE_TEXT.value] >= 1
    assert report.counts[ScanFlagKind.PDF_ATTACHMENT.value] == 1
    assert report.total_flags == report.counts[ScanFlagKind.INVISIBLE_TEXT.value] + 1


def test_scan_directory_records_unparseable_files_as_errors_not_clean(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")

    report = scan_directory(tmp_path, PdfHiddenContentDetector())

    assert report.file_count == 1
    assert report.total_flags == 0
    assert len(report.errors) == 1
    (error_path,) = report.errors
    assert error_path.endswith("broken.pdf")


def test_scan_directory_recurses_into_subdirectories(tmp_path: Path) -> None:
    sub = tmp_path / "batch1"
    sub.mkdir()
    (sub / "invoice.pdf").write_bytes(pb.clean_invoice_pdf())

    report = scan_directory(tmp_path, PdfHiddenContentDetector())

    assert report.file_count == 1


def test_format_report_lists_every_technique_and_the_elapsed_time(tmp_path: Path) -> None:
    (tmp_path / "clean.pdf").write_bytes(pb.clean_invoice_pdf())
    report = scan_directory(tmp_path, PdfHiddenContentDetector())

    text = format_report(tmp_path, report)

    for kind in ScanFlagKind:
        assert kind.value in text
    assert "elapsed" in text


def test_main_exits_nonzero_when_a_file_fails_to_scan(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")

    exit_code = main([str(tmp_path)])

    assert exit_code == 2


def test_main_fail_on_flags_exits_nonzero_only_when_requested(tmp_path: Path) -> None:
    (tmp_path / "invisible.pdf").write_bytes(pb.invisible_text_pdf())

    assert main([str(tmp_path)]) == 0
    assert main([str(tmp_path), "--fail-on-flags"]) == 1


def test_scan_directory_collects_no_details_by_default(tmp_path: Path) -> None:
    (tmp_path / "invisible.pdf").write_bytes(pb.invisible_text_pdf())

    report = scan_directory(tmp_path, PdfHiddenContentDetector())

    assert report.details == []


def test_scan_directory_collects_details_when_requested(tmp_path: Path) -> None:
    (tmp_path / "invisible.pdf").write_bytes(pb.invisible_text_pdf())

    report = scan_directory(tmp_path, PdfHiddenContentDetector(), collect_details=True)

    assert report.details, "expected at least one (path, flag) detail"
    path, flag = report.details[0]
    assert path == tmp_path / "invisible.pdf"
    assert flag.kind is not None


def test_format_details_reports_file_page_bbox_and_measured_colours(tmp_path: Path) -> None:
    (tmp_path / "invisible.pdf").write_bytes(pb.invisible_text_pdf())
    report = scan_directory(tmp_path, PdfHiddenContentDetector(), collect_details=True)

    text = format_details(tmp_path, report)

    assert "invisible.pdf" in text
    assert "page=0" in text
    assert ScanFlagKind.INVISIBLE_TEXT.value in text
    assert "fg=" in text
    assert "bg=" in text
    assert "contrast=" in text


def test_format_details_with_no_flags_says_so(tmp_path: Path) -> None:
    (tmp_path / "clean.pdf").write_bytes(pb.clean_invoice_pdf())
    report = scan_directory(tmp_path, PdfHiddenContentDetector(), collect_details=True)

    text = format_details(tmp_path, report)

    assert "no flags" in text


def test_main_details_flag_prints_per_flag_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "invisible.pdf").write_bytes(pb.invisible_text_pdf())

    exit_code = main([str(tmp_path), "--details"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "invisible.pdf" in captured.out
    assert "contrast=" in captured.out
