"""CLI: scans a directory of PDFs with the M6.2 hidden-content detector and reports flag counts
per technique, plus per-document timing. Used to check the factory's clean invoice corpus
against the "zero flags on 750 clean invoices" gate and for ad hoc scanning; `--details` dumps
every individual flag (file, page, bbox, text, and -- for invisible-fill -- the measured
foreground/background colours and contrast ratio) to diagnose a false positive or negative
(PRD §3, §6.3 SEC-02, M6.2)."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from apx.common.errors import ScanError
from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlag, ScanFlagKind


class ScanDirectoryReport:
    """Aggregate result of scanning every `*.pdf` under one directory."""

    def __init__(self) -> None:
        self.file_count = 0
        self.counts: Counter[str] = Counter()
        self.errors: dict[str, str] = {}
        self.elapsed_seconds = 0.0
        self.details: list[tuple[Path, ScanFlag]] = []
        """Every individual (file, flag) pair, in scan order. Only populated when
        `scan_directory` is called with `collect_details=True` (default: empty, to keep the
        normal aggregate-counts path from holding every flag in memory)."""

    @property
    def total_flags(self) -> int:
        return sum(self.counts.values())


def scan_directory(
    directory: Path, detector: PdfHiddenContentDetector, *, collect_details: bool = False
) -> ScanDirectoryReport:
    """Scan every `*.pdf` under `directory`, recursively. A file that raises `ScanError` is
    recorded under `errors` and never counted as clean (SEC-02). With `collect_details=True`,
    also records every individual flag for `format_details`/`--details`."""
    report = ScanDirectoryReport()
    start = time.perf_counter()
    for path in sorted(directory.rglob("*.pdf")):
        report.file_count += 1
        try:
            flags = detector.detect(path.read_bytes())
        except ScanError as exc:
            report.errors[str(path)] = str(exc)
            continue
        for flag in flags:
            report.counts[flag.kind.value] += 1
            if collect_details:
                report.details.append((path, flag))
    report.elapsed_seconds = time.perf_counter() - start
    return report


def format_report(directory: Path, report: ScanDirectoryReport) -> str:
    lines = [
        f"apx.scan.cli: {report.file_count} PDF(s) under {directory}",
        "",
        f"{'technique':<16} {'count':>6}",
    ]
    for kind in ScanFlagKind:
        lines.append(f"{kind.value:<16} {report.counts.get(kind.value, 0):>6}")
    lines.append(f"{'total':<16} {report.total_flags:>6}")

    if report.errors:
        lines.append("")
        lines.append(
            f"{len(report.errors)} file(s) failed to scan (ScanError, never counted as clean):"
        )
        lines.extend(f"  {path}: {message}" for path, message in report.errors.items())

    per_doc_ms = (report.elapsed_seconds / report.file_count * 1000) if report.file_count else 0.0
    lines.append("")
    lines.append(f"elapsed: {report.elapsed_seconds:.3f}s total, {per_doc_ms:.1f}ms/doc")
    return "\n".join(lines)


def format_details(directory: Path, report: ScanDirectoryReport) -> str:
    """One line per flag: file (relative to `directory` when possible), page, bbox, technique
    and evidence text -- which, for `invisible_text`, already carries the measured foreground/
    background colours and contrast ratio (see `apx.scan.visibility`)."""
    if not report.details:
        return f"apx.scan.cli --details: no flags under {directory}"

    lines = [f"apx.scan.cli --details: {len(report.details)} flag(s) under {directory}", ""]
    for path, flag in report.details:
        try:
            file_label = str(path.relative_to(directory))
        except ValueError:
            file_label = str(path)
        bbox = flag.bbox
        bbox_label = f"({bbox.x0:.3f},{bbox.y0:.3f})-({bbox.x1:.3f},{bbox.y1:.3f})" if bbox else "-"
        lines.append(
            f"{file_label} page={flag.page if flag.page is not None else '-'} "
            f"bbox={bbox_label} [{flag.kind.value}] {flag.evidence.text}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan a directory of PDFs for hidden content (M6.2)."
    )
    parser.add_argument(
        "directory", type=Path, help="Directory to scan recursively for *.pdf files."
    )
    parser.add_argument(
        "--fail-on-flags",
        action="store_true",
        help="Exit 1 if any flag was raised (for CI gating).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also print one line per individual flag (file, page, bbox, text, and -- for "
        "invisible_text -- the measured foreground/background colours and contrast ratio), "
        "to diagnose a false positive or negative.",
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    report = scan_directory(
        args.directory, PdfHiddenContentDetector(), collect_details=args.details
    )
    print(format_report(args.directory, report))
    if args.details:
        print()
        print(format_details(args.directory, report))

    if report.errors:
        return 2
    if args.fail_on_flags and report.total_flags > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
