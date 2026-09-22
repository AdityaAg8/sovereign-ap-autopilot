"""Fails the suite if any test skips or xfails itself: missing prerequisites must fail loudly and
opt-in tests are selected by marker instead (gpu, integration, artifacts, external)."""

import re
from pathlib import Path

TESTS_DIR = Path(__file__).parent
FORBIDDEN = re.compile(
    r"pytest\.skip\(|mark\.skip\b|mark\.skipif\b|mark\.xfail\b|pytest\.xfail\(|importorskip\("
)


def test_no_test_skips_or_xfails_itself() -> None:
    offenders = [
        f"{path.relative_to(TESTS_DIR)}:{lineno}"
        for path in TESTS_DIR.rglob("*.py")
        if path.name != Path(__file__).name
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]
    assert not offenders, f"skip/xfail found (fail loudly or use a marker instead): {offenders}"
