"""Leakage gate for PRD §6.2 I5 ("eval, cal and test data never reach training") and ADR-0005:
validates the committed evals/splits.json has no cross-pool sha256 overlap and no duplicate
within a split, greps apx/ to confirm only apx/trust references the cal split (quoted "cal" or a
`*Split.CAL` member; the one exempt line is the DB enum declaring the value), and proves the
checker is not a no-op by injecting a duplicate into a temp copy of splits.json. No network,
no GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evals.splits import (
    SPLITS_PATH,
    find_duplicate_documents,
    load_splits,
    validate_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
APX_ROOT = REPO_ROOT / "apx"

# A "cal" split name used as a literal string (e.g. `split == "cal"` or a dict key), not the
# substring inside a longer word like "calibrator" or "calendar" (PRD §5: "Only apx/trust
# uses the cal split").
_QUOTED_CAL_RE = re.compile(r"""(['"])cal\1""")
# Referencing the split through an enum member (e.g. `LabelSplit.CAL`) is a use too.
_CAL_MEMBER_RE = re.compile(r"\b\w*Split\.CAL\b")
# The only exemption (ADR-0005 amendment): the DB schema *declaring* the split value so label
# rows can record it. Declaring is not reading; any other "cal" in that file is still flagged.
_DECLARATION_EXEMPTIONS: dict[tuple[str, ...], re.Pattern[str]] = {
    ("db", "enums.py"): re.compile(r"""^\s*CAL = "cal"$"""),
}


def find_unauthorized_cal_references(apx_root: Path = APX_ROOT) -> list[str]:
    """Grep-based static check: returns "path:lineno" for every quoted "cal"/'cal' token (or
    `*Split.CAL` member) in an apx/ Python file outside apx/trust/, except the single
    declaration line in `_DECLARATION_EXEMPTIONS`. Empty means only apx/trust references it."""
    violations: list[str] = []
    for path in sorted(apx_root.rglob("*.py")):
        relative_parts = path.relative_to(apx_root).parts
        if relative_parts[0] == "trust" or "__pycache__" in relative_parts:
            continue
        exemption = _DECLARATION_EXEMPTIONS.get(relative_parts)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if exemption is not None and exemption.match(line):
                continue
            if _QUOTED_CAL_RE.search(line) or _CAL_MEMBER_RE.search(line):
                violations.append(f"apx/{path.relative_to(apx_root).as_posix()}:{lineno}")
    return violations


def test_real_splits_have_no_leakage_or_duplicates() -> None:
    splits = load_splits()
    problems = validate_splits(splits)
    assert problems == [], "\n".join(problems)


def test_real_splits_json_exists_and_is_nonempty_for_present_sets() -> None:
    # Sanity guard so a broken build (e.g. build_from_manifest silently returning empty
    # groups) doesn't slip through test_real_splits_have_no_leakage_or_duplicates() vacuously.
    assert SPLITS_PATH.exists(), f"{SPLITS_PATH} is missing; run `python -m evals.splits build`"
    splits = load_splits()
    fatura_docs = sum(len(v) for v in splits.sets["fatura"].values())
    extractbench_docs = sum(len(v) for v in splits.sets["extractbench"].values())
    assert fatura_docs > 0, "fatura split is empty; did the fatura fetch run?"
    assert extractbench_docs > 0, "extractbench split is empty; did the extractbench fetch run?"


def test_only_apx_trust_references_the_cal_split() -> None:
    violations = find_unauthorized_cal_references()
    assert violations == [], "\n".join(violations)


def test_find_unauthorized_cal_references_detects_a_planted_violation(tmp_path: Path) -> None:
    """Proves the grep check is not a no-op: a fake apx/ tree with a non-trust module quoting
    "cal" must be flagged, while the same reference inside apx/trust/ must not be."""
    fake_apx = tmp_path / "apx"
    (fake_apx / "policy").mkdir(parents=True)
    (fake_apx / "trust").mkdir(parents=True)
    (fake_apx / "policy" / "engine.py").write_text('split = "cal"\n', encoding="utf-8")
    (fake_apx / "trust" / "calibrate.py").write_text('split = "cal"\n', encoding="utf-8")

    violations = find_unauthorized_cal_references(fake_apx)

    assert len(violations) == 1
    assert "policy" in violations[0]
    assert "trust" not in violations[0]


def test_enum_member_use_is_flagged_and_only_the_exact_declaration_is_exempt(
    tmp_path: Path,
) -> None:
    """`LabelSplit.CAL` outside apx/trust is a use and is flagged; in apx/db/enums.py only the
    exact `CAL = "cal"` declaration is exempt, so any other "cal" there is still caught."""
    fake_apx = tmp_path / "apx"
    (fake_apx / "workflows").mkdir(parents=True)
    (fake_apx / "db").mkdir(parents=True)
    (fake_apx / "workflows" / "select.py").write_text(
        "rows = labels_for(LabelSplit.CAL)\n", encoding="utf-8"
    )
    (fake_apx / "db" / "enums.py").write_text(
        'class LabelSplit:\n    CAL = "cal"\n    DEFAULT = "cal"\n', encoding="utf-8"
    )

    violations = find_unauthorized_cal_references(fake_apx)

    assert violations == ["apx/db/enums.py:3", "apx/workflows/select.py:1"]


def test_checker_fails_on_a_duplicate_injected_into_a_temp_copy_of_splits_json(
    tmp_path: Path,
) -> None:
    """Prove the negative: copy the real evals/splits.json, inject one sha256 into both
    fatura's train and test lists (a within-split duplicate is impossible without cross-pool
    leakage here, so this single edit trips both find_duplicate_documents-adjacent leakage and
    find_leaks), and confirm validate_splits reports it instead of passing silently."""
    real_data = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    injected_sha = "f" * 64

    mutated = json.loads(json.dumps(real_data))  # deep copy
    mutated["sets"]["fatura"]["train"] = [*mutated["sets"]["fatura"]["train"], injected_sha]
    mutated["sets"]["fatura"]["test"] = [*mutated["sets"]["fatura"]["test"], injected_sha]

    mutated_path = tmp_path / "splits.json"
    mutated_path.write_text(json.dumps(mutated, indent=2) + "\n", encoding="utf-8", newline="\n")

    mutated_splits = load_splits(mutated_path)
    problems = validate_splits(mutated_splits)

    assert problems != []
    assert any(injected_sha in problem for problem in problems)
    # The un-mutated original must still be clean, so the failure above is caused by the
    # injection and not by a pre-existing issue in the real file.
    assert validate_splits(load_splits(SPLITS_PATH)) == []


def test_checker_fails_on_a_duplicate_within_a_single_split_in_a_temp_copy(
    tmp_path: Path,
) -> None:
    """Same proof-of-negative for the other failure mode named in the task: a document listed
    twice inside one split (no cross-pool leakage involved)."""
    real_data = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    any_split_sha = next(iter(real_data["sets"]["fatura"]["train"]), None) or ("e" * 64)

    mutated = json.loads(json.dumps(real_data))
    mutated["sets"]["fatura"]["train"] = [*mutated["sets"]["fatura"]["train"], any_split_sha]

    mutated_path = tmp_path / "splits.json"
    mutated_path.write_text(json.dumps(mutated, indent=2) + "\n", encoding="utf-8", newline="\n")

    problems = find_duplicate_documents(load_splits(mutated_path))

    assert any(any_split_sha in problem and "fatura.train" in problem for problem in problems)
