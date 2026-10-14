"""Leakage gate for I5 ("eval, cal and test data never reach training") and ADR-0005:
validates the committed evals/splits.json has no cross-pool sha256 overlap and no duplicate
within a split, checks no real hold-out (CORD, DocILE val, ExtractBench; ADR-0019 §4.5) sits in an
ml/ training manifest, greps apx/ to confirm only apx/trust references the cal split (quoted "cal"
or a `*Split.CAL` member; the one exempt line is the DB enum declaring the value), and proves the
checkers are not no-ops with planted violations. No network, no GPU."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from evals.splits import (
    SPLITS_PATH,
    SetName,
    SplitsFile,
    find_duplicate_documents,
    load_splits,
    validate_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
APX_ROOT = REPO_ROOT / "apx"

# A "cal" split name used as a literal string (e.g. `split == "cal"` or a dict key), not the
# substring inside a longer word like "calibrator" or "calendar" ("Only apx/trust
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


# --- ADR-0019 §4.5: real hold-outs never reach an ml/ training manifest ------------------------

DATA_ROOT = REPO_ROOT / "data"
HOLDOUT_SPLITS: tuple[tuple[SetName, str], ...] = (
    ("cord", "val"),
    ("cord", "test"),
    ("docile", "val"),
    ("extractbench", "eval"),
    ("fatura", "test"),
    ("uae", "cal"),
    ("uae", "test"),
)


def holdout_shas(splits: SplitsFile, data_root: Path = DATA_ROOT) -> set[str]:
    """Every hold-out document sha256: the splits.json hold-outs plus, when DocILE is fetched,
    the sha256 of every DocILE val PDF (its manifest records archives, not documents)."""
    shas: set[str] = set()
    for set_name, split in HOLDOUT_SPLITS:
        shas.update(splits.sets.get(set_name, {}).get(split, []))
    val_list = data_root / "docile" / "val.json"
    if val_list.is_file():
        for doc_id in json.loads(val_list.read_text(encoding="utf-8")):
            pdf = data_root / "docile" / "pdfs" / f"{doc_id}.pdf"
            if pdf.is_file():
                shas.add(hashlib.sha256(pdf.read_bytes()).hexdigest())
    return shas


def ml_manifest_leaks(holdouts: set[str], ml_root: Path) -> list[str]:
    """`path:line sha` for each training-manifest row whose document is a hold-out."""
    leaks: list[str] = []
    for manifest in sorted(ml_root.rglob("manifest.jsonl")) if ml_root.is_dir() else []:
        for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            sha = json.loads(line).get("document_sha256")
            if sha in holdouts:
                leaks.append(f"{manifest}:{lineno} {sha}")
    return leaks


def test_cord_splits_are_registered() -> None:
    splits = load_splits()
    assert len(splits.sets["cord"]["test"]) == 100
    assert len(splits.sets["cord"]["val"]) == 100
    assert splits.sets["cord"]["train"], "CORD train is a train set (ADR-0019 §4.5)"


def test_real_holdouts_never_appear_in_ml_training_manifests() -> None:
    holdouts = holdout_shas(load_splits())
    assert len(holdouts) >= 200 + 252, "CORD test/val and ExtractBench must be hold-outs"
    assert ml_manifest_leaks(holdouts, DATA_ROOT / "ml") == []


def test_ml_manifest_leak_check_catches_a_planted_holdout(tmp_path: Path) -> None:
    splits = load_splits()
    planted = sorted(splits.sets["cord"]["test"])[0]
    train_sha = sorted(splits.sets["cord"]["train"])[0]
    run = tmp_path / "ml" / "extractor_b"
    run.mkdir(parents=True)
    rows = [{"document_sha256": train_sha}, {"document_sha256": planted}]
    (run / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )

    leaks = ml_manifest_leaks(holdout_shas(splits, tmp_path), tmp_path / "ml")

    assert len(leaks) == 1
    assert planted in leaks[0]


def test_docile_val_pdfs_count_as_holdouts_when_fetched(tmp_path: Path) -> None:
    (tmp_path / "docile" / "pdfs").mkdir(parents=True)
    (tmp_path / "docile" / "val.json").write_text('["d1"]', encoding="utf-8")
    (tmp_path / "docile" / "pdfs" / "d1.pdf").write_bytes(b"%PDF-docile")

    shas = holdout_shas(load_splits(), tmp_path)

    assert hashlib.sha256(b"%PDF-docile").hexdigest() in shas
