"""Builds the injection-guard dataset: positives are the attack-suite adversarial
payloads plus public prompt-injection data, negatives are public benign text plus invoice hard
negatives (EN/AR). Splits are assigned per template family so no family spans train/cal/test."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from attacks.manifest import AttackCase, derive_seed, load_all_cases
from attacks.textsource import template_variant_count
from factory.ids import generate_ae_iban
from ml.guard.mine import MinedText, drop_attack_overlap, template_key

GuardSplit = Literal["train", "cal", "test"]
SPLITS: tuple[GuardSplit, ...] = ("train", "cal", "test")

# Categories whose payload is not instruction text (a spoofed vendor name): the guard is not
# the control for them (SEC-15 is), so they are neither positives nor negatives here.
EXCLUDED_ATTACK_CATEGORIES = frozenset({"homoglyph_vendor"})

# Order families are dealt in within each group; 3 of every 5 go to train. A group with two
# families gets one train and one test family, so every multi-family category is tested.
_DEAL_PATTERN: tuple[GuardSplit, ...] = ("train", "test", "cal", "train", "train")
BUILD_SALT = "m6.4-guard-v1"
# Most numeric variants kept per mined line template, so no one line dominates a split.
MINED_PER_TEMPLATE = 50


class SplitLeakError(ValueError):
    """A family or a normalized text appears in more than one split."""


@dataclass(frozen=True)
class GuardExample:
    text: str
    label: int  # 1 = injection
    source: Literal["attack", "public", "hard_negative"]
    family: str
    group: str  # stratification group: attack category, hardneg language or public split
    lang: str
    split: GuardSplit


def normalize(text: str) -> str:
    """NFKC, casefolded, whitespace-collapsed form used for duplicate/leak detection."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def deal_families(
    groups: Mapping[str, Sequence[str]], salt: str = BUILD_SALT
) -> dict[str, GuardSplit]:
    """Assigns each family to one split. Within a group, families are ordered by a salted hash
    (deterministic, independent of input order) and dealt by `_DEAL_PATTERN`."""
    assignment: dict[str, GuardSplit] = {}
    for group in sorted(groups):
        families = sorted(set(groups[group]), key=lambda f: _hash(salt, group, f))
        for i, family in enumerate(families):
            if family in assignment:
                raise SplitLeakError(f"family {family!r} appears in two groups")
            assignment[family] = _DEAL_PATTERN[i % len(_DEAL_PATTERN)]
    return assignment


def attack_family(case: AttackCase) -> str:
    """The template the payload was generated from: (category, goal, lang, variant index), the
    same derivation `attacks.textsource.TemplateTextSource` uses."""
    n_variants = template_variant_count(case.category, case.goal, case.lang)
    variant = derive_seed(case.seed, "variant") % n_variants if n_variants else 0
    return f"attack:{case.category}:{case.goal.value}:{case.lang.value}:v{variant}"


# --- invoice hard negatives -------------------------------------------------------------
# Legitimate AP text that shares surface features with the attacks (bank details, "pay to",
# approvals, remittance wording). Each template is its own family.
_BANKS = ("Emirates NBD", "First Abu Dhabi Bank", "Mashreq Bank", "ADCB", "Dubai Islamic Bank")
_BANKS_AR = ("بنك الإمارات دبي الوطني", "بنك أبوظبي الأول", "بنك المشرق", "بنك دبي الإسلامي")
_HARD_NEGATIVES_EN: tuple[str, ...] = (
    "Please pay to the account on file.",
    "Payment to be made to the account on file within 30 days of the invoice date.",
    "Bank details: {bank}, IBAN {iban}, SWIFT {swift}. Please quote invoice {ref} as reference.",
    "Kindly remit payment to the IBAN shown above and quote reference {ref}.",
    "Our bank details remain unchanged. Payment terms: net 30.",
    "Account name: {company}. Bank: {bank}. IBAN: {iban}.",
    "Please transfer the total amount due to {bank}, account {iban}, before {day}/03/2026.",
    "This invoice was issued against purchase order PO-{ref} approved by your procurement team.",
    "For any questions regarding this invoice, contact accounts@{domain}.",
    "Late payments may incur a fee of 2% per month on the outstanding balance.",
    "Consulting services for March 2026 as per agreement {ref}.",
    "Thank you for your business. Payment is due upon receipt.",
    "Remittance advice can be sent to finance@{domain} after payment.",
    "Goods remain the property of {company} until paid in full.",
)
_HARD_NEGATIVES_AR: tuple[str, ...] = (
    "يرجى الدفع إلى الحساب المسجل لدينا.",
    "يتم الدفع إلى الحساب المسجل خلال 30 يومًا من تاريخ الفاتورة.",
    "البيانات المصرفية: {bank_ar}، رقم الآيبان {iban}، يرجى ذكر رقم الفاتورة {ref} كمرجع.",
    "يرجى تحويل المبلغ إلى رقم الآيبان المذكور أعلاه مع ذكر المرجع {ref}.",
    "بياناتنا المصرفية لم تتغير. شروط الدفع: 30 يومًا.",
    "اسم الحساب: {company}. البنك: {bank_ar}. الآيبان: {iban}.",
    "صدرت هذه الفاتورة بناءً على أمر الشراء {ref} المعتمد من قسم المشتريات لديكم.",
    "لأي استفسار بخصوص هذه الفاتورة يرجى التواصل مع accounts@{domain}.",
    "خدمات استشارية لشهر مارس 2026 وفقًا للاتفاقية {ref}.",
    "شكرًا لتعاملكم معنا. تستحق الفاتورة الدفع عند الاستلام.",
    "تظل البضائع ملكًا لـ {company} حتى السداد الكامل.",
)
_COMPANIES = ("Gulf Horizon Trading LLC", "Al Noor Supplies FZE", "Desert Palm Services LLC")
_DOMAINS = ("gulfhorizon.ae", "alnoor-supplies.ae", "desertpalm.ae")


def hard_negative_examples(per_template: int = 12, seed: int = 64) -> list[tuple[str, str, str]]:
    """(text, family, lang) for each filled hard-negative template, deterministic from `seed`."""
    out: list[tuple[str, str, str]] = []
    for lang, pool in (("en", _HARD_NEGATIVES_EN), ("ar", _HARD_NEGATIVES_AR)):
        for index, template in enumerate(pool):
            rng = random.Random(derive_seed(seed, lang, index))
            seen: set[str] = set()
            for _ in range(per_template):
                text = template.format(
                    bank=rng.choice(_BANKS),
                    bank_ar=rng.choice(_BANKS_AR),
                    iban=generate_ae_iban(rng, set()),
                    swift=rng.choice(("EBILAEAD", "NBADAEAA", "BOMLAEAD")),
                    ref=rng.randint(100000, 999999),
                    day=rng.randint(1, 28),
                    company=rng.choice(_COMPANIES),
                    domain=rng.choice(_DOMAINS),
                )
                if text not in seen:  # fixed templates have one distinct fill
                    seen.add(text)
                    out.append((text, f"hardneg:{lang}:t{index}", lang))
    return out


def _read_public(path: Path) -> list[tuple[str, int]]:
    table = pq.read_table(path, columns=["text", "label"])
    rows = table.to_pylist()
    return [(str(r["text"]), int(r["label"])) for r in rows]


def build_examples(
    public_train: Path | None,
    public_test: Path | None,
    cases: Iterable[AttackCase] | None = None,
    mined: Sequence[MinedText] = (),
) -> list[GuardExample]:
    """Assembles every example with its split. Public data keeps its upstream test split; its
    train split is divided into train/cal by text hash. Duplicated texts are dropped from the
    lower-priority split (test > cal > train) so no text leaks across splits. Mined invoice
    text is grouped by its miner template (one family per entity field, per line item across
    EN/AR/bilingual, else per leading-words key; see `ml.guard.mine`), capped at
    `MINED_PER_TEMPLATE` per template."""
    attack_cases = [
        c
        for c in (load_all_cases() if cases is None else cases)
        if c.category not in EXCLUDED_ATTACK_CATEGORIES
    ]
    hardnegs = hard_negative_examples()
    handwritten = {template_key(text) for text, _, _ in hardnegs}
    mined_kept: list[MinedText] = []
    per_template: dict[str, int] = defaultdict(int)
    seen_text: set[str] = set()
    for m in drop_attack_overlap(mined, (c.payload_text for c in attack_cases)):
        if m.template in handwritten or normalize(m.text) in seen_text:
            continue
        if per_template[m.template] >= MINED_PER_TEMPLATE:
            continue
        per_template[m.template] += 1
        seen_text.add(normalize(m.text))
        mined_kept.append(m)

    groups: dict[str, list[str]] = defaultdict(list)
    for case in attack_cases:
        groups[case.category].append(attack_family(case))
    for _, family, lang in hardnegs:
        groups[f"hardneg:{lang}"].append(family)
    for m in mined_kept:
        groups["mined"].append(mined_family(m))
    assignment = deal_families(groups)

    examples: list[GuardExample] = [
        GuardExample(
            text=c.payload_text,
            label=1,
            source="attack",
            family=attack_family(c),
            group=c.category,
            lang=c.lang.value,
            split=assignment[attack_family(c)],
        )
        for c in attack_cases
    ]
    examples += [
        GuardExample(text, 0, "hard_negative", fam, f"hardneg:{lang}", lang, assignment[fam])
        for text, fam, lang in hardnegs
    ]
    examples += [
        GuardExample(
            m.text,
            0,
            "hard_negative",
            mined_family(m),
            "mined",
            m.lang,
            assignment[mined_family(m)],
        )
        for m in mined_kept
    ]
    for path, upstream in ((public_train, "train"), (public_test, "test")):
        if path is None:
            continue
        for text, label in _read_public(path):
            family = f"public:{_hash(normalize(text))[:16]}"
            if upstream == "test":
                split: GuardSplit = "test"
            else:
                split = "cal" if int(_hash(BUILD_SALT, family), 16) % 5 == 0 else "train"
            examples.append(
                GuardExample(text, label, "public", family, f"public:{upstream}", "", split)
            )
    return dedupe(examples)


def mined_family(m: MinedText) -> str:
    if m.template.startswith("field:"):
        return f"mined:{m.template}"
    return f"mined:{_hash(m.template)[:16]}"


def dedupe(examples: Sequence[GuardExample]) -> list[GuardExample]:
    """Keeps one copy per normalized text, preferring test, then cal, then train."""
    priority = {"test": 0, "cal": 1, "train": 2}
    kept: dict[str, GuardExample] = {}
    for ex in sorted(examples, key=lambda e: priority[e.split]):
        kept.setdefault(normalize(ex.text), ex)
    return [ex for ex in examples if kept.get(normalize(ex.text)) is ex]


def check_no_leak(examples: Sequence[GuardExample]) -> None:
    """Raises `SplitLeakError` if any family or normalized text occurs in two splits, or if two
    hard negatives that differ only in digits (one line template) land in different splits."""
    checks: tuple[tuple[str, Callable[[GuardExample], str | None]], ...] = (
        ("family", lambda e: e.family),
        ("text", lambda e: normalize(e.text)),
        ("template", lambda e: template_key(e.text) if e.source == "hard_negative" else None),
    )
    for key_name, key in checks:
        seen: dict[str, str] = {}
        for ex in examples:
            k = key(ex)
            if k is None:
                continue
            if seen.setdefault(k, ex.split) != ex.split:
                raise SplitLeakError(f"{key_name} {k[:80]!r} in both {seen[k]} and {ex.split}")


def write_splits(examples: Sequence[GuardExample], out_dir: Path) -> dict[str, int]:
    """Writes `{train,cal,test}.jsonl` (LF, UTF-8, stable order) after the leak check; returns
    per-split counts."""
    check_no_leak(examples)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = [asdict(e) for e in examples if e.split == split]
        counts[split] = len(rows)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def read_split(path: Path) -> list[GuardExample]:
    with path.open(encoding="utf-8") as fh:
        return [GuardExample(**json.loads(line)) for line in fh if line.strip()]
