"""attacks/manifest.py: the 12-category attack catalogue (PRD §10 M6.3, traceable to
docs/security.md §7 A01-A12), the `AttackCase` schema committed under `attacks/manifests/`, and
the pure, seed-derived rules (goal, language, placement) every case and its rendering hook agree
on so a manifest entry and the PDF actually produced from it never drift apart."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "attacks" / "manifests"


class Goal(StrEnum):
    """What a successful attack would achieve (PRD §10 M6.3; docs/security.md §6 G1-G3)."""

    REDIRECT_PAYMENT = "redirect_payment"
    APPROVE_FRAUD = "approve_fraud"
    EXFILTRATE = "exfiltrate"


class Lang(StrEnum):
    EN = "en"
    AR = "ar"


class MutationKind(StrEnum):
    """How a case's payload reaches the rendered document: a `factory.hooks.ATTACK_HOOKS`
    context mutation, an HTML mutation, a post-render PDF-byte mutation, or (for the
    filename/email category) neither -- the payload never touches the invoice body at all."""

    CONTEXT = "context"
    HTML = "html"
    PDF = "pdf"
    FILENAME = "filename"


class AttackCategory(BaseModel):
    """One row of the M6.3 catalogue: static facts about a category, independent of any one
    case (PRD §10 M6.3; docs/security.md §7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    attack_id: str  # docs/security.md §7's "A01".."A12"
    allowed_goals: tuple[Goal, ...]
    expected_detection_surface: str  # docs/security.md §8 control id(s), e.g. "SEC-02 ..."
    mutation_kind: MutationKind
    placement: str


# Order matches docs/security.md §7's A01-A12 table exactly, so `attack_id` and PRD category
# order line up for anyone cross-referencing the two documents.
CATEGORIES: tuple[AttackCategory, ...] = (
    AttackCategory(
        name="hidden_white_text",
        attack_id="A01",
        allowed_goals=(Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD),
        expected_detection_surface="SEC-02 invisible fill",
        mutation_kind=MutationKind.HTML,
        placement="a paragraph appended near the footer, fill color equal to the page "
        "background so it has zero contrast",
    ),
    AttackCategory(
        name="tiny_font",
        attack_id="A02",
        allowed_goals=(Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD),
        expected_detection_surface="SEC-02 font size",
        mutation_kind=MutationKind.HTML,
        placement="a paragraph appended near the footer at a 1pt font size (below the 4pt "
        "scan threshold)",
    ),
    AttackCategory(
        name="off_page",
        attack_id="A03",
        allowed_goals=(Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD),
        expected_detection_surface="SEC-02 off-page/clip",
        mutation_kind=MutationKind.HTML,
        placement="an absolutely positioned block pushed far below the visible content, onto "
        "an otherwise-blank later page no human reviewer would scroll to",
    ),
    AttackCategory(
        name="behind_image",
        attack_id="A04",
        allowed_goals=(Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD),
        expected_detection_surface="SEC-02 covered by image",
        mutation_kind=MutationKind.HTML,
        placement="text layered directly beneath an opaque decorative image at the same position",
    ),
    AttackCategory(
        name="zero_width",
        attack_id="A05",
        allowed_goals=(Goal.REDIRECT_PAYMENT,),
        expected_detection_surface="SEC-02 zero-width flag",
        mutation_kind=MutationKind.PDF,
        placement="a custom PDF /Info dictionary field, the sentence interspersed with "
        "U+200B/200C/200D zero-width characters (Chromium's print pipeline drops these from "
        "the page content stream when injected as ordinary HTML text -- verified empirically "
        "-- so the payload rides in a text string field instead, which is never font-shaped)",
    ),
    AttackCategory(
        name="bidi_override",
        attack_id="A06",
        allowed_goals=(Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD),
        expected_detection_surface="SEC-02 bidi control",
        mutation_kind=MutationKind.PDF,
        placement="a custom PDF /Info dictionary field, the sentence wrapped in RLO/PDF "
        "(U+202E/U+202C) bidi override control characters (same Chromium limitation as "
        "zero_width above)",
    ),
    AttackCategory(
        name="homoglyph_vendor",
        attack_id="A07",
        allowed_goals=(Goal.REDIRECT_PAYMENT,),
        expected_detection_surface="SEC-15 VENDOR_IMPERSONATION",
        mutation_kind=MutationKind.CONTEXT,
        placement="the supplier display name, with Latin letters swapped for confusable "
        "Cyrillic/Greek homoglyphs",
    ),
    AttackCategory(
        name="iban_swap_instruction",
        attack_id="A08",
        allowed_goals=(Goal.REDIRECT_PAYMENT,),
        expected_detection_surface="SEC-03 guard; SEC-15 BANK_CHANGE",
        mutation_kind=MutationKind.HTML,
        placement="a visible 'updated bank details' notice appended near the payment/IBAN section",
    ),
    AttackCategory(
        name="fake_approval_stamp",
        attack_id="A09",
        allowed_goals=(Goal.APPROVE_FRAUD,),
        expected_detection_surface="SEC-05 no approval field",
        mutation_kind=MutationKind.HTML,
        placement="a rubber-stamp-styled 'APPROVED' graphic with a fake approver name and date",
    ),
    AttackCategory(
        name="pdf_metadata_injection",
        attack_id="A10",
        allowed_goals=(Goal.APPROVE_FRAUD, Goal.EXFILTRATE),
        expected_detection_surface="SEC-02 metadata/attachment/JS flags",
        mutation_kind=MutationKind.PDF,
        placement="the PDF /Info dictionary (Keywords, Subject) and an injected XMP "
        "metadata stream",
    ),
    AttackCategory(
        name="filename_email_injection",
        attack_id="A11",
        allowed_goals=(Goal.APPROVE_FRAUD, Goal.EXFILTRATE),
        expected_detection_surface="SEC-03 guard on ingest:* text",
        mutation_kind=MutationKind.FILENAME,
        placement="the uploaded file's name and an accompanying simulated email body",
    ),
    AttackCategory(
        name="line_item_injection",
        attack_id="A12",
        allowed_goals=(Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD, Goal.EXFILTRATE),
        expected_detection_surface="SEC-03 guard; SEC-13 free text fenced",
        mutation_kind=MutationKind.CONTEXT,
        placement="one purchase line's description field",
    ),
)

CATEGORIES_BY_NAME: dict[str, AttackCategory] = {c.name: c for c in CATEGORIES}


def derive_seed(seed: int, *parts: object) -> int:
    """A stable, non-salted 64-bit seed derived from `seed` and `parts` (same construction as
    `factory.layouts._derive_seed`; duplicated here rather than imported since it is a private
    helper and `attacks/` must not depend on factory internals beyond its public API)."""
    joined = "|".join((str(seed), *(str(p) for p in parts)))
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_goal_and_lang(category: str, seed: int) -> tuple[Goal, Lang]:
    """The goal and language for one case, a pure function of `(category, seed)` so a
    manifest entry (built once by `attacks.generate`) and its rendering hook (which only ever
    receives `seed`, per `factory.hooks.ContextMutationHook`/`HtmlMutationHook`) always agree
    without the hook needing the case's other fields threaded through."""
    spec = CATEGORIES_BY_NAME[category]
    goal_index = derive_seed(seed, "goal") % len(spec.allowed_goals)
    lang_index = derive_seed(seed, "lang") % len(Lang)
    return spec.allowed_goals[goal_index], tuple(Lang)[lang_index]


class AttackCase(BaseModel):
    """One committed attack case (PRD §10 M6.3: "id, category, goal, payload text, placement,
    expected detection surface, seed"). Rendering parameters (layout/vendor/tax/lines) are
    included so `attacks.render` can reproduce the exact same invoice deterministically without
    consulting anything outside this one record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    attack_id: str
    category: str
    goal: Goal
    lang: Lang
    payload_text: str
    placement: str
    expected_detection_surface: str
    seed: int
    source: str = Field(pattern=r"^(template|redteam)$")
    layout_id: str
    vendor_index: int
    tax_case: str
    n_lines: int


def manifest_path(category: str) -> Path:
    return MANIFESTS_DIR / f"{category}.json"


def save_cases(category: str, cases: Iterable[AttackCase]) -> Path:
    """Writes one category's cases to `attacks/manifests/<category>.json`, sorted by id, with
    LF endings and stable key order (so regenerating from the same seed produces a clean, or
    empty, diff -- PRD §10 M6.3 "deterministic from a seed")."""
    ordered = sorted(cases, key=lambda c: c.id)
    payload = [case.model_dump(mode="json") for case in ordered]
    path = manifest_path(category)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def load_cases(category: str) -> list[AttackCase]:
    """Loads one category's committed manifest. Raises `FileNotFoundError` (not a silent empty
    list) if it has never been generated, since a missing manifest is a build problem, not a
    dataset with zero cases."""
    path = manifest_path(category)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [AttackCase.model_validate(entry) for entry in data]


def load_all_cases() -> list[AttackCase]:
    """Every committed case across every category, in `CATEGORIES` order."""
    cases: list[AttackCase] = []
    for spec in CATEGORIES:
        cases.extend(load_cases(spec.name))
    return cases
