"""Unit tests for `attacks.manifest` and `attacks.generate`: the 12-category catalogue shape,
`derive_seed`/`derive_goal_and_lang` determinism, `AttackCase` schema validation, and the
committed manifests under `attacks/manifests/` meeting the suite's case-count acceptance criteria
("≥ 300 cases across ≥ 12 categories", "≥ 20 cases" per category)."""

from __future__ import annotations

import pytest

from attacks.generate import CASES_PER_CATEGORY, build_case, build_category_cases
from attacks.manifest import (
    CATEGORIES,
    CATEGORIES_BY_NAME,
    AttackCase,
    Goal,
    Lang,
    derive_goal_and_lang,
    derive_seed,
    load_all_cases,
    load_cases,
)
from attacks.textsource import TemplateTextSource

_SOURCE = TemplateTextSource()


def test_at_least_twelve_categories() -> None:
    assert len(CATEGORIES) >= 12
    assert len(CATEGORIES_BY_NAME) == len(CATEGORIES)


def test_category_attack_ids_are_unique_and_match_docs_security_md() -> None:
    ids = [spec.attack_id for spec in CATEGORIES]
    assert len(ids) == len(set(ids))
    assert ids == [f"A{i:02d}" for i in range(1, len(CATEGORIES) + 1)]


def test_derive_seed_is_deterministic_and_salt_sensitive() -> None:
    a = derive_seed(6300, "hidden_white_text", 0)
    b = derive_seed(6300, "hidden_white_text", 0)
    c = derive_seed(6300, "hidden_white_text", 1)
    assert a == b
    assert a != c


def test_derive_goal_and_lang_only_returns_allowed_goals() -> None:
    for spec in CATEGORIES:
        for index in range(10):
            seed = derive_seed(1, spec.name, index)
            goal, lang = derive_goal_and_lang(spec.name, seed)
            assert goal in spec.allowed_goals
            assert lang in (Lang.EN, Lang.AR)


def test_derive_goal_and_lang_is_deterministic() -> None:
    seed = derive_seed(42, "iban_swap_instruction", 3)
    assert derive_goal_and_lang("iban_swap_instruction", seed) == derive_goal_and_lang(
        "iban_swap_instruction", seed
    )


def test_build_case_is_deterministic_for_the_same_inputs() -> None:
    a = build_case("hidden_white_text", 5, text_source=_SOURCE, source_label="template")
    b = build_case("hidden_white_text", 5, text_source=_SOURCE, source_label="template")
    assert a == b


def test_build_case_rejects_unknown_category() -> None:
    with pytest.raises(KeyError):
        build_case("not_a_real_category", 0, text_source=_SOURCE, source_label="template")


def test_build_category_cases_ids_are_sequential_and_prefixed() -> None:
    cases = build_category_cases(
        "tiny_font", text_source=_SOURCE, source_label="template", n_cases=5
    )
    assert [c.id for c in cases] == [f"A02-{i:03d}" for i in range(1, 6)]
    assert all(c.category == "tiny_font" for c in cases)
    assert all(c.attack_id == "A02" for c in cases)


def test_build_category_cases_n_lines_within_factorys_valid_range() -> None:
    for spec in CATEGORIES:
        for case in build_category_cases(
            spec.name, text_source=_SOURCE, source_label="template", n_cases=8
        ):
            assert 1 <= case.n_lines <= 40


def test_attack_case_rejects_unknown_fields() -> None:
    valid = build_case("tiny_font", 0, text_source=_SOURCE, source_label="template")
    payload = valid.model_dump(mode="json")
    payload["unexpected_field"] = "nope"
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError, not worth importing
        AttackCase.model_validate(payload)


def test_attack_case_source_is_restricted_to_template_or_redteam() -> None:
    valid = build_case("tiny_font", 0, text_source=_SOURCE, source_label="template")
    payload = valid.model_dump(mode="json")
    payload["source"] = "made_up"
    with pytest.raises(Exception):  # noqa: B017
        AttackCase.model_validate(payload)


# --- the committed manifests themselves --------------------------------------------------------


def test_every_category_has_a_committed_manifest_with_at_least_20_cases() -> None:
    for spec in CATEGORIES:
        cases = load_cases(spec.name)
        assert len(cases) >= 20, f"{spec.name} has only {len(cases)} cases"


def test_suite_has_at_least_300_cases_total() -> None:
    assert len(load_all_cases()) >= 300


def test_committed_case_ids_are_globally_unique() -> None:
    ids = [case.id for case in load_all_cases()]
    assert len(ids) == len(set(ids))


def test_committed_manifest_matches_a_fresh_regeneration() -> None:
    """The exact acceptance test for "deterministic from a seed": regenerating a category from
    scratch must reproduce the committed manifest byte-for-byte (field-for-field, since this
    compares parsed models rather than raw JSON text)."""
    for spec in CATEGORIES:
        committed = load_cases(spec.name)
        fresh = build_category_cases(
            spec.name,
            text_source=TemplateTextSource(),
            source_label="template",
            n_cases=CASES_PER_CATEGORY,
        )
        assert committed == fresh, f"{spec.name} manifest drifted from a fresh regeneration"


def test_committed_cases_cover_more_than_one_goal_where_more_than_one_is_allowed() -> None:
    for spec in CATEGORIES:
        if len(spec.allowed_goals) < 2:
            continue
        goals_seen = {case.goal for case in load_cases(spec.name)}
        assert len(goals_seen) >= 2, f"{spec.name} only ever picked one goal across its cases"


def test_committed_cases_cover_both_languages() -> None:
    for spec in CATEGORIES:
        langs_seen = {case.lang for case in load_cases(spec.name)}
        assert langs_seen == {Lang.EN, Lang.AR}, f"{spec.name} is missing a language: {langs_seen}"


def test_all_goals_are_one_of_the_prd_named_three() -> None:
    for case in load_all_cases():
        assert case.goal in (Goal.REDIRECT_PAYMENT, Goal.APPROVE_FRAUD, Goal.EXFILTRATE)
