"""Unit tests for `attacks.textsource`: `TemplateTextSource` determinism and coverage of every
category/goal/language combination the manifest builder can request, plus the standalone text
utilities (`homoglyph_spoof`, `zero_width_obfuscate`, `bidi_wrap`) each attack category's hook
or PDF mutation relies on ("template-based variants now (EN + AR)")."""

from __future__ import annotations

import pytest

from attacks.manifest import (
    CATEGORIES,
    AttackCategory,
    Goal,
    Lang,
    derive_goal_and_lang,
    derive_seed,
)
from attacks.textsource import (
    BIDI_PDF,
    BIDI_RLO,
    TemplateTextSource,
    bidi_wrap,
    homoglyph_spoof,
    zero_width_obfuscate,
)

_SOURCE = TemplateTextSource()


@pytest.mark.parametrize("spec", CATEGORIES, ids=lambda spec: spec.name)
def test_template_text_source_covers_every_allowed_goal_and_language(spec: AttackCategory) -> None:
    for goal in spec.allowed_goals:
        for lang in (Lang.EN, Lang.AR):
            for index in range(3):
                seed = derive_seed(1, spec.name, index)
                text = _SOURCE.generate(category=spec.name, goal=goal, lang=lang, seed=seed)
                assert isinstance(text, str)
                assert text.strip() != ""


def test_template_text_source_is_deterministic() -> None:
    seed = derive_seed(9, "hidden_white_text", 4)
    goal, lang = derive_goal_and_lang("hidden_white_text", seed)
    a = _SOURCE.generate(category="hidden_white_text", goal=goal, lang=lang, seed=seed)
    b = _SOURCE.generate(category="hidden_white_text", goal=goal, lang=lang, seed=seed)
    assert a == b


def test_template_text_source_rejects_unknown_category_goal_pair() -> None:
    with pytest.raises(ValueError, match="no template payload"):
        _SOURCE.generate(category="fake_approval_stamp", goal=Goal.EXFILTRATE, lang=Lang.EN, seed=1)


def test_homoglyph_spoof_always_changes_a_latin_name() -> None:
    for seed in range(20):
        spoofed = homoglyph_spoof("Al Falah Trading LLC", seed=seed)
        assert spoofed != "Al Falah Trading LLC"
        assert len(spoofed) == len("Al Falah Trading LLC")


def test_homoglyph_spoof_is_a_noop_on_text_with_no_confusable_letters() -> None:
    assert homoglyph_spoof("مؤسسة", seed=1) == "مؤسسة"


def test_homoglyph_spoof_is_deterministic() -> None:
    assert homoglyph_spoof("Marina Bay LLC", seed=7) == homoglyph_spoof("Marina Bay LLC", seed=7)


def test_zero_width_obfuscate_interleaves_only_known_zero_width_chars() -> None:
    result = zero_width_obfuscate("abc", seed=1)
    assert len(result) == 6  # one zero-width char after each of the 3 visible chars
    visible = result[0::2]
    zw = result[1::2]
    assert visible == "abc"
    assert all(ch in (chr(0x200B), chr(0x200C), chr(0x200D)) for ch in zw)


def test_zero_width_obfuscate_is_deterministic() -> None:
    assert zero_width_obfuscate("hello", seed=3) == zero_width_obfuscate("hello", seed=3)


def test_bidi_wrap_adds_rlo_and_pdf_around_the_text() -> None:
    wrapped = bidi_wrap("hello")
    assert wrapped == BIDI_RLO + "hello" + BIDI_PDF
    assert wrapped[0] == chr(0x202E)
    assert wrapped[-1] == chr(0x202C)
