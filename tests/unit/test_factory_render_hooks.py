"""Unit tests for `factory.hooks`: the module-level fraud-typology registry stays empty until
M8 registers into it (always renders a clean, hook-free dataset), the M6.3 attack registry
never carries anything beyond the categories `attacks.hooks` documents, and a local
`HookRegistry` correctly registers, orders and no-ops context/HTML mutation hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from factory.hooks import ATTACK_HOOKS, FRAUD_TYPOLOGY_HOOKS, HookRegistry

# `ATTACK_HOOKS` is a single process-wide registry: importing `attacks.hooks`
# anywhere in a `pytest` run -- which `tests/unit/test_attacks_*.py` does -- registers these
# names for the rest of that process, whether or not this module happens to run first. This
# test module deliberately never imports `attacks`, so it cannot assert the registry is empty;
# it instead asserts nothing *unexpected* got in, which holds whether `attacks` was imported or
# not (`attacks/hooks.py`'s module docstring lists exactly these eight names).
_KNOWN_M6_3_HTML_OR_CONTEXT_HOOK_NAMES = frozenset(
    {
        "hidden_white_text",
        "tiny_font",
        "off_page",
        "behind_image",
        "iban_swap_instruction",
        "fake_approval_stamp",
        "homoglyph_vendor",
        "line_item_injection",
    }
)


def test_module_level_registries_start_empty_or_known() -> None:
    assert FRAUD_TYPOLOGY_HOOKS.names() == ()
    assert set(ATTACK_HOOKS.names()) <= _KNOWN_M6_3_HTML_OR_CONTEXT_HOOK_NAMES


def test_apply_context_with_no_names_returns_equal_but_independent_dict() -> None:
    registry = HookRegistry(kind="test")
    context = {"a": 1}
    result = registry.apply_context(context, names=(), seed=1)
    assert result == context
    result["a"] = 2
    assert context["a"] == 1


def test_apply_html_with_no_names_is_a_noop() -> None:
    registry = HookRegistry(kind="test")
    assert registry.apply_html("<html></html>", context={}, names=(), seed=1) == "<html></html>"


def test_apply_unknown_hook_name_is_a_noop_not_an_error() -> None:
    registry = HookRegistry(kind="test")
    context = {"a": 1}
    result = registry.apply_context(context, names=("does_not_exist",), seed=1)
    assert result == context


def test_register_and_apply_context_hook() -> None:
    registry = HookRegistry(kind="test")

    def bump(context: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
        return {**context, "seed_seen": seed}

    registry.register_context_hook("bump", bump)
    assert registry.names() == ("bump",)
    result = registry.apply_context({"a": 1}, names=["bump"], seed=42)
    assert result == {"a": 1, "seed_seen": 42}


def test_register_and_apply_html_hook() -> None:
    registry = HookRegistry(kind="test")

    def inject(html: str, *, context: Mapping[str, Any], seed: int) -> str:
        return html.replace("</body>", f"<!-- seed {seed} --></body>")

    registry.register_html_hook("inject", inject)
    result = registry.apply_html("<body></body>", context={}, names=["inject"], seed=7)
    assert result == "<body><!-- seed 7 --></body>"


def test_registering_duplicate_context_hook_name_raises() -> None:
    registry = HookRegistry(kind="test")
    registry.register_context_hook("dup", lambda context, *, seed: dict(context))
    with pytest.raises(ValueError, match="already registered"):
        registry.register_context_hook("dup", lambda context, *, seed: dict(context))


def test_registering_duplicate_html_hook_name_raises() -> None:
    registry = HookRegistry(kind="test")
    registry.register_html_hook("dup", lambda html, *, context, seed: html)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_html_hook("dup", lambda html, *, context, seed: html)


def test_apply_context_threads_multiple_hooks_in_order() -> None:
    registry = HookRegistry(kind="test")

    def _append(context: Mapping[str, Any], *, seed: int, marker: str) -> dict[str, Any]:
        return {**context, "order": [*context.get("order", []), marker]}

    registry.register_context_hook(
        "first", lambda context, *, seed: _append(context, seed=seed, marker="first")
    )
    registry.register_context_hook(
        "second", lambda context, *, seed: _append(context, seed=seed, marker="second")
    )
    result = registry.apply_context({}, names=["first", "second"], seed=1)
    assert result["order"] == ["first", "second"]


def test_names_reports_union_of_context_and_html_hooks_sorted() -> None:
    registry = HookRegistry(kind="test")
    registry.register_context_hook("zeta", lambda context, *, seed: dict(context))
    registry.register_html_hook("alpha", lambda html, *, context, seed: html)
    assert registry.names() == ("alpha", "zeta")
