"""factory/hooks.py: no-op mutation-hook registries that the M8 fraud-typology and M6 attack
suites will register into, so those milestones can inject a fraudulent or adversarial variant of
one invoice's render context or HTML without changing `factory.build`'s generation loop again.
Nothing is registered here: the M2.3 5,000-invoice run applies zero hooks by name, so it always
produces the clean baseline dataset (PRD §10 M2.3, M6.3, M8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ContextMutationHook(Protocol):
    """Mutates one invoice's Jinja2 render context (the per-invoice dict returned by
    `factory.layouts.build_invoice_context`) before it is rendered to HTML."""

    def __call__(self, context: Mapping[str, Any], *, seed: int) -> dict[str, Any]: ...


class HtmlMutationHook(Protocol):
    """Mutates one invoice's already-rendered HTML before `factory.render` turns it into a PDF,
    e.g. to inject hidden text, a zero-width character or a bidi override (M6 attack suite)."""

    def __call__(self, html: str, *, context: Mapping[str, Any], seed: int) -> str: ...


class HookRegistry:
    """A named set of context/HTML mutation hooks for one hook family (fraud typologies or
    attacks). Empty until a later milestone registers into it via `register_context_hook` /
    `register_html_hook`; applying an empty (or unknown) name list is always a no-op."""

    def __init__(self, *, kind: str) -> None:
        self._kind = kind
        self._context_hooks: dict[str, ContextMutationHook] = {}
        self._html_hooks: dict[str, HtmlMutationHook] = {}

    def register_context_hook(self, name: str, hook: ContextMutationHook) -> None:
        if name in self._context_hooks:
            raise ValueError(f"{self._kind} context hook {name!r} is already registered")
        self._context_hooks[name] = hook

    def register_html_hook(self, name: str, hook: HtmlMutationHook) -> None:
        if name in self._html_hooks:
            raise ValueError(f"{self._kind} HTML hook {name!r} is already registered")
        self._html_hooks[name] = hook

    def names(self) -> tuple[str, ...]:
        """Every registered hook name (context or HTML), sorted, for discovery/CLI listing."""
        return tuple(sorted(set(self._context_hooks) | set(self._html_hooks)))

    def apply_context(
        self, context: Mapping[str, Any], *, names: Sequence[str], seed: int
    ) -> dict[str, Any]:
        """Applies each named context hook in order, threading the (possibly mutated) context
        through. A name with no registered context hook is skipped, not an error, so a name that
        only has an HTML hook can still be passed to both `apply_context` and `apply_html`."""
        result: dict[str, Any] = dict(context)
        for name in names:
            hook = self._context_hooks.get(name)
            if hook is not None:
                result = hook(result, seed=seed)
        return result

    def apply_html(
        self, html: str, *, context: Mapping[str, Any], names: Sequence[str], seed: int
    ) -> str:
        """Applies each named HTML hook in order, threading the (possibly mutated) HTML through."""
        result = html
        for name in names:
            hook = self._html_hooks.get(name)
            if hook is not None:
                result = hook(result, context=context, seed=seed)
        return result


FRAUD_TYPOLOGY_HOOKS = HookRegistry(kind="fraud typology")
"""M8 fraud-typology hooks (PRD §10 M8): unverified-IBAN redirects, over-billing, duplicate
invoices and similar. Empty until M8 registers into it."""

ATTACK_HOOKS = HookRegistry(kind="attack")
"""M6 attack-suite hooks (PRD §10 M6.3): hidden text, zero-width characters, bidi overrides,
homoglyph vendor names, fake approval stamps and similar. Empty until M6 registers into it."""
