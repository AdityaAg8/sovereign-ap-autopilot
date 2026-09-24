"""attacks: the M6.3 adversarial invoice suite (PRD §10 M6.3, §4 red-team model, §5 attack
suite row) -- deterministic, committed manifests plus generators that render them into UAE
invoice PDFs at test/run time for M6.2's scanner and M6.6's end-to-end attack-success eval."""

from __future__ import annotations

# Populates `factory.hooks.ATTACK_HOOKS` as a side effect (PRD §10 M6.3): importing any part of
# `attacks` must be enough for `factory.hooks.ATTACK_HOOKS.apply_context`/`apply_html` to know
# every M6.3 category, so callers never have to remember a separate registration step.
from attacks import hooks as _hooks  # noqa: F401
