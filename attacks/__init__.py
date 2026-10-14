"""attacks: the adversarial invoice suite
(attack suite row) -- deterministic, committed manifests plus generators that render them into UAE
invoice PDFs at test/run time for the hidden-content scanner and the end-to-end attack eval."""

from __future__ import annotations

# Populates `factory.hooks.ATTACK_HOOKS` as a side effect: importing any part of
# `attacks` must be enough for `factory.hooks.ATTACK_HOOKS.apply_context`/`apply_html` to know
# every attack category, so callers never have to remember a separate registration step.
from attacks import hooks as _hooks  # noqa: F401
