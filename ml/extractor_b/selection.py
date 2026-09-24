"""Best-checkpoint selection for Extractor B (ADR-0011 §2): highest UAE dev micro field F1 from
generative evals, ties within epsilon broken by lower dev loss, then the earlier step. Export
reads the step chosen here through `best.json`, never the last step."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ml.extractor_b.interfaces import EvalResult


def select_best(evals: Sequence[EvalResult], *, tie_epsilon: float) -> int | None:
    """Step of the best generative eval: highest dev F1, ties within `tie_epsilon` points broken
    by lower dev loss, then the earlier step. None when no generative eval exists. A generative
    eval without its own dev loss borrows the dev-loss eval logged at the same step, if any."""
    generative = [e for e in evals if e.kind == "generative" and e.dev_f1 is not None]
    if not generative:
        return None
    loss_at = {e.step: e.dev_loss for e in evals if e.kind == "dev_loss" and e.dev_loss is not None}

    def dev_loss(e: EvalResult) -> float:
        loss = e.dev_loss if e.dev_loss is not None else loss_at.get(e.step)
        return loss if loss is not None and math.isfinite(loss) else math.inf

    top = max(e.dev_f1 for e in generative if e.dev_f1 is not None)
    tied = [e for e in generative if e.dev_f1 is not None and e.dev_f1 >= top - tie_epsilon]
    return min(tied, key=lambda e: (dev_loss(e), e.step)).step
