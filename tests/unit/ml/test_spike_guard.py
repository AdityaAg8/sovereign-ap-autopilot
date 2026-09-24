"""Tests for the pure Extractor B spike guard (ADR-0011 §5): non-finite loss and gradient skips,
the consecutive and windowed stop rules, spike detection armed after warmup, dev-loss blow-up and
a JSON round-trip of the state."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from ml.extractor_b.guards import dev_loss_diverged, is_loss_spike, spike_guard_observe
from ml.extractor_b.interfaces import GuardAction, GuardConfig, LoopState, SpikeGuardState

CFG = GuardConfig()


def observe(
    state: SpikeGuardState, step: int, loss: float, grad: float | None = 1.0
) -> tuple[SpikeGuardState, GuardAction]:
    return spike_guard_observe(
        state, step=step, loss=loss, grad_norm=grad, in_warmup=False, cfg=CFG
    )


def test_nan_loss_skips_microbatch() -> None:
    state, action = observe(SpikeGuardState(), 5, math.nan, None)
    assert action is GuardAction.SKIP_MICROBATCH
    assert state.total_skipped_steps == 1 and state.skip_steps == (5,)


def test_inf_grad_skips_step() -> None:
    state, action = observe(SpikeGuardState(), 5, 1.0, math.inf)
    assert action is GuardAction.SKIP_STEP
    assert state.consecutive_skips == 1


def test_nonfinite_is_skipped_even_in_warmup() -> None:
    _, action = spike_guard_observe(
        SpikeGuardState(), step=1, loss=math.nan, grad_norm=None, in_warmup=True, cfg=CFG
    )
    assert action is GuardAction.SKIP_MICROBATCH


def test_three_consecutive_skipped_steps_stop() -> None:
    state = SpikeGuardState()
    actions = []
    for step in (10, 11, 12):
        state, action = observe(state, step, 1.0, math.nan)
        actions.append(action)
    assert actions == [GuardAction.SKIP_STEP, GuardAction.SKIP_STEP, GuardAction.STOP]


def test_clean_step_resets_consecutive_count() -> None:
    state = SpikeGuardState()
    state, _ = observe(state, 10, 1.0, math.nan)
    state, _ = observe(state, 11, 1.0, math.nan)
    state, action = observe(state, 12, 1.0, 0.5)
    assert action is GuardAction.OK and state.consecutive_skips == 0
    state, action = observe(state, 13, 1.0, math.nan)
    assert action is GuardAction.SKIP_STEP


def test_six_skips_in_a_hundred_steps_stop() -> None:
    state = SpikeGuardState()
    actions = []
    for step in (10, 30, 50, 70, 90, 105):
        state, action = observe(state, step, 1.0, math.inf)
        state, _ = observe(state, step + 1, 1.0, 0.5)
        actions.append(action)
    assert actions[:5] == [GuardAction.SKIP_STEP] * 5
    assert actions[5] is GuardAction.STOP


def test_old_skips_leave_the_window() -> None:
    state = SpikeGuardState()
    for step in (10, 30, 50, 70, 90, 120):
        state, action = observe(state, step, 1.0, math.inf)
        state, _ = observe(state, step + 1, 1.0, 0.5)
    assert action is GuardAction.SKIP_STEP
    assert 10 not in state.skip_steps and state.total_skipped_steps == 6


def test_micro_and_step_skip_at_same_step_count_once() -> None:
    state, _ = observe(SpikeGuardState(), 7, math.nan, None)
    state, action = observe(state, 7, math.nan, 1.0)
    assert action is GuardAction.SKIP_MICROBATCH
    assert state.total_skipped_steps == 1 and state.consecutive_skips == 1


def test_spike_not_armed_during_warmup() -> None:
    state = SpikeGuardState()
    for step in range(1, 200):
        state, _ = observe(state, step, 1.0 + 0.01 * (step % 3))
    assert is_loss_spike(state, 50.0, in_warmup=False, cfg=CFG)
    assert not is_loss_spike(state, 50.0, in_warmup=True, cfg=CFG)
    assert not is_loss_spike(state, 1.01, in_warmup=False, cfg=CFG)
    assert not is_loss_spike(SpikeGuardState(), 50.0, in_warmup=False, cfg=CFG)


def test_dev_loss_blowup() -> None:
    state, diverged = dev_loss_diverged(SpikeGuardState(), 1.0, CFG)
    assert not diverged and state.best_dev_loss == 1.0
    state, diverged = dev_loss_diverged(state, 1.4, CFG)
    assert not diverged
    _, diverged = dev_loss_diverged(state, 1.6, CFG)
    assert diverged
    _, diverged = dev_loss_diverged(state, math.nan, CFG)
    assert diverged


@given(
    st.lists(
        st.tuples(
            st.one_of(st.floats(0, 10), st.just(math.nan)),
            st.one_of(st.floats(0, 10), st.just(math.inf), st.none()),
        ),
        max_size=40,
    )
)
def test_state_round_trips_through_json(events: list[tuple[float, float | None]]) -> None:
    state = SpikeGuardState()
    for step, (loss, grad) in enumerate(events, start=1):
        state, _ = observe(state, step, loss, grad)
        restored = LoopState(spike=state)
        assert LoopState.model_validate_json(restored.model_dump_json()) == restored
