"""Tests for the torch-free Extractor B loop control (ADR-0011 §3-5): run resolution and resume
checks, the SIGTERM pause flag, the save/eval plan, NaN/skip handling with `spikes.jsonl`,
divergence stops, best tracking and a hypothesis save/resume round trip of the loop state."""

from __future__ import annotations

import json
import math
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml.extractor_b.interfaces import (
    CheckpointCorruptError,
    EvalResult,
    GuardAction,
    LadderRung,
    ResumeMismatchError,
    TrainConfig,
    TrainingError,
)
from ml.extractor_b.loop import (
    LOOP_STATE_JSON,
    LoopController,
    PauseFlag,
    ProbeResult,
    assert_offline_env,
    choose_rung,
    cuda_memory_fraction,
    make_run_id,
    mlflow_run_params,
    probe_samples,
    read_loop_state,
    resolve_run,
    training_arguments_kwargs,
    vram_over_cap,
)
from tests.unit.ml.conftest import RowFactory, make_fingerprint

NOW = datetime(2026, 9, 23, 10, 30, tzinfo=UTC)
N_TRAIN = 800  # 50 steps/epoch at accumulation 16, 100 total, warmup 3


def _new(cfg: TrainConfig, now: datetime = NOW, **fp: object) -> LoopController:
    fingerprint = make_fingerprint(**fp)
    run = resolve_run(
        cfg, mode="new", fingerprint=fingerprint, now=now, new_mlflow_run_id=lambda r: f"ml-{r}"
    )
    return LoopController(
        cfg=cfg, store=run.store, fingerprint=fingerprint, n_train=N_TRAIN, pause=PauseFlag()
    )


def _save(loop: LoopController, step: int) -> None:
    out = loop.store.begin(step)
    (out / "adapter_model.safetensors").write_bytes(f"w{step}".encode())
    loop.write_loop_state(out, step)
    loop.store.commit(step, loop.checkpoint_meta(step, now=NOW))
    loop.after_commit(step)


# ---- run resolution -------------------------------------------------------------------------


def test_offline_env_is_required() -> None:
    assert_offline_env(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    )
    with pytest.raises(TrainingError, match="TRANSFORMERS_OFFLINE"):
        assert_offline_env({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})


def test_run_id_format() -> None:
    assert make_run_id(make_fingerprint(), NOW).startswith("b-20260923T1030-")


def test_new_run_then_resume_auto(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    assert (loop.store.run_dir / "run.json").is_file()
    _save(loop, 5)
    _save(loop, 10)
    torn = loop.store.begin(15)
    resumed = resolve_run(
        cfg,
        mode="auto",
        fingerprint=make_fingerprint(git_sha="new"),
        now=NOW,
        new_mlflow_run_id=str,
    )
    assert resumed.record.mlflow_run_id.startswith("ml-b-")
    assert resumed.resume_from is not None and resumed.resume_from.step == 10
    assert resumed.removed_partials == (torn,)
    pinned = resolve_run(
        cfg, mode=5, fingerprint=make_fingerprint(), now=NOW, new_mlflow_run_id=str
    )
    assert pinned.resume_from is not None and pinned.resume_from.step == 5


def test_resume_picks_newest_run(cfg: TrainConfig) -> None:
    _new(cfg)
    later = _new(cfg, now=NOW + timedelta(hours=1))
    resumed = resolve_run(
        cfg, mode="auto", fingerprint=make_fingerprint(), now=NOW, new_mlflow_run_id=str
    )
    assert resumed.run_dir == later.store.run_dir and resumed.resume_from is None


def test_resume_without_run_or_with_mismatch_fails(cfg: TrainConfig) -> None:
    with pytest.raises(TrainingError, match="--new-run"):
        resolve_run(
            cfg, mode="auto", fingerprint=make_fingerprint(), now=NOW, new_mlflow_run_id=str
        )
    _new(cfg)
    with pytest.raises(ResumeMismatchError, match="manifest, rung"):
        resolve_run(
            cfg,
            mode="auto",
            fingerprint=make_fingerprint(manifest_sha256="0" * 64, rung=LadderRung.L1),
            now=NOW,
            new_mlflow_run_id=str,
        )
    with pytest.raises(CheckpointCorruptError):
        resolve_run(cfg, mode=25, fingerprint=make_fingerprint(), now=NOW, new_mlflow_run_id=str)
    with pytest.raises(TrainingError, match="already exists"):
        _new(cfg)


# ---- pause flag -----------------------------------------------------------------------------


def test_pause_flag_on_signal() -> None:
    flag = PauseFlag()
    before = signal.getsignal(signal.SIGINT)
    flag.install([signal.SIGINT])
    try:
        signal.raise_signal(signal.SIGINT)
        assert flag.requested
    finally:
        flag.uninstall()
    assert signal.getsignal(signal.SIGINT) == before


# ---- controller -----------------------------------------------------------------------------


def test_plan_step_end(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    assert loop.total_steps == 100
    assert loop.plan_step_end(1) == loop.plan_step_end(1).__class__(False, False, False, False)
    assert loop.plan_step_end(25).save and loop.plan_step_end(25).generative
    assert loop.plan_step_end(50).dev_loss and loop.plan_step_end(50).save
    assert not loop.plan_step_end(26).save
    loop.pause.request()
    plan = loop.plan_step_end(26)
    assert plan.save and plan.stop


def test_nan_microbatch_is_logged_and_the_step_still_runs(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    assert loop.on_microbatch(7, 1.2, ["ok-1"]) is GuardAction.OK
    assert loop.on_microbatch(7, math.nan, ["bad-1"]) is GuardAction.SKIP_MICROBATCH
    assert loop.on_pre_optimizer(7, 0.8) is GuardAction.OK  # the finite micro-batches still train
    assert not loop.take_skip_flag()
    line = json.loads(loop.spikes_path.read_text(encoding="utf-8").splitlines()[0])
    assert line == {
        "step": 7,
        "sample_ids": ["bad-1"],
        "loss": "nan",
        "grad_norm": None,
        "action": "skip_microbatch",
    }
    assert loop.state.spike.total_skipped_steps == 1


def test_all_nan_step_skips_the_optimizer_step(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    loop.on_microbatch(8, math.nan, ["bad-1"])
    loop.on_microbatch(8, math.inf, ["bad-2"])
    assert loop.on_pre_optimizer(8, 0.0) is GuardAction.SKIP_MICROBATCH
    assert loop.take_skip_flag() and not loop.take_skip_flag()
    last = json.loads(loop.spikes_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["action"] == "skip_step" and last["sample_ids"] == ["bad-1", "bad-2"]
    assert loop.state.spike.total_skipped_steps == 1


def test_inf_grad_skips_then_three_in_a_row_stop(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    for step in (10, 11):
        loop.on_microbatch(step, 1.0, [f"s{step}"])
        assert loop.on_pre_optimizer(step, math.inf) is GuardAction.SKIP_STEP
        assert loop.take_skip_flag()
    loop.on_microbatch(12, 1.0, ["s12"])
    assert loop.on_pre_optimizer(12, math.inf) is GuardAction.STOP
    assert loop.stop_reason is not None and loop.plan_step_end(12).stop


def test_clean_step_keeps_optimizer_running(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    loop.on_microbatch(5, 1.0, ["a"])
    assert loop.on_pre_optimizer(5, 0.5) is GuardAction.OK
    assert not loop.take_skip_flag()
    assert not loop.spikes_path.exists()


def test_loss_spike_after_warmup_is_logged_not_skipped(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    for step in range(1, 40):
        loop.on_microbatch(step, 1.0 + 0.01 * (step % 2), ["x"])
        loop.on_pre_optimizer(step, 0.5)
    loop.on_microbatch(40, 80.0, ["spiky"])
    assert loop.on_pre_optimizer(40, 0.5) is GuardAction.OK
    assert not loop.take_skip_flag()
    record = json.loads(loop.spikes_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["action"] == "spike_logged" and record["sample_ids"] == ["spiky"]


def test_dev_loss_blowup_requests_stop(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    loop.record_eval(EvalResult(step=50, kind="dev_loss", dev_loss=1.0))
    assert loop.stop_reason is None
    loop.record_eval(EvalResult(step=100, kind="dev_loss", dev_loss=2.0))
    assert loop.stop_reason is not None and "dev loss" in loop.stop_reason


def test_best_follows_dev_f1_and_survives_retention(cfg: TrainConfig) -> None:
    loop = _new(cfg)
    loop.record_eval(EvalResult(step=25, kind="generative", dev_f1=90.0))
    _save(loop, 25)
    assert loop.store.best_step() == 25
    for step in (50, 75, 100):
        loop.record_eval(EvalResult(step=step, kind="generative", dev_f1=80.0))
        _save(loop, step)
    assert loop.store.verified_steps() == [25, 75, 100]
    assert read_loop_state(loop.store.path_for(100)).best_step == 25
    meta = loop.store.verify(loop.store.path_for(100))
    assert meta is not None and meta.eval is not None and meta.eval.dev_f1 == 80.0
    assert meta.epoch == 2.0


def test_read_loop_state_missing_is_typed(tmp_path: Path) -> None:
    with pytest.raises(TrainingError):
        read_loop_state(tmp_path)


@settings(max_examples=20, deadline=None)
@given(
    events=st.lists(
        st.tuples(
            st.one_of(st.floats(0.1, 5), st.just(math.nan)),
            st.one_of(st.floats(0, 3), st.just(math.inf)),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_loop_state_save_resume_round_trip(
    tmp_path_factory: pytest.TempPathFactory, events: list[tuple[float, float]]
) -> None:
    cfg = TrainConfig(runs_dir=tmp_path_factory.mktemp("runs"))
    loop = _new(cfg)
    for step, (loss, grad) in enumerate(events, start=1):
        loop.on_microbatch(step, loss, [f"s{step}"])
        if loop.on_pre_optimizer(step, grad) is GuardAction.STOP:
            break
    step = len(events)
    _save(loop, step)
    resumed = resolve_run(
        cfg, mode="auto", fingerprint=make_fingerprint(), now=NOW, new_mlflow_run_id=str
    )
    assert resumed.resume_from is not None and resumed.resume_from.step == step
    again = LoopController(
        cfg=cfg,
        store=resumed.store,
        fingerprint=make_fingerprint(),
        n_train=N_TRAIN,
        pause=PauseFlag(),
    )
    again.resumed(read_loop_state(resumed.store.path_for(step)))
    assert again.state.spike == loop.state.spike
    assert again.state.resume_count == loop.state.resume_count + 1
    assert (resumed.store.path_for(step) / LOOP_STATE_JSON).is_file()


# ---- HF args, VRAM and probe helpers ----------------------------------------------------------


def test_training_arguments_are_the_fixed_design_values(cfg: TrainConfig, tmp_path: Path) -> None:
    kw = training_arguments_kwargs(cfg, n_train=N_TRAIN, output_dir=tmp_path)
    assert kw["gradient_accumulation_steps"] == 16 and kw["per_device_train_batch_size"] == 1
    assert kw["save_strategy"] == "no" and kw["eval_strategy"] == "no"
    assert kw["report_to"] == "none" and kw["ignore_data_skip"] is False
    assert kw["load_best_model_at_end"] is False and kw["save_total_limit"] is None
    assert kw["warmup_steps"] == 3 and kw["optim"] == "adamw_8bit" and kw["seed"] == 3407


def test_vram_helpers(cfg: TrainConfig) -> None:
    assert cuda_memory_fraction(11.5, 12_884_901_888) == pytest.approx(0.8925, abs=1e-3)
    assert cuda_memory_fraction(11.5, 8_000_000_000) == 1.0
    with pytest.raises(TrainingError):
        cuda_memory_fraction(11.5, 0)
    assert vram_over_cap(9_100_000_000, cfg) and not vram_over_cap(8_900_000_000, cfg)


def test_probe_helpers(cfg: TrainConfig, make_row: RowFactory) -> None:
    train = [make_row(f"t{n}", tokens=(100 + n, 10)) for n in range(40)]
    dev = [make_row("d0", role="dev"), make_row("d1", role="dev", tokens=(900, 9))]
    longest, dev_long = probe_samples(train, dev, n=30)
    assert len(longest) == 30 and longest[0].sample_id == "t39" and dev_long.sample_id == "d1"
    with pytest.raises(TrainingError):
        probe_samples([], dev)
    results = [ProbeResult(LadderRung.L1, 11.0, 5.0), ProbeResult(LadderRung.L0, 11.9, 4.0)]
    chosen = choose_rung(results, cfg)
    assert chosen is not None and chosen.rung is LadderRung.L1
    assert choose_rung([ProbeResult(LadderRung.L3, 12.0, 1.0)], cfg) is None


# ---- MLflow params on start and resume ------------------------------------------------------


def test_mlflow_params_new_run_logs_fingerprint_and_run_id() -> None:
    fp = make_fingerprint()
    params, tags = mlflow_run_params(fp, "b-run", {})
    assert params == {"fingerprint": fp.model_dump_json(), "run_id": "b-run"}
    assert tags == {"git_sha": "abc123"}


def test_mlflow_params_resume_after_code_change_logs_no_params_and_retags_git_sha() -> None:
    first = make_fingerprint(git_sha="48cc865")
    logged = {"fingerprint": first.model_dump_json(), "run_id": "b-run"}
    params, tags = mlflow_run_params(make_fingerprint(git_sha="a2d8ac3"), "b-run", logged)
    assert params == {}  # MLflow params are immutable: re-logging a changed value is rejected
    assert tags == {"git_sha": "a2d8ac3"}


def test_mlflow_params_resume_with_a_different_identity_is_refused() -> None:
    logged = {"fingerprint": make_fingerprint().model_dump_json()}
    with pytest.raises(TrainingError, match="different run fingerprint"):
        mlflow_run_params(make_fingerprint(config_sha256="0" * 64), "b-run", logged)
