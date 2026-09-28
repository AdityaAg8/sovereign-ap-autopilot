"""GPU half of the Extractor B run flow (design §4 steps 4-8 and the probe): check VRAM/RAM
headroom (ADR-0013), resolve the run, cap VRAM, load the base with LoRA via Unsloth, train with
`GuardedSFTTrainer` and log to MLflow. A model whose Gated DeltaNet layers are off the fast kernels
is refused at load (ADR-0011 amendment 6).
Runs only inside the trainer image; imported lazily by `train.py` so CI never needs torch."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from apx.llm.preflight import (
    GpuWorkload,
    HeadroomCheck,
    check_gpu_headroom,
    check_workload_headroom,
)
from ml.extractor_b.checkpoints import atomic_write_bytes
from ml.extractor_b.config import effective_lora_r, rung_spec
from ml.extractor_b.evaluator import HfDevEvaluator
from ml.extractor_b.gpu import query_used_mib
from ml.extractor_b.interfaces import LadderRung, TrainConfig, TrainingError, TrainingSample
from ml.extractor_b.kernels import require_deltanet_fast_path
from ml.extractor_b.loop import (
    LoopController,
    PauseFlag,
    ProbeResult,
    choose_rung,
    cuda_memory_fraction,
    mlflow_run_params,
    probe_samples,
    resolve_run,
    training_arguments_kwargs,
)
from ml.extractor_b.prompting import (
    INSTRUCTION_PART,
    LINE_ITEMS_SUFFIX,
    MASK_LINE_ITEMS_KEY,
    RESPONSE_PART,
    SAMPLE_IDS_KEY,
    LazyTrainingExamples,
    mask_line_items_labels,
)
from ml.extractor_b.selection import select_best
from ml.extractor_b.trainer import guarded_trainer_class, loop_callback_class

if TYPE_CHECKING:
    from ml.extractor_b.train import Args


def _load_rows(path: Path) -> list[TrainingSample]:
    return [
        TrainingSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cap_vram(cfg: TrainConfig) -> None:
    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction(cfg.vram.cap_gb, total))


def _load_model(cfg: TrainConfig, env: dict[str, str]) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    from ml.extractor_b.train import resolve_base

    base = str(resolve_base(env).path)
    spec = rung_spec(cfg)
    model, processor = FastVisionModel.from_pretrained(
        base,
        load_in_4bit=cfg.lora.load_in_4bit,
        max_seq_length=spec.max_seq_len,
        local_files_only=True,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
        r=effective_lora_r(cfg),
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        random_state=cfg.optim.seed,
        use_gradient_checkpointing="unsloth",
    )
    # ADR-0011 amendment 6: never train, probe or eval on the slow DeltaNet reference path.
    require_deltanet_fast_path(model)
    return model, processor


def _collator(model: Any, processor: Any) -> Any:
    import torch
    from unsloth.trainer import UnslothVisionDataCollator

    inner = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=True,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )
    suffix_ids = processor.tokenizer(LINE_ITEMS_SUFFIX, add_special_tokens=False)["input_ids"]

    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = inner(examples)
        for i, example in enumerate(examples):  # ADR-0012 §3: no loss on unlabelled line items
            if example.get(MASK_LINE_ITEMS_KEY):
                masked = mask_line_items_labels(
                    batch["input_ids"][i].tolist(), batch["labels"][i].tolist(), suffix_ids
                )
                batch["labels"][i] = torch.tensor(masked, dtype=batch["labels"].dtype)
        batch[SAMPLE_IDS_KEY] = [e[SAMPLE_IDS_KEY] for e in examples]
        return batch

    return collate


class _Mlflow:
    def __init__(self, cfg: TrainConfig) -> None:
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        self.mlflow = mlflow

    def new_run(self, run_id: str) -> str:
        with self.mlflow.start_run(run_name=run_id) as active:
            return str(active.info.run_id)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.mlflow.log_metrics(metrics, step=step)


def run(
    args: Args,
    cfg: TrainConfig,
    env: dict[str, str],
    *,
    headroom_check: HeadroomCheck = check_gpu_headroom,
) -> int:
    from apx.extract.prompt_b import SERVING_RUNG
    from ml.extractor_b.train import EXIT_NO_RUNG, EXIT_OK, EXIT_RUNG_CHANGE, fingerprint_for

    # ADR-0013 preflight: refuse (typed) before any model loads if VRAM or RAM is short.
    check_workload_headroom(GpuWorkload.TRAINER, check=headroom_check)

    if args.probe:
        chosen = _probe(cfg, env)
        if chosen is None:
            return EXIT_NO_RUNG
        if chosen.value != SERVING_RUNG:  # ADR-0012 §5: a reviewed change, never silent
            print(f"probe: first fitting rung is {chosen.value}, SERVING_RUNG is {SERVING_RUNG}")
            return EXIT_RUNG_CHANGE
        return EXIT_OK
    if args.mode is None:  # pragma: no cover - argparse requires a mode
        raise TrainingError("no run mode")

    tracker = _Mlflow(cfg)
    fingerprint = fingerprint_for(cfg, env)
    resolved = resolve_run(
        cfg,
        mode=args.mode,
        fingerprint=fingerprint,
        now=datetime.now(UTC),
        new_mlflow_run_id=tracker.new_run,
    )
    train_rows = _load_rows(cfg.data_dir / "train.jsonl")

    _cap_vram(cfg)
    model, processor = _load_model(cfg, env)
    pause = PauseFlag()
    pause.install()
    try:
        loop = LoopController(
            cfg=cfg,
            store=resolved.store,
            fingerprint=fingerprint,
            n_train=len(train_rows),
            pause=pause,
        )
        evaluator = HfDevEvaluator(
            _load_rows(cfg.data_dir / "dev.jsonl"), cfg.data_dir, run_dir=resolved.run_dir
        )
        evaluator.set_processor(processor)
        from trl import SFTConfig
        from unsloth import FastVisionModel

        trainer_cls = guarded_trainer_class()
        resume_dir = (
            None
            if resolved.resume_from is None
            else resolved.store.path_for(resolved.resume_from.step)
        )
        with tracker.mlflow.start_run(run_id=resolved.record.mlflow_run_id) as active:
            params, tags = mlflow_run_params(
                fingerprint, resolved.record.run_id, active.data.params
            )
            if params:
                tracker.mlflow.log_params(params)
            tracker.mlflow.set_tags(tags)
            callback = loop_callback_class()(
                loop,
                evaluator,
                processor=processor,
                resume_dir=resume_dir,
                log_metrics=tracker.log,
                for_inference=FastVisionModel.for_inference,
                for_training=FastVisionModel.for_training,
            )
            trainer = trainer_cls(
                model=model,
                tokenizer=processor,
                data_collator=_collator(model, processor),
                train_dataset=LazyTrainingExamples(train_rows, cfg.data_dir),
                args=SFTConfig(
                    **training_arguments_kwargs(
                        cfg, n_train=len(train_rows), output_dir=resolved.run_dir
                    )
                ),
                callbacks=[callback],
            )
            trainer.attach_loop(loop)
            trainer.train(resume_from_checkpoint=None if resume_dir is None else str(resume_dir))
            if pause.requested:
                return EXIT_OK
            best = select_best(loop.state.evals, tie_epsilon=cfg.cadence.f1_tie_epsilon)
            if best is not None:
                resolved.store.set_best(best)
                tracker.mlflow.log_metric("best_step", best)
    finally:
        pause.uninstall()
    return EXIT_OK


def _probe(cfg: TrainConfig, env: dict[str, str]) -> LadderRung | None:
    """Walk L0 → L3 on the longest samples, each rung at its own image size, polling nvidia-smi
    every 200 ms; write probe.json and return the first rung that fits."""
    import gc

    import torch
    from trl import SFTConfig
    from unsloth import FastVisionModel

    from ml.extractor_b.data import rows_at_rung

    base_train, base_dev = probe_samples(
        _load_rows(cfg.data_dir / "train.jsonl"), _load_rows(cfg.data_dir / "dev.jsonl")
    )
    results: list[ProbeResult] = []
    for rung in LadderRung:
        rcfg = cfg.model_copy(update={"rung": rung})
        if rung is cfg.rung:
            train, dev = base_train, base_dev
        else:
            train = rows_at_rung(base_train, rung.value, cfg.data_dir)
            dev = rows_at_rung([base_dev], rung.value, cfg.data_dir)[0]
        peak = [0.0]
        stop = threading.Event()

        def poll(peak: list[float] = peak, stop: threading.Event = stop) -> None:
            while not stop.is_set():
                peak[0] = max(peak[0], max(query_used_mib()) / 1024)
                time.sleep(0.2)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        model, processor = _load_model(rcfg, env)
        kwargs = training_arguments_kwargs(
            rcfg, n_train=len(train), output_dir=cfg.data_dir / "probe"
        )
        kwargs.update(gradient_accumulation_steps=1, max_steps=len(train), num_train_epochs=1)
        started = time.monotonic()
        from trl import SFTTrainer

        SFTTrainer(
            model=model,
            tokenizer=processor,
            data_collator=_collator(model, processor),
            train_dataset=LazyTrainingExamples(train, cfg.data_dir),
            args=SFTConfig(**kwargs),
        ).train()
        elapsed = (time.monotonic() - started) / max(len(train), 1)
        FastVisionModel.for_inference(model)
        HfDevEvaluator([dev], cfg.data_dir).generative(model, processor, 0)
        stop.set()
        poller.join()
        results.append(ProbeResult(rung, peak[0], elapsed))
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        if results[-1].peak_gb <= cfg.vram.probe_peak_gb:
            break
    chosen = choose_rung(results, cfg)
    payload = {
        "results": [
            {"rung": r.rung.value, "peak_gb": r.peak_gb, "seconds_per_step": r.seconds_per_step}
            for r in results
        ],
        "chosen": None if chosen is None else chosen.rung.value,
    }
    atomic_write_bytes(cfg.data_dir / "probe.json", json.dumps(payload, indent=2).encode("utf-8"))
    return None if chosen is None else chosen.rung
