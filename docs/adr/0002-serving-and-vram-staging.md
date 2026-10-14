# ADR-0002: Model serving and VRAM staging with llama-swap groups

- Status: accepted
- Date: 2026-09-22

## Context and problem statement

Seven runtime models total far more than the 12 GB RTX 4080 Laptop GPU. Every stage must peak at
≤ 11.5 GB with its working context and no CPU offload. We need one serving layer that loads the
right models per pipeline stage, and a scheduler so two invoices never fight over VRAM.

## Considered options

1. **llama-swap in front of llama-server, one group per `Stage`**, plus an app-level
   `StageScheduler` holding a single GPU lock; transformers models (guard, embeddings, parse)
   served by small internal services that load/unload on the same lock.
2. vLLM with multiple engines: poor GGUF + mmproj support, higher VRAM floor.
3. Ollama: less control over grammar/JSON-schema, logprobs and exact GGUF revisions.
4. Keep everything resident: exceeds 12 GB.

## Decision outcome

Option 1. `apx/llm` defines `Stage` groups (see `docs/architecture.md` table):
`scan`, `parse`, `extract` (A + B together), `arbitrate`, `fraud`, `fraud_page` (alone),
`agent`. `StageScheduler.stage(s)` is an async context: acquire lock → ensure group loaded →
yield → release. Invoices are batched per stage by the worker to amortise swaps. Extraction uses
`StructuredRequest` (JSON-schema, temperature ≤ 0.2, fixed seed, logprobs, thinking disabled).

Fallback rule: if a stage's measured peak exceeds 11.5 GB, that group becomes sequential
(one model at a time within the stage) and the decision is recorded in `reports/smoke.md`.

## Consequences

- Good: one OpenAI-compatible API, pinned GGUFs from `infra/models.lock.json`, VRAM bounded by
  construction, stage swaps visible in traces.
- Bad: swap latency (seconds per stage change) inflates p95; mitigated by per-stage batching.
- Single GPU job at a time also covers training: `make gpu-free` stops llama-swap first.

## Proof (gate)

- `make smoke`: per model load, peak VRAM sampled every 200 ms, latency, JSON-schema output,
  logprobs, thinking disabled; every stage ≤ 11.5 GB or switched to sequential.
- Scoreboard row "peak VRAM per stage", re-measured before each release.
- Unit test: `StageScheduler` never yields two stages concurrently (fake llama-swap client).
