# Documentation

New to the project? Read [Getting started](getting-started.md), then the architecture, then the
ADR behind whichever module you are changing.

## Guides

| Document | Read it for |
|---|---|
| [Getting started](getting-started.md) | Fresh machine to running demo, step by step |
| [Commands](commands.md) | Every `make` target grouped by task, plus useful one-liners |
| [Evaluation guide](evaluation.md) | Datasets, splits, what each report measures and how to regenerate it |
| [Training Extractor B](training.md) | Running, watching, pausing and exporting the fine-tune; first-run results |
| [Glossary](glossary.md) | UAE e-invoicing, pipeline, security and ML terms in plain words |
| [Architecture](architecture.md) | Modules, import direction, data flow, trust boundaries, GPU staging, the payee pin, the decision index |
| [Security threat model](security.md) | Assets, attackers, STRIDE per boundary, attack trees, the A01-A12 attack categories, the SEC-01..25 control catalogue, residual risks |
| [Runbook](runbook.md) | Starting and stopping the stack, service URLs, GPU budget and the training lock, common failures, migrations, the ops dashboard, audit verification |
| [Demo walkthrough](demo.md) | What `make demo` stages and what each scenario shows |
| [Models, datasets and licenses](licenses.md) | Every runtime and offline model and dataset, with its license |
| [Extractor B training loop](design/training-loop.md) | Builder-level design of the pausable M7.1 fine-tune (checkpoints, resume, data guards) |

## Architecture decision records

| ADR | Decision |
|---|---|
| [0001](adr/0001-local-only-runtime.md) | The runtime is local-only: app containers sit on an internal network and nothing calls out |
| [0002](adr/0002-serving-and-vram-staging.md) | llama-swap serves every model, with one group per pipeline stage on the 12 GB GPU |
| [0003](adr/0003-trust-boundary.md) | Extractors return typed values only; boxes come from parse words (I1); document text reaches tool-using models only as `UntrustedText` (I2) |
| [0004](adr/0004-trust-gate-method.md) | The trust gate calibrates P(correct) per field and picks its threshold with Learn-then-Test |
| [0005](adr/0005-eval-protocol.md) | Fixed sha256 splits, a leakage test and a generated scoreboard; numbers are never hand-written |
| [0006](adr/0006-host-deviation-native-windows.md) | The host is native Windows 11 with Docker Desktop, not WSL2 Ubuntu |
| [0008](adr/0008-typed-boundary-and-capabilities.md) | Typed boundary and capability model: every path to a payment ends at a deterministic check |
| [0009](adr/0009-extractor-b-training-and-trust-features.md) | Extractor B recipe (bf16 LoRA on Qwen3.5-4B) and the trust-gate features |
| [0010](adr/0010-agent-tool-contracts.md) | Agent tools read or propose, never act; spotlighted prompts; MCP exposes the same tools |
| [0011](adr/0011-training-loop.md) | Extractor B training loop: checkpoints, deterministic resume, data guards, `gpu.lock` by flock |
| [0012](adr/0012-extraction-contract-and-b-training-inputs.md) | Extraction contract for BT-3 and BT-151, B's training inputs, no money-path defaults in the UBL builder |
| [0013](adr/0013-live-serving-budget-and-backpressure.md) | Whole-card VRAM budget of 11.8 GB per stage, A and B swap, preflight headroom checks |
| [0014](adr/0014-p1-multiple-verified-accounts.md) | With several verified accounts, the invoice IBAN selects one and never introduces a payee; the payee is pinned from routing to build |
| [0015](adr/0015-promotion-gates.md) | An Extractor B challenger is promoted only if no gate (G1-G11) regresses |
| [0019](adr/0019-real-data-evaluation-track.md) | Real-data track: A and B measured on ExtractBench, CORD v2 and DocILE, then B retrained with CORD and DocILE train and measured before and after on the same hold-outs |

There is no ADR-0007; the number was not used.

## Reports

Measured results live in [`reports/`](../reports/). Each file names the command that regenerates it;
the [evaluation guide](evaluation.md) lists them all in one table.
