# Sovereign AP Autopilot

Local-only AP agent: supplier invoices (EN/AR) become validated UAE PINT AE
e-invoices, with fraud checks, a provable field error rate and
injection-proof payments. See [`PRD.md`](PRD.md) for the full spec and
[`PROGRESS.md`](PROGRESS.md) for current state.

Status: **M0 Foundations** (scaffolding in progress).

## Quickstart

```bash
make bootstrap   # install toolchains and pre-commit hooks
make up          # start local infra (Postgres, Temporal, Phoenix, MLflow, llama-swap)
make models      # download and verify local model weights
make smoke       # verify every model loads and serves correctly
make test        # run the unit test suite
```

See `make help`-equivalent target list in the [Makefile](Makefile).

<!-- SCOREBOARD:START -->
No scoreboard yet — populated by `evals/run.py` from M1 onward.
<!-- SCOREBOARD:END -->

## Repository layout

See [PRD §7](PRD.md#7-repository-layout).

## Models and licenses

See `docs/licenses.md` (from M10) and [PRD §4](PRD.md#4-models).

## Security

See [SECURITY.md](SECURITY.md) and `docs/security.md` (from M6).
