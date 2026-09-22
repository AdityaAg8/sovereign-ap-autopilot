# Contributing

This is currently a single-maintainer, agent-driven project run out of `PROGRESS.md`
against the spec in `PRD.md`. The notes below are for anyone (human or agent) working
in this repository.

## Workflow

1. Read `CLAUDE.md`, `PROGRESS.md` and `BLOCKED.md` before starting; open PRD sections
   by number rather than reading the whole spec.
2. One branch per milestone: `m<N>-<slug>`, cut from `main`.
3. One task ≈ one commit, using [Conventional Commits](https://www.conventionalcommits.org/).
4. Every new source file starts with the PRD §8 header comment (what it does, where it
   fits).
5. Tests ship with the code that needs them. Never skip or `xfail` a test, never mock
   the unit under test, never hardcode an expected eval answer, and never relax a gate
   without an ADR in `docs/adr/`.
6. Before pushing: `make lint fmt typecheck test` locally, and
   `uv run pre-commit run --all-files`.
7. Open a PR against `main` using the template; merge only after CI is green with
   `gh pr merge --squash --delete-branch`. Never force-push `main`.

## Local setup

```bash
make bootstrap   # uv sync, pnpm install (web/), pre-commit install
make lint
make test
```

See the [README](README.md) quickstart for the full command list.

## Code style

- Python: ruff (lint + format), mypy `--strict` on `apx/`, `factory/`, `evals/`.
- TypeScript (`web/`): strict mode, ESLint, Prettier.
- Commit messages: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, …).
