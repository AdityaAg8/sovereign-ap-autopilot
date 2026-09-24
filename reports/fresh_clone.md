# Fresh-clone check

Commit `98a05336166fdceb63234f25412ec45ce3df005e`, cloned into an empty directory with no untracked files.

| Step | Result | Seconds |
|---|---|---|
| uv sync --frozen | pass | 6 |
| pnpm install --frozen-lockfile | pass | 12 |
| ruff check + format | pass | 1 |
| mypy --strict | pass | 25 |
| web lint + typecheck | pass | 9 |
| unit tests | pass | 66 |
| compose config | pass | 1 |
| apx image build | pass | 3 |
