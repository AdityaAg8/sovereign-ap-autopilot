# Review UI

The Next.js 15 app reviewers and approvers use: inbox, review queue, invoice detail with each
field's box drawn on the page image, fraud-flag explanations, an operations board and the
assistant panel. Source lives in `src/`; the design system is described in [DESIGN.md](DESIGN.md).

## Run it

```bash
pnpm install --ignore-workspace   # or `make bootstrap` from the repo root
pnpm dev                          # http://localhost:3000 on built-in fixtures, no backend needed
```

The app reads its data through one of two adapters, picked at build time:

- `NEXT_PUBLIC_API_MODE` unset or `fixture`: in-browser fixtures. `make preview` from the repo
  root builds this mode and serves it on http://localhost:3001; the login page lists the demo
  accounts.
- `NEXT_PUBLIC_API_MODE=http`: the live API from `make up`, at `NEXT_PUBLIC_API_BASE_URL`
  (default `http://localhost:8000`). `make demo` builds and starts the app this way on port 3000.

## Scripts

| Script                                   | What it does                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `pnpm dev` / `pnpm build` / `pnpm start` | Development server, production build, serve the build                                |
| `pnpm lint` / `pnpm typecheck`           | ESLint, `tsc --noEmit`                                                               |
| `pnpm format` / `pnpm format:check`      | Prettier write / check                                                               |
| `pnpm test:e2e`                          | Playwright specs in `e2e/` against a fixture-mode build (`make e2e` builds it first) |
