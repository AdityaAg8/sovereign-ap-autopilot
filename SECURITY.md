# Security

Sovereign AP Autopilot runs locally; there is no hosted deployment.

## Scope

- The runtime makes no outbound network calls; only the host-side `make models` download
  reaches the internet.
- The full threat model (assets, trust boundaries, STRIDE table, attacker goals, control
  catalogue) is in [`docs/security.md`](docs/security.md).
- The adversarial invoice suite (`attacks/`) and the injection-boundary design
  (`apx/policy`, `apx/scan`) implement the mitigations tracked by that threat model.

## Supported versions

Only the latest release is supported.

## Reporting a vulnerability

This is a single-maintainer research project. Please report vulnerabilities through GitHub's
private advisory form (Security tab, "Report a vulnerability"), not a public issue. I aim to
reply within 7 days.

## Handling of secrets

- Real credentials live only in `.env` (gitignored); `.env.example` documents key
  names, never values.
- `gitleaks` runs in pre-commit and CI to catch accidental secret commits.
- Model weights, data and logs are gitignored; nothing in `data/`, `models/` or
  `logs/` is ever committed.
