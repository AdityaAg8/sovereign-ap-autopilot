# Security

Sovereign AP Autopilot is a local-only, private-repo project (PRD §2). There is no
hosted deployment and no public attack surface today.

## Scope

- The runtime makes no outbound network calls once bootstrapped (PRD §2, §8).
- The full threat model (assets, trust boundaries, STRIDE table, attacker goals, control
  catalogue) is in [`docs/security.md`](docs/security.md).
- The adversarial invoice suite (`attacks/`) and the injection-boundary design
  (`apx/policy`, `apx/scan`) implement the mitigations tracked by that threat model.

## Reporting a vulnerability

This is a private, single-maintainer research project. If you have access to this
repository and find a security issue, open a private security advisory on the GitHub
repository (Security tab → "Report a vulnerability") rather than a public issue, or
contact the maintainer (@AdityaAg8) directly.

## Handling of secrets

- Real credentials live only in `.env` (gitignored); `.env.example` documents key
  names, never values.
- `gitleaks` runs in pre-commit and CI to catch accidental secret commits.
- Model weights, data and logs are gitignored; nothing in `data/`, `models/` or
  `logs/` is ever committed.
