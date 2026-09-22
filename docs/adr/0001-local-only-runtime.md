# ADR-0001: Local-only runtime

- Status: accepted
- Date: 2026-09-22
- Deciders: architect (M0.3)
- PRD: §2 C1, C5, C6

## Context and problem statement

Invoices contain bank details and commercial data; the product promise is that nothing leaves the
machine. The runtime needs model serving, a database, workflows and tracing, all of which normally
reach out (telemetry, model hubs, package indexes). How do we guarantee and prove zero outbound
calls at runtime?

## Considered options

1. **Compose app network with `internal: true`**, models pre-downloaded by `make models`, mounted
   read-only; only a separate `egress` network used by setup tooling.
2. Host firewall rules (Windows Defender Firewall / iptables).
3. Convention only: env flags such as `HF_HUB_OFFLINE=1`, `DO_NOT_TRACK=1`.

## Decision outcome

Option 1, with option 3 as defence in depth. All runtime containers (api, worker, llama-swap,
parse, postgres, temporal, phoenix, mlflow) join only the internal network. Model download and
image builds happen before `make up` on a network that runtime containers never join. Offline
env flags (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PHOENIX` / MLflow telemetry off) are set in
images. Browser access is by published ports on localhost only.

Option 2 is host-specific and not reproducible in `make fresh-clone`; option 3 alone is unprovable.

## Consequences

- Good: enforced by Docker, identical on every host, testable from inside the network.
- Bad: no runtime model or package fetch; a missing model is a hard `ConfigError`, never a
  silent download. Updating models requires re-running `make models`.
- Code rule: `apx/llm` is the only module that opens model connections, and only to the internal
  llama-swap hostname.

## Proof (gate)

- M0.4: `make up` healthy with the internal network.
- Integration test (`tests/integration/test_no_egress.py`): from the api and worker containers,
  DNS resolution and TCP connect to a public host (for example `1.1.1.1:443`, `huggingface.co`)
  must fail; connect to llama-swap and postgres must succeed. Part of `verify-m0` and M11 audit.
