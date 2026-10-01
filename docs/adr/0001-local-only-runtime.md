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

## Amendment (2026-09-22): `gateway` service and the `edge` network

`internal: true` turned out to mean more than "no egress": Docker also never sets up host
port publishing for a container whose only network is internal (`docker port` shows nothing,
`curl 127.0.0.1:PORT` gets connection refused), regardless of a service's own `ports:` list.
That's the opposite of this ADR's original claim ("Published ports still work: Docker
publishes ports via the host's docker-proxy regardless of `internal`") -- verified wrong
empirically on this host, not just in theory. Host tools and a browser had no way to reach
llama-swap, Phoenix, MLflow or Temporal's UI.

**Decision:** add one more container, `gateway` (static nginx, `infra/docker/gateway/`),
that joins both `app` and a new network, `edge` (`internal: false`). `gateway` is the *only*
container on `edge`, and it publishes ports to `127.0.0.1` only (never `0.0.0.0`). It proxies:
llama-swap (8080), Phoenix (6006), MLflow (5000), Temporal's Web UI (8233), and Postgres
(5432, a raw TCP passthrough via nginx's `stream` module -- Postgres isn't HTTP) for local
dev tools. Every other service dropped its now-dead `ports:` entries. `gateway` runs no
application code: its nginx config is a fixed, read-only set of `proxy_pass` targets with no
client-controlled routing, so there is no way to make it reach anything other than those five
named upstreams.

This does change the trust boundary slightly: because `edge` is non-internal, `gateway`
itself *does* have outbound routing to the internet (confirmed: `wget http://1.1.1.1` from
inside the `gateway` container succeeds) -- a non-internal network can't be inbound-only.
That is accepted, not a gap, for two reasons: `gateway` runs no application code that could
be induced to fetch or exfiltrate anything, and its fixed nginx config cannot be pointed at
an arbitrary host by a client request. The C1 no-egress guarantee this ADR proves is
therefore scoped to the containers that actually handle invoice data or call models --
**api, worker, llama-swap, toolbox** (`tests/integration/test_no_egress.py`) -- not `gateway`,
which is checked differently: proven to reach only its five declared proxy targets.

Also added: `toolbox` (profile `dev`, `infra/docker/toolbox/`), a Linux Python 3.12 + CUDA
torch/transformers container on `app` only (no `edge`), used by `make smoke` and later
evals/training. It exists because the Windows host blocks native DLLs from these wheels
under Smart App Control (ADR-0006's addendum); it carries the same no-egress guarantee as
llama-swap, proven by the same integration test.

## Amendment 2 (2026-09-25): parse on the gateway for host-run GPU evals

The GPU evals (`make bakeoff-probe`, `bakeoff`, `extraction-a-adr0012`, `cal-rows-a/b`,
`eval-b`, `gates-b`) run on the host through `uv run`, since the toolbox image carries only the
smoke/training stack, not the project's dependencies. They parse through the same
`ParseServiceClient` the worker uses, whose default `http://parse:8000` resolves only on `app`,
so no host run could reach it.

**Decision:** `gateway` proxies parse on `127.0.0.1:8010` (loopback only, like every other
gateway port), and those Makefile targets set `PARSE_BASE_URL=http://127.0.0.1:8010`. The
runtime path is unchanged: api and worker still call `parse:8000` over `app`. Parse holds no
secrets and makes no outbound calls; it is one more fixed `proxy_pass` target, and
`tests/integration/test_no_egress.py` checks that 8010 reaches parse and nothing else.

## Amendment 3 (2026-09-27): Postgres and the Temporal UI move to an opt-in `devports` profile

M11 audit pass 1 (REVIEW.md C-02, C-03; commit dab25f3) narrowed the first amendment. The
default `gateway` no longer publishes Postgres (5432) or Temporal's Web UI (8233): a direct
Postgres login is the schema owner, and from the Temporal UI anyone on the host could send a
`review` signal. Both are served only by a second gateway under the Compose `devports` profile
(`COMPOSE_PROFILES=devports make up`, `infra/docker/gateway/nginx.devports.conf`), still bound to
127.0.0.1, and its Temporal UI proxy passes GET and HEAD only. Database passwords are generated
into `.env` by `make env`; no service, `apx/db/session.py` or `alembic.ini` carries a default.
Proof: `tests/unit/test_compose_security.py`, `tests/unit/test_db_session.py`. The no-egress
scope and the proof above are unchanged.
