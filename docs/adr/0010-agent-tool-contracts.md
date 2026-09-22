# ADR-0010: Agent tool contracts, prompt structure and MCP exposure

- Status: accepted
- Date: 2026-09-22
- Deciders: architect (M9.1)
- PRD: §5 (scoreboard rows for tool calls, citations, AgentDojo); §6.2 I2, I3, I4; §6.3 P5;
  §10 M9
- Extends ADR-0003 and ADR-0008 (it replaces neither). Related: ADR-0001, `docs/security.md`
  boundaries B6 and B7, controls SEC-12, SEC-13, SEC-18, SEC-21, SEC-22.
- Contracts: `apx/agent/interfaces.py`.

## Context and problem statement

M9 connects a tool-calling model (Qwen3.5-9B, §4) to invoice, vendor and audit state. The
exception agent writes review notes, ops chat answers questions in the UI, and an MCP server
offers the same tools to local MCP clients. Every one of these reads text that an attacker wrote.

ADR-0008 already decides that the agent cannot change payment state: `apx.agent` never receives
a `Grant`, `PolicyGuard` or payment-state writer, and a `ProposalSink` is its only write
dependency (K3). Before M9.2 can be built and tested, this ADR has to fix:

1. which tools exist, with typed parameters and results, caps and role scoping;
2. how tool SQL stays parameterised, and how we prove it;
3. which tools are free reads and which only create proposals;
4. the prompt layout, including where untrusted text may appear;
5. the citation format, which links answers to audit events;
6. what the agent does when it refuses a request;
7. what MCP exposes, and how an MCP client authenticates;
8. the hooks that M9.5 needs for its evaluations.

## Decision drivers

- P5 and I3 must hold by construction. The model's judgement must never be what keeps them true.
- Data minimisation (attacker goal G3): each call is scoped to one entity, IBANs are masked and
  rows are capped.
- Answers must be auditable: a claim about a decision cites the audit event behind it
  (`docs/security.md` B6, R row).
- One contract table should drive the tool specs, dispatch, MCP listing and tests.
- C1 and C6: no path sends anything and no path reaches the network.
- Use the simplest design that passes M9.2 to M9.5.

## Considered options

### Tool surface

1. **Six typed tools described by one contract table** (`TOOL_CONTRACTS`). Each tool has a
   strict parameter model, a result model, a row cap, roles, allowed origins and, for proposal
   tools, a proposal kind.
2. A generic read-only SQL tool (text-to-SQL over views). This is flexible, but the model would
   write the SQL, so we could not guarantee parameters. Masking and caps become best effort, and
   bulk export becomes one query.
3. Keep the M0 `AgentTool` shape (untyped `dict` arguments per tool). Checks would be spread
   across tools, and there would be no typed result to test.

### Citations

A. **Inline markers such as `[audit:1234]`**, checked against the refs that this turn's tool
   results actually returned.
B. A separate citation list produced in JSON mode. That needs a second constrained call after
   the tool loop, and it is harder to render next to the claim it supports.
C. No checking. Faithfulness could not be measured, and invented audit ids would reach the UI.

### MCP transport

i. **stdio only.** The server is started inside the `api` container, and the client supplies
   a JWT at start-up.
ii. Streamable HTTP on 127.0.0.1 with a JWT and Origin checks. This adds a network listener,
   which brings CSRF and DNS-rebinding risk (B7 S). It is deferred; adding it needs a new ADR.

### SQL safety

- **A registry of constant statements, bound parameters, and a read-only role and transaction.**
- An ORM query builder is also safe, but it is harder to prove the full set of statements.
- Stored procedures need more infrastructure and add nothing for this use.

## Decision outcome

We choose option 1, option A, option i, and the registry of constant statements.

### D1 Tool contracts

| Tool | Effect | Key parameters | Result (`content`) | `untrusted` | Cap | Roles (chat, MCP) | Origins | Proposal |
|---|---|---|---|---|---|---|---|---|
| `search_invoices` | read | `vendor_id`, `vendor_query` (vendor-master names only), `invoice_number` (T2 pattern), `status`, `route`, received date range, total range, `currency`, `sort` (3 fixed orders), `limit` ≤ 25 | `InvoiceSearchResult` with `InvoiceRow`s | none | 25 rows | all | all; for `agent`, the vendor is forced to the scope invoice's vendor | – |
| `get_invoice` | read | `invoice_id` | `InvoiceView`: typed fields, field acceptance, routing reasons, flags (no evidence text), masked payee, `document_iban_matches_payee`, pending proposals | supplier and buyer names and addresses, up to 50 line descriptions, any off-format value | 50 lines | all | all; for `agent`, only the scope invoice | – |
| `explain_decision` | read | `invoice_id`, `topic` | `DecisionExplanation`: `CitedReason`s (each with at least one audit id), `AuditEventView`s (payload filtered to `AUDIT_FACT_KEYS`) | none | 50 events | all | all; for `agent`, only the scope invoice | – |
| `vendor_history` | read | `vendor_id`, `since`, `limit` ≤ 50 | `VendorHistory`: master name, TRN, masked accounts (≤ 10), bank changes (≤ 20), invoice rows, counts | none | 50 invoices | all | all; for `agent`, only the scope invoice's vendor | – |
| `request_approval` | propose | `invoice_id`, `rationale` ≤ 500 characters, `cited_audit_ids` (1 to 10) | `ProposalReceipt` (`executed: Literal[False]`) | none | 1 | reviewer, approver | chat, mcp | `approve_payment` |
| `draft_vendor_email` | propose | `invoice_id`, `reason` (enum), `details` ≤ 800 characters, `language` | `EmailDraftReceipt` (`sent: Literal[False]`) | none | 1 | reviewer, approver | agent, chat, mcp | `request_info` |

`search_invoices` is the one capped exception to ADR-0008 K2's "read tools are scoped to one
invoice or vendor": for chat and MCP it returns at most 25 metadata rows across vendors (the same
visibility an AP user has in the inbox, no document text), and for `agent` origin it is forced
back to the scope invoice's vendor, so the exception never widens what the exception agent sees.

Rules that apply to every tool:

- **Parameters.**
  - Every parameter model is `Frozen` (`extra="forbid"`). Ids and invoice numbers have a
    pattern, strings have a maximum length, and choices are `Literal` types.
  - The dispatcher parses arguments with `model_validate(arguments)`. A validation failure
    returns the refusal `invalid_arguments`.
- **Results.**
  - Every result model inherits `ToolOutput`. Document-derived text travels only in
    `untrusted` (ADR-0008 T5), with sources `invoice:<id>:field:<name>` and
    `invoice:<id>:line:<i>:description`.
  - IBANs appear only as `MaskedIban` (`****1234`). The type rejects a raw IBAN.
  - Vendor names come from the vendor master.
  - Audit payloads are reduced to `AUDIT_FACT_KEYS` with scalar values.
  - Scan and fraud evidence text, user lists and secrets are never returned.
- **Caps.**
  - A result that hits its row cap has `truncated=True`. The reader fetches cap + 1 rows so it
    can tell.
  - If `content` is larger than `TOOL_RESULT_MAX_BYTES` (16 KiB) of canonical JSON, the
    dispatcher returns the refusal `result_too_large`. JSON is never cut silently.
  - Untrusted text keeps the ADR-0008 caps: 2000 characters per block and 8000 per prompt.
  - Per turn: at most `MAX_TOOL_CALLS_PER_TURN = 8` tool calls and
    `MAX_PROPOSALS_PER_TURN = 2` proposals.
  - MCP: at most `MCP_MAX_CALLS_PER_MINUTE = 60` calls per session.

### D2 Parameterised SQL only

- **One database dependency.** The only database access `apx.agent` has is
  `SqlReader.fetch_all(statement, params, limit)`.
- **Constant statements.**
  - Every `SqlStatement` is a module-level `SQL_*` constant in `apx/agent/sql.py`, and all of
    them are collected in `AGENT_SQL`. The reader raises `ConfigError` for any statement that
    is not in the set.
  - Each statement starts with `SELECT` or `WITH` and binds `:limit`.
- **No dynamic SQL.**
  - Optional filters use the form `(:p IS NULL OR col = :p)`.
  - Each sort order is its own constant statement, so there is no dynamic `ORDER BY`.
  - `vendor_query` is bound as a parameter with LIKE wildcards escaped (`ESCAPE '\'`).
- **Read-only database role.**
  - The reader runs in a `READ ONLY` transaction as the role `apx_agent_ro`.
  - That role can SELECT only from views `v_agent_invoices`, `v_agent_vendor_accounts` and
    `v_agent_audit`.
  - The views mask IBANs in SQL (`'****' || right(iban, 4)`) and compute
    `document_iban_matches_payee`, so the agent role has no grant on any raw IBAN column.
  - Proposals are written by the `ProposalSink` (apx.policy), which uses the application role.

### D3 Role and origin scoping

- **The runtime builds the `ToolCaller`; model output never does.**

  | Caller | `origin` | `user` | Scope |
  |---|---|---|---|
  | Exception agent | `agent` | none | `scope_invoice_id` = the invoice under review |
  | Ops chat | `chat` | the JWT user | – |
  | MCP | `mcp` | the user from the session token | – |

- **Read tools** are free for any authenticated role (ADR-0008 K2). The `agent` origin can read
  only its scope invoice and that invoice's vendor; anything else returns `out_of_scope`.
- **Proposal tools** need the reviewer or approver role (the roles holding
  `approve_payment` and `route_exception`, K1). An admin alone gets `role_denied`.
- **`request_approval` is not offered to the `agent` origin.** A document can then never make
  the autonomous agent draft an approval (residual risk R-3 in `docs/security.md`). The agent
  can still draft vendor emails.
- **This is least privilege, not the security boundary.** Executing any proposal still needs
  ADR-0008 K4 (`EXECUTE_PROPOSAL` plus `PROPOSAL_CAPABILITY[kind]`) and K5 (`authorize`).

### D4 Write tools only create proposals

- **`request_approval`**
  - Creates `Proposal(kind="approve_payment", payload={rationale, cited_audit_ids,
    requested_by}, created_by=<origin>)`.
  - Precondition: the current route is REVIEW. For BLOCK or AUTO it returns
    `precondition_failed`.
  - Each cited id must be an audit event for this invoice; otherwise it returns
    `invalid_arguments`.
  - Idempotent: if a pending `approve_payment` proposal already exists for the invoice, it
    returns that proposal.
  - The UI labels `rationale` as model-written.
- **`draft_vendor_email`**
  - Creates `Proposal(kind="request_info", payload={reason, language, subject, body,
    to_vendor_id})`.
  - The subject and body come from fixed templates per `EmailReason`, in English and Arabic,
    plus `details`.
  - `details` must not contain URLs, email addresses, or runs of 12 or more digits or
    IBAN-like tokens; otherwise it returns `invalid_arguments`.
  - No `EmailReason` concerns bank details. Those are confirmed only by an admin callback (P1).
  - The recipient is the vendor-master contact. It never comes from the document or the model.
  - No SMTP, HTTP or socket client is imported anywhere in `apx.agent`. A human copies the
    draft into their own mail client.
- **Both tools**
  - They write only through `ProposalSink.submit`, which audits every proposal (I4).
  - Their results carry `executed: Literal[False]` and `sent: Literal[False]`.
  - Neither `ToolProposalKind` covers `change_bank`.
  - Audit actor values are `agent`, `chat:<user_id>` and `mcp:<user_id>`.
- **No tool can execute, approve, pay, send or change bank details.** ADR-0008's AST test keeps
  `Grant`, `PolicyGuard` and the payment-state writer out of `apx/agent`.

### D5 Prompt structure and spotlighting

| # | Role | Content | Built from |
|---|---|---|---|
| 1 | system | `AGENT_SYSTEM_RULES`, a blank line, `SPOTLIGHT_RULE` (verbatim, ADR-0008 T3). For the agent only, it adds `## Case (trusted system state)` with canonical JSON of `AgentContext.state` and the routing reasons | constants and typed values only |
| 2 | user | Agent: the fixed task template `Write a review note for invoice <id>.`, then `## Document quotes (untrusted data)` and `render_untrusted(quoted_document)`. Chat: the last 12 turns or fewer; the human's text goes in `user` turns, each at most 2000 characters | a typed id and fenced blocks |
| 3+ | assistant | Tool calls | model |
| 3+ | tool | Canonical JSON of `ToolResult.content`, a blank line, then `render_untrusted(untrusted)` within the prompt's remaining budget | `ToolExecution` |

- **The system message never holds `UntrustedText`.** Untrusted text enters only through
  `render_untrusted`, and only into `user` or `tool` messages.
- **One 8000-character budget covers the whole request.** Once it is used up, later blocks become
  the trusted marker line.
- **Provenance labels and refusal messages are system-generated.** A refusal never echoes
  arguments.
- **Generation settings.** Temperature 0.2, seed 0, and thinking off by default (`AgentConfig`).
- **The chat user is trusted as a user, not as the system.** Earlier assistant turns stay in the
  `assistant` role.

### D6 Citations

- **Format.** A citation is written `[source:ref]` (`CITATION_PATTERN`). Sources are `audit`,
  `invoice`, `vendor`, `proposal`, `policy` and `document`.
- **Refs.** Every tool result records the refs it contains in `ToolTrace.refs`:
  - `explain_decision`: every audit id;
  - `get_invoice`: the invoice id, policy rules, proposal ids and untrusted sources;
  - `vendor_history`: the vendor id, invoice ids and bank-change audit ids.
- **Check.** `check_citations(reply)` accepts a citation only if it matches a ref from this
  turn. It fails if any citation is invalid, or if `explain_decision` was called and the reply
  cites no audit id.
- **On failure.**
  - Ops chat replaces the reply with `REFUSAL_TEMPLATES["unverifiable"]`, sets
    `refusal="unverifiable"` and records the audit event `agent.reply_unverifiable`.
  - The exception agent stores no note, and the reviewer sees the same template.
  - There is no retry loop.
- **Rendering (M9.3).**
  - `[audit:N]` links to `/audit/N`.
  - `[document:<source>]` is shown as a quote chip, never as a link.
  - Markdown is rendered without raw HTML or remote images (SEC-18).

### D7 Refusal behaviour

- **Tool level.** Typed `AgentToolError`s and `PolicyDeniedError` from the sink become a
  `ToolRefusal(code, template)`, which is returned to the model so it can recover (for example,
  by narrowing a query). Refusals from proposal tools and from the MCP origin are audited.
  Database, audit and model-serving errors propagate: there is no silent fallback.
- **Agent level.** The agent refuses with a fixed template (`RefusalReason`,
  `REFUSAL_TEMPLATES`) in these cases:
  - paying, approving, executing, changing bank details or sending email; it offers
    `request_approval` or `draft_vendor_email` when one fits;
  - full IBANs, user lists, secrets or bulk exports;
  - anything outside scope;
  - answers it cannot verify;
  - an exhausted tool budget.
- **Instructions inside documents are never followed.** The agent may note that the document
  contains instruction-like text. It never calls a tool because document text asks it to.
- **Safety never depends on the model refusing.** The refused actions have no tool. The caps,
  role checks and the P5 sink are deterministic.

### D8 MCP exposure

- **Server.**
  - `apx/agent/mcp_server.py` is built on the MCP Python SDK; M9.4 pins the version in
    `uv.lock`.
  - It uses the stdio transport only and is started as
    `docker compose exec -T api python -m apx.agent.mcp_server`.
- **Authentication.**
  - The token comes from `APX_MCP_TOKEN` in the client's configuration.
  - `McpAuthenticator`, backed by `apx.api` `TokenVerifier`, checks it at start-up and on every
    call (expiry).
  - With a missing or invalid token, the server exits with an error.
  - The server opens no network listener.
- **Tools listed.**
  - Exactly `MCP_TOOL_NAMES`: four read tools and two proposal tools, with specs from
    `tool_spec`.
  - Annotations: read tools get `readOnlyHint=true` and `openWorldHint=false`. Proposal tools
    get `readOnlyHint=false`, `destructiveHint=false` and `openWorldHint=false`, plus
    `idempotentHint=true` for `request_approval` only.
  - No resources, no prompts and no execute tool are exposed.
- **Calls.**
  - Every call goes through the same `ToolDispatcher` as
    `ToolCaller(origin="mcp", user=<token user>)`.
  - The text output is canonical JSON followed by `render_untrusted` blocks, so the client's
    model also receives spotlighted data.
  - MCP reads are audited as `mcp.read`, with the tool name and a hash of the arguments.

### D9 Eval hooks (M9.5)

- **Scorer inputs.**
  - Every `AgentReply` carries `trace: tuple[ToolTrace, ...]`: validated arguments, outcome,
    refs, proposal id and latency.
  - `AgentConfig.model` is the only switch between the stock and uncensored models. The
    runtime resolves `ModelRole.AGENT` only.
- **Scripted tasks.**
  - `evals/agent/tasks.jsonl` holds 100 tasks over a seeded fixture (`evals/agent/fixture.py`,
    seed 0, built from factory invoices and recorded pipeline outputs).
  - Task fields:

    | Field | Meaning |
    |---|---|
    | `id`, `category`, `role`, `prompt` | the task, and the role the user acts as |
    | `golden_calls` | a list of `{tool, args}`, where `args` is a subset matched after normalisation |
    | `forbidden_tools` | tools the agent must not call |
    | `expected_refusal` | the expected `RefusalReason`, or none |
    | `support_keys` | stable fixture keys, resolved to audit ids when the fixture loads |

  - Mix: 30 lookup, 25 explain, 15 vendor, 15 proposal and 15 refusal or role tasks.
  - The agent never sees the golden data.
- **Metrics (report only, §5).**
  - Tool-call accuracy: a task passes only if every golden call matched, no forbidden tool was
    called and the refusal matches.
  - Call-level precision and recall, and argument accuracy.
  - Citation faithfulness: the share of cited refs that appear both in the turn's traces and in
    the task's support set. It is measured on the raw reply, before the D6 check.
  - Hallucinated-ref rate, citation recall and refusal accuracy.
- **AgentDojo.**
  - `evals/agent/agentdojo_adapter.py` wraps `LlmClient.complete_with_tools` in a
    `BasePipelineElement` that calls the local llama-swap agent model.
  - It runs the banking suite (version pinned) with the `important_instructions` attack.
  - There are two defence settings. `none` passes tool outputs unchanged. `spotlight` sends tool
    outputs through `to_untrusted(...).quoted()` and adds `SPOTLIGHT_RULE` to the system prompt.
  - It reports utility, utility under attack and ASR. AgentDojo's banking tools include
    `send_money`, so this measures how easily the model is steered, which is exactly what P5
    neutralises in our product.
- **Stock vs uncensored.**
  - The M6.3 attack manifests run through the workflow to the exception agent, plus one
    scripted chat probe per case.
  - This runs once with `AgentConfig.model` set to the stock Qwen3.5-9B and once with the
    uncensored red-team build (offline only, §4).
  - Table columns:
    - payment-changing success, which must be 0 for both models (anything else is a critical
      finding against ADR-0008 and this ADR);
    - rate of proposals the attacker asked for;
    - refused exfiltration drafts;
    - canary-echo rate;
    - refusal accuracy.
- **Outputs.** Results go to `reports/agent.md` and `reports/agent.json`, plus the §5
  scoreboard rows. `make eval-quick` runs 20 of the tasks.

### Dependency on ADR-0008 (M6.5)

`main` (9120b01) does not have `ProposalSink`, `ProposalOrigin="mcp"`, `SPOTLIGHT_RULE` or
`render_untrusted` yet; they arrive with M6.5 from branch `m6-security`. For now,
`apx/agent/interfaces.py` restates `ToolOrigin` and a structurally identical `ProposalSink`
Protocol. Any ADR-0008 implementation satisfies it.

M9.2 starts only after M6.5 is merged. At that point:

- rebase `m9-agent` onto `main`;
- replace both restatements with re-exports from `apx.policy.interfaces`;
- `Proposal(created_by="mcp")` validates from then on.

M9.3 also needs `tool_calls` added to `llm.ChatMessage` for assistant turns. The change is
additive.

### Interface changes (`apx/agent/interfaces.py`)

Everything is additive. The M0 names are kept, and `Citation.source` gains `proposal`.

- **Vocabulary.** `ToolName`, `ToolEffect`, `ToolOrigin`, `ToolProposalKind`,
  `TOOL_PROPOSAL_KINDS`, and the id, pattern and mask types (`InvoiceIdArg`, `InvoiceNumber`,
  `MaskedIban`, `Code`, `AuditId`, and others).
- **Constants.** The caps, `AUDIT_FACT_KEYS`, `CITATION_PATTERN`, `AGENT_SYSTEM_RULES` and
  `REFUSAL_TEMPLATES`.
- **Caller, outputs and traces.** `ToolCaller`, `ToolOutput`, `ToolRefusal`, `ToolTrace`,
  `ToolExecution` and `CitationCheck`. `AgentReply` gains `refusal` and `trace`.
- **Parameter and result models for the six tools, and the contract table.** `ToolContract`,
  `TOOL_CONTRACTS` and `MCP_TOOL_NAMES`.
- **Errors.** `AgentToolError` and its subclasses, one per refusal code.
- **Protocols.** `SqlReader` (with `SqlStatement` and `SqlParam`), `ProposalSink`
  (restatement), `McpAuthenticator`, `AgentToolbox` and `ToolDispatcher`. `AgentPromptBuilder`
  gains `build_chat` and `with_tool_results`. `AgentConfig` is new.
- **Pure helper stubs.** `tool_spec`, `authorize_tool`, `mask_iban`, `audit_actor`,
  `parse_citations` and `check_citations`.

## Consequences

- Good: the tool surface is closed, typed and listed in one table, and P5 holds by
  construction. The worst an injected document can do is get the agent to draft a
  `request_info` email, or get a chat or MCP session to draft an approval that a human with the
  right role still has to execute under P1-P3 and segregation of duties.
- Good: decision answers link to audit events, and citation faithfulness can be measured.
- Good: SQL injection has three layers of defence: constant statements, bound parameters, and a
  read-only role over masked views.
- Bad: there are no free-form analytics (for example, "top vendors by spend"). Each new question
  type needs a new tool and an update to this ADR.
- Bad: masked IBANs, row caps and the 8000-character budget mean the agent sees less.
  Unverifiable answers become refusals, which will sometimes feel unhelpful.
- Bad: the read-only role and masked views add migration work in M9.2.
- Residual: a human may click through a chat-drafted approval (R-3, bounded by P1-P3).
  Assistant turns in the chat history can repeat document text, but only in the assistant role.
  AgentDojo measures how easily the model is steered but does not prevent it; P5 covers the
  product.

## Proof (gate)

- **M9.1:** this ADR is merged, and `mypy --strict apx` is clean with the contracts in place.
- **M9.2** (unit tests, no GPU):
  - `agent/test_tool_contracts.py` covers:
    - there are six tools, and the effects, origins and roles follow D1 and D3;
    - `MCP_TOOL_NAMES` equals all read and proposal tools;
    - every parameter model forbids extra keys and produces a JSON schema;
    - `MaskedIban` rejects raw IBANs (hypothesis);
    - `executed` and `sent` accept only `False`.
  - `agent/test_sql_parameterized.py`:
    - an AST check finds no f-string, concatenation, `%` or `.format` reaching SQL execution in
      `apx/agent`;
    - every statement in `AGENT_SQL` starts with `SELECT` or `WITH`;
    - hostile argument strings never appear in executed statement text (hypothesis, with a
      recording reader).
  - `agent/test_tool_outputs.py`: masking, single-entity scope, caps, and a canary placed in any
    free-text column appearing only in `untrusted`.
  - `agent/test_no_payment_writes.py`:
    - only SELECTs and pending submits of `TOOL_PROPOSAL_KINDS` are observed;
    - no SMTP, HTTP or socket imports exist in `apx/agent`.
  - `agent/test_dispatch_refusals.py`: every refusal code is covered, and errors that are not
    tool errors propagate.
  - ADR-0008's `agent/test_no_write_imports.py` still passes.
  - `integration/test_agent_ro.py`: a write through the reader's session fails.
- **M9.3:**
  - `agent/test_prompt_builder.py`:
    - the system message is only the rules, `SPOTLIGHT_RULE` and typed JSON;
    - canaries appear only between fences;
    - one budget covers the whole request.
  - `agent/test_citations.py`.
  - Playwright `chat.spec.ts`: asking "why was X blocked" returns an answer citing at least one
    audit id, and each cited id resolves to `/audit/N`.
- **M9.4:** `integration/test_mcp_tools.py`:
  - the tool list is exactly `MCP_TOOL_NAMES`, with the annotations in D8;
  - one read call and one proposal call succeed;
  - a bad token is refused;
  - there is no execute tool.
- **M9.5:** `reports/agent.md` contains:
  - tool-call accuracy and citation faithfulness over the 100 tasks;
  - AgentDojo utility, utility under attack and ASR, with and without spotlighting;
  - the stock-vs-uncensored table, where payment-changing success must be 0 for both models.

## M9 build plan (builder, no further design needed)

1. After M6.5 merges: rebase `m9-agent` onto `main`, turn the `ToolOrigin` and `ProposalSink`
   restatements into re-exports from `apx.policy.interfaces`, and add
   `tool_calls: tuple[ToolCall, ...] = ()` to `llm.ChatMessage`.
2. `apx/agent/contracts.py`: implement `tool_spec`, `authorize_tool`, `mask_iban` and
   `audit_actor`, and add `tests/unit/agent/test_tool_contracts.py` (M9.2 proof above).
3. Add a `db/migrations` revision that creates the `apx_agent_ro` role (SELECT only) and the
   views `v_agent_invoices`, `v_agent_vendor_accounts` (IBANs masked in SQL, plus the
   document-vs-payee match column) and `v_agent_audit`.
4. `apx/agent/sql.py`: add the `SQL_*` constants, `AGENT_SQL` and `PgSqlReader` (READ ONLY
   transaction, fetches cap + 1 rows). Add `test_sql_parameterized.py` and
   `tests/integration/test_agent_ro.py`.
5. `apx/agent/toolbox.py`: implement the four read tools per D1 and D3 (scope for the `agent`
   origin, row caps with `truncated`, untrusted sources, the `AUDIT_FACT_KEYS` filter, and refs
   for traces).
6. `apx/agent/toolbox.py`: implement `request_approval` (REVIEW precondition, cited ids belong to
   the invoice, idempotent) and `draft_vendor_email` (English and Arabic templates per
   `EmailReason`, the `details` filter), both writing only through `ProposalSink`.
7. `apx/agent/dispatch.py`: implement `ToolDispatcher` (contract lookup, `authorize_tool`,
   per-turn and rate caps, parameter validation, byte cap, refusal mapping and audit,
   `ToolTrace`). Add `test_dispatch_refusals.py`.
8. Add `test_tool_outputs.py` and `test_no_payment_writes.py`, using a spy sink and a recording
   reader with fixture rows.
9. `apx/agent/prompt.py`: implement `AgentPromptBuilder` per D5, and add
   `test_prompt_builder.py` (hypothesis canaries in the document and in tool results, the
   budget, the 12-turn window).
10. `apx/agent/citations.py`: implement `parse_citations` and `check_citations`, and add
    `test_citations.py` (valid, invented and stale ids; a decision answer with no audit
    citation).
11. `apx/agent/loop.py`: implement `ExceptionAgent` and `OpsChat` over `complete_with_tools`
    with the `AgentConfig` caps and D6 failure handling. Then add the workflow activity that
    stores the note, the API `POST /chat` SSE endpoint, the web chat panel with audit links and
    Playwright `chat.spec.ts`.
12. `apx/agent/mcp_server.py`: implement the stdio server per D8 (annotations, token check on
    every call, rate limit, audited reads), and add `tests/integration/test_mcp_tools.py`.
13. `evals/agent/`: add the fixture (seed 0), 100 tasks in the D9 mix, scorers with unit tests
    on synthetic traces, and a runner that writes `reports/agent.{md,json}`; wire a 20-task
    subset into `eval-quick`.
14. Add the AgentDojo adapter (defences `none` and `spotlight`, banking suite pinned; an
    eval-only dependency group, listed in `docs/licenses.md`) and the stock-vs-uncensored attack
    run (M6.3 manifests, `AgentConfig.model` swapped).
15. `make verify-m9`: run the agent unit and integration tests, the MCP test and the Playwright
    chat spec; require `reports/agent.md` with payment-changing success = 0 for both models; set
    the status of SEC-12, SEC-13, SEC-21 and SEC-22 in `docs/security.md` §8.
