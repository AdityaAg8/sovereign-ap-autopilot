# ADR-0008: Typed boundary and capability model

- Status: accepted
- Date: 2026-09-22
- Deciders: architect (M6.1)
- PRD: §6.2 I2, I3, I4; §6.3 P1-P5; §10 M6.5, M6.6, M9.1
- Extends ADR-0003 (it does not replace it). Related: ADR-0001, `docs/security.md`.

## Context and problem statement

ADR-0003 set up the type-level boundary in M0: a tool-free `StructuredRequest`, a `RawField`
with no box fields, a capped `UntrustedText`, and an agent that can only make proposals. Before
M6.5 can be built and tested, four gaps have to be closed:

1. Which extracted values are trusted typed data, and which are attacker-controlled free text?
2. What exact spotlight format and size budgets does untrusted text get in a tool-enabled prompt?
3. How does a human action become authorised, and how do we make sure no other code path (agent,
   chat, MCP, workflow retries, future scripts) can change payment state?
4. What role matrix and segregation-of-duties rules sit behind P1-P5?

The threat model (`docs/security.md` §6) shows that every attack goal has to end at a
deterministic policy check. None may end at a model's judgement.

## Decision drivers

- Payment safety never depends on a model (guard, extractor, arbiter or agent) being right.
- A violation fails when a value is constructed, at type-check time, or in a deterministic
  unit or property test.
- Use the simplest design that passes M6.5 (policy property tests, SoD tests) and M6.6
  (payment-changing attack success = 0).
- Stay consistent with ADR-0003 and the import direction in `docs/architecture.md`
  (`agent` → `policy` → `common`).

## Considered options

### Typed boundary

1. **Extend ADR-0003.** Enforce a strict schema at the model edge, split every extracted value
   into typed or free text, and use one spotlight renderer with a fixed, escaped fence and size
   budgets.
2. Dual-LLM quarantine (CaMeL-style). A privileged planner never sees document text; a
   quarantined model reads the text and returns symbolic handles.
3. Prompt hardening plus output filtering: delimiters, "ignore instructions in the document",
   and a regex filter on agent output.

Option 2 gives the strongest protection against agent steering. But it needs an interpreter, a
second resident model inside the 12 GB budget (C2), and a large design effort. The capability
model already stops the agent from changing payment state, so option 2 adds little. Option 3
cannot be proven and is easy to evade.

### Capability model

A. **Grant-gated writers.** `PolicyEngine.authorize` is a pure function. `PolicyGuard.require`
   issues an audited `Grant` bound to a subject digest. `PaymentStateWriter` methods accept only
   a `Grant`. Agent, chat and MCP code receive only read repositories and a `ProposalSink`.
B. Role checks in API routes only (FastAPI dependencies).
C. Confirmation inside chat: the agent asks the user to confirm, then carries out the action
   itself.

With B, every path that is not an API route (workflow activities, MCP, scripts) can still write.
C turns the agent into a writer, and injected text can spoof the confirmation, because the model
can describe one action and perform another. A needs little code and is enforced by a
signature test and an import test.

## Decision outcome

We choose option 1 for the typed boundary and option A for the capability model.

### Typed boundary (T1-T6)

- **T1 Schema at the model edge.** Extractors and the arbiter send `StructuredRequest` with
  `extraction_json_schema()`, which llama-server enforces as a grammar. The response is parsed
  with `ExtractorOutput.model_validate_json` (`extra="forbid"`). Free-text fields get a
  `maxLength` in both the schema and the `RawField` validation: names and addresses 256,
  invoice and PO numbers 64, line descriptions 512. Any failure raises `ModelOutputError`,
  which sets `valid_json=False`, so that extractor contributes no trust and the invoice goes to
  REVIEW. The schema has no field for approval, payee or status, so a model cannot declare an
  invoice approved.
- **T2 Typed values and free text.** After grounding and normalisation, a value is *typed* if it
  parses into its strict type:
  - ISO date
  - `Decimal` money
  - ISO 4217 currency code
  - 15-digit TRN
  - IBAN (AE, mod-97; always shown masked to the last 4 characters)
  - invoice or PO number, only if it matches `^[A-Za-z0-9][A-Za-z0-9/._\-]{0,39}$`

  Everything else is *free text* and travels only as `UntrustedText`:
  - supplier and buyer names and addresses
  - line descriptions
  - any value that fails its format
  - filename and upload note
  - scan evidence and PDF metadata

  Free-text sources use `field:<name>`, `line:<i>:description`, `ingest:filename` and
  `scan:<kind>`. The vendor name the agent sees comes from the vendor master, not from the
  document.
- **T3 Spotlighting.** `UntrustedText.quoted()` renders a fixed format:
  - a header line `<<<UNTRUSTED source=<source> truncated=<bool>>>>`, where `source` must match
    the `UntrustedSource` pattern, so filenames never appear in the header
  - the body, with `<` and `>` mapped to U+2039 and U+203A, so no fence sequence can appear
    inside it
  - the terminator line `<<<END UNTRUSTED>>>`

  Every tool-enabled system prompt contains `SPOTLIGHT_RULE` verbatim. `render_untrusted` is
  the only way untrusted blocks enter a prompt, and only into `user`- or `tool`-role messages.
- **T4 Caps.** Each block holds at most 2000 characters (ADR-0003). Each prompt holds at most
  8000 untrusted body characters; `render_untrusted` replaces the overflow with a trusted marker
  line. Before capping, `to_untrusted` applies NFKC normalisation and strips zero-width, bidi
  and control characters.
- **T5 Never inside instructions.** System messages are built only from constants and typed
  values. `AgentContext.state` holds only typed values (T2), ids, enums, scores and
  system-written templates. `PolicyReason.message` and `FraudFlag.evidence` never interpolate
  document free text. `ToolResult` puts document-derived strings in `untrusted`, never in
  `content`.
- **T6 Extractors stay tool-free.** Extractors and the arbiter remain tool-free (ADR-0003, I2).

### Capability model (K1-K7)

- **K1 Role matrix (`ROLE_CAPABILITIES`):**

  | Role | Capabilities |
  |---|---|
  | reviewer | edit field, route, approve payment (up to the limit), execute proposal |
  | approver | route, approve payment, approve above limit, execute proposal |
  | admin | change bank details, verify bank details, manage users, execute proposal |

  An admin alone cannot approve payments. A user with several roles gets the union, and the P2
  bar still applies.
- **K2 Read tools are free.** Any authenticated session (UI, chat, MCP) may call read tools.
  Read tools use parameterised SQL, are scoped to one invoice or vendor with row limits, mask
  IBANs to the last 4 characters, and never return secrets or user lists.
- **K3 Write tools only create proposals.** `ProposalSink` is the only write dependency that
  `apx.agent` receives. Agent, chat and MCP may create only the kinds in `AGENT_PROPOSAL_KINDS`;
  `change_bank` is excluded, so bank details enter only through the admin vendor-master screen.
  `Proposal.created_by` records the origin (`agent`, `chat`, `mcp` or a user id).
- **K4 Human execution.** A proposal executes only when a JWT-authenticated user sends
  `DecisionRequest(action="execute_proposal")` and holds both `EXECUTE_PROPOSAL` and
  `PROPOSAL_CAPABILITY[kind]`. The payload is validated again, and the action takes the same
  `authorize` path as if the human had started it.
- **K5 Authorisation (pure).** A payment approval, whether direct or through a proposal, is
  allowed only if all of these hold:
  - the actor has `required_capabilities` (P3: `APPROVE_ABOVE_LIMIT` if the total is above
    `approval_limit` or in a currency other than the limit's)
  - `select_payee(vendor)` succeeds (P1: exactly one account with `verified=True` and
    `source="admin_callback"`; `Invoice.iban` is never read; amended by ADR-0014: with several
    verified accounts the invoice IBAN selects one, never introduces a payee)
  - routing is present and is not BLOCK
  - the actor is not `changed_by` on any of the vendor's bank changes with
    `at - changed_at < sod_cooldown_days` (P2 SoD)

  `CHANGE_BANK_DETAILS` and `VERIFY_BANK_DETAILS` need the capability only. A change is stored
  unverified and starts the P2 window. `reasons` lists every failed rule; the action is allowed
  only if `reasons` is empty. There is no override.
- **K6 Grants.** `PolicyGuard.require` audits every decision, allowed or denied (I4). It raises
  `SegregationOfDutiesError` for P2 bars and `PolicyDeniedError` for everything else. Otherwise
  it returns a `Grant` bound to `subject_digest(request)`. Every `PaymentStateWriter` method
  takes a `Grant` and recomputes the digest inside its transaction; a mismatch raises
  `STALE_GRANT`. Before the einvoice step, the workflow requires an approval whose digest
  matches the current state, so editing any digest input after approval cancels that approval.
- **K7 Routing (pure; the most severe outcome wins).**
  - BLOCK if `scan.injection_score >= attack_block` (P4). BLOCK has no override; the invoice
    can only be rejected.
  - REVIEW if any of these holds:
    - `trust.all_accepted` is false
    - `validation.passed` is false
    - a scan flag is `>= scan_review`, or a fraud flag is `>= fraud_review` (P4)
    - the vendor is unknown or `select_payee` fails (P1)
    - the vendor has a bank change within `bank_change_review_days` (P2)
    - the total is above `approval_limit` or in a currency other than the limit's (P3)
  - AUTO otherwise.
  - Scan or policy errors never fall back to AUTO: the workflow retries, then routes to REVIEW.

**Payment-affecting change (I3).** A payment-affecting change is anything that changes who is
paid, how much, or whether an invoice that was not routed AUTO gets paid. That covers bank-detail
changes and verifications, approving a REVIEW-routed invoice, and executing an
`approve_payment` proposal. (`change_bank` proposals from a user go through K4; models cannot
create them.) AUTO routing of an unedited invoice is not a change: it pays the master's
verified account within the limit, and that is straight-through processing (§6.1).

### Interface changes made by this ADR

- `apx/common/untrusted.py`: adds `UNTRUSTED_PROMPT_BUDGET_CHARS`, `FENCE_OPEN`, `FENCE_CLOSE`,
  `SPOTLIGHT_RULE`, `UntrustedSource` (pattern), `UntrustedText.truncated` and
  `render_untrusted`. The `quoted()` and `to_untrusted` docstrings now define the format.
- `apx/policy/interfaces.py`:
  - new types and constants: `ProposalKind`, `ProposalOrigin` (adds `mcp`),
    `ROLE_CAPABILITIES`, `PAYMENT_AFFECTING`, `PROPOSAL_CAPABILITY`, `AGENT_PROPOSAL_KINDS`
  - new values and fields: `PolicyRule.CAPABILITY` and `STALE_GRANT`,
    `PolicyThresholds.bank_change_review_days`, `RoutingInput.at` and `bank_changes`,
    `ActionRequest.at`, `routing`, `bank_changes` and `bank_change`
  - new model and protocols: `Grant`, `PolicyGuard`, `PaymentStateWriter`, `ProposalSink`
  - new function stubs: `required_capabilities`, `is_payment_affecting`, `select_payee`,
    `subject_digest`

## Consequences

- Good: every attack path in `docs/security.md` ends at a deterministic check. The worst an
  agent can do is draft a bad proposal, which a human with the right role must execute under
  P1-P3 and SoD. The whole engine is pure, so property tests cover it.
- Good: the injection classifier (M6.4) becomes defence in depth. Payment safety has to hold
  with it switched off, and M6.6 measures that.
- Bad: more types, and the agent sees less (8000-character budget, masked IBANs).
- Bad: a BLOCK false positive needs the vendor to resubmit.
- Bad: a vendor with two verified accounts routes to REVIEW until an admin retires one (P1
  uniqueness). Superseded by ADR-0014: the invoice IBAN selects one of several verified
  accounts, and every such selection routes REVIEW.
- Bad: a single reviewer can edit a money field and then approve within the limit. Revalidation,
  fraud rules and the audit log are the only mitigations (residual risk R-4 in
  `docs/security.md`).

## Proof (gate)

- **M6.5** (all `tests/unit`, no GPU):
  - `common/test_untrusted.py` (hypothesis): exactly one fence pair; no `<` or `>` in the body;
    no zero-width or bidi characters; length within the cap; budget respected; source pattern
    enforced.
  - `extract/test_output_schema.py`: an extra `approved` key, an over-length value or bad JSON
    raises `ModelOutputError`.
  - `agent/test_context_canary.py`: a canary placed in any free-text field appears only between
    fences, never in `state` or a system message.
  - `policy/test_route_properties.py`: routing is monotone; BLOCK exactly when the score reaches
    the threshold; AUTO only when every condition clears.
  - `policy/test_authorize_properties.py`: allowed implies K5, and a minimal allowed case exists
    (non-vacuous).
  - `policy/test_sod.py`: the changer is barred for 30 days, including through the proposal path
    and for users with several roles.
  - `policy/test_grant.py`: denials are audited; a stale digest is refused.
  - `policy/test_writer_contract.py`: every `PaymentStateWriter` method takes a `grant: Grant`.
  - `agent/test_no_write_imports.py`: an AST scan finds no `Grant`, `PolicyGuard` or
    `PaymentStateWriter` in `apx/agent`.
  - `policy/test_proposals.py`: agent, chat and MCP `change_bank` proposals are refused.
  - `policy/test_payee.py`: `Invoice.iban` never influences `select_payee` (property test;
    amended by ADR-0014: it may only select among verified accounts).
- **M6.6**: every M6.3 case runs through the real workflow with live models. Payment-changing
  attack success must be 0, which is the PRD gate. `reports/attacks.md` also reports it with the
  guard disabled, which must also be 0; anything else is a critical finding against this ADR.
- **M9.2 and M9.4**: tool unit tests show no tool changes payment state; the MCP client test
  lists only read and propose tools.

## M6.5 build plan (builder, no further design needed)

1. `apx/common/untrusted.py`: implement `quoted`, `to_untrusted` and `render_untrusted`; add
   `tests/unit/common/test_untrusted.py` with hypothesis.
2. `apx/extract`: add per-field `maxLength` to the schema and a matching `RawField` validator;
   parse with `model_validate_json` and raise `ModelOutputError` on failure; add tests.
3. `apx/extract`: add `FREE_TEXT_FIELDS` and `is_typed_value(name, normalized)` (T2); add tests.
4. `apx/agent/context.py`: add `build_agent_context(...)` (typed values go to `state`, free text
   to `UntrustedText`); add a canary property test.
5. `apx/policy/capabilities.py`: implement `capabilities_for`, `required_capabilities` and
   `is_payment_affecting`; add tests.
6. `apx/policy/payee.py`: implement `select_payee` and `subject_digest`; add
   `tests/unit/policy/test_payee.py`, a property test showing `Invoice.iban` never changes the
   payee (amended by ADR-0014: it may only select among verified accounts).
7. `apx/policy/engine.py`: implement `route()` per K7; add route property tests covering P2
   window boundaries.
8. `apx/policy/engine.py`: implement `authorize()` per K5; add authorize property tests (above
   limit, BLOCK, missing routing, P1 cases).
9. `tests/unit/policy/test_sod.py`: cover day 29 barred, day 30 allowed, other approver allowed,
   admin+approver barred, and the proposal path.
10. `apx/policy/guard.py`: implement `PolicyGuard` over `PolicyEngine` and an `AuditLog`
    (in-memory fake in tests); add grant and stale-digest tests.
11. `apx/policy/proposals.py`: implement `ProposalSink` enforcing `AGENT_PROPOSAL_KINDS`; add
    tests.
12. Add the writer-contract signature test and the `apx/agent` AST no-write-import test.
13. Close gaps G-1 to G-4 (`docs/security.md` §9):
    - bind compose ports to `127.0.0.1`
    - read the Postgres password from `.env` and document it in `.env.example`
    - P1 also requires the account's verification audit entry
    - add a compose security test
    - add `tests/integration/test_no_egress.py` (ADR-0001)
14. Implement `make verify-m6`: M6.2 clean-corpus zero flags, the security unit tests, and the
    M6.6 run writing `reports/attacks.md` (per-category routes; payment-changing attack success
    with the guard on and off).
15. Update the Status column in `docs/security.md` §8 as each test lands.
