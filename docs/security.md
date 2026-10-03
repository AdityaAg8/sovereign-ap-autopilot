# Security: threat model

Threat model for Sovereign AP Autopilot (M6.1).

- Sources: PRD §2 (C1, C5, C6), §6.2 (I1-I6), §6.3 (P1-P5); ADR-0001 (local-only runtime),
  ADR-0003 (trust boundary), ADR-0008 (typed boundary and capability model), ADR-0014 (P1 with
  several verified accounts, payee pin from routing to build).
- Test paths below are planned unless marked *exists*. §8 records each control's status, and
  the builder updates it as tests land.

## 1. Security objectives

The product must never do any of the following. The first two are money outcomes, and they
must hold even if every model is fully compromised by a malicious document.

1. **O1: No redirected payment.** Nothing is paid to an account other than the vendor's
   admin-verified account.
2. **O2: No fraudulent approval.** No fake, duplicate, inflated or blocked invoice is paid
   unless a human with an allowed role has approved it under P1-P3 and SoD.
3. **O3: No data leaves.** No invoice, vendor or bank data leaves the host, and none is exposed
   to anyone who is not an authenticated user of this installation.

Design thesis (ADR-0008): every path to O1 or O2 ends at a deterministic check in
`apx/policy`. Models (guard, extractors, arbiter, agent) can lower or raise suspicion, but they
cannot authorise anything.

## 2. Assets

| Asset | Where it lives | Priority | Who may change it |
|---|---|---|---|
| Payments (approval state, payee, amount, e-invoice send) | Postgres `approvals`, `decisions`, `invoices`; Temporal state; mock ASP | Integrity (critical) | Policy AUTO route; a human through `PolicyGuard` Grant |
| Vendor bank master | Postgres `vendors`, `bank_details` (`verified`, `source`, history) | Integrity (critical), confidentiality | Admin (change); admin callback record (verify, P1) |
| Approval rights (users, roles, JWT secret) | Postgres `users`; `.env` secret | Integrity, confidentiality | Admin (`MANAGE_USERS`); host operator (`.env`) |
| Audit log | Postgres `audit_log` hash chain (§6.4) | Integrity, non-repudiation | Append-only by the system; nobody edits it |
| Models | `models/` weights, `infra/models.lock.json`, MLflow `champion`/`challenger` aliases | Integrity | `make models` (sha256 verified); promotion gates (M10.2) |
| Data | `data/` invoices, parse output, labels, eval/cal/test splits; Phoenix traces; logs | Confidentiality, integrity (I5) | Pipeline; reviewers (labels) |

## 3. Attackers

| Attacker | Capability | Main goals |
|---|---|---|
| External document attacker (fraudulent or compromised vendor, BEC) | Submits any PDF or image, filename or note through upload or the watch folder | O1, O2 |
| Malicious or compromised insider (reviewer, approver, admin) | Valid account; uses the UI, API, chat and MCP within its role | O1, O2, O3 |
| Local-network attacker (same LAN or Wi-Fi) | Reaches any port published on a non-loopback interface | O1 (database tampering), O3 |
| Malicious local web page or MCP client | Browser requests to `localhost` (CSRF, DNS rebinding); a local process speaking MCP | O2, O3 |
| Supply chain (setup time) | Tampered model file, image or package | All |

Out of scope:

- compromise of the host OS or Docker Desktop (root or administrator)
- physical theft (full-disk encryption is recommended in the runbook)
- real payment rails and real Peppol transmission (PRD §1)
- multiple tenants

## 4. Trust boundaries

```mermaid
flowchart LR
  ATT([document / filename / note]) -->|B1| IN[ingest + scan + parse]
  IN -->|B2 text + images| EX[Extractors A/B, arbiter<br/>tool-free]
  EX -->|B3 JSON| TF[typed fields<br/>grounded, trust-gated]
  TF -->|B4| PO{policy P1-P5}
  PO -->|B5| HU[human reviewer / approver]
  TF -.->|B6 UntrustedText only| AG[agent / ops chat]
  AG -->|proposals only| PS[(proposals)]
  MC([MCP client]) -->|B7| AG
  BR([browser]) -->|B8 JWT| API[api] --> PO
  subgraph edge[B9 host edge: published ports, internal network]
    API
  end
```

| ID | Boundary | Crossing | Trust change |
|---|---|---|---|
| B1 | Document → ingest, scan, parse | Bytes, filename, upload note | Attacker bytes become a stored, scanned, parsed document |
| B2 | Parse → extractors and arbiter | Words, lines, page images | Attacker text reaches a model with no tools (I2) |
| B3 | Extractor output → typed fields | JSON → `RawField` → `FieldValue` | Model claims become schema-checked, grounded (I1), trust-gated values |
| B4 | Typed fields → policy | `RoutingInput` | Values become an AUTO, REVIEW or BLOCK decision |
| B5 | Policy → human | Review UI, `DecisionRequest` | A human decision becomes a Grant and then a payment-state change |
| B6 | Agent and ops chat | `UntrustedText` in; tool calls out | Untrusted text reaches a tool-calling model; the model can read or propose only |
| B7 | MCP | MCP tool calls | An outside process reaches the same read and propose tools |
| B8 | Web/API | HTTPS/JSON with a JWT bearer | A browser becomes an authenticated role-holder |
| B9 | Gateway / edge network | Published ports, internal Compose network, setup-time egress | Host and LAN reach services; services cannot reach the internet |

There is no reverse proxy. The edge (B9) is Docker Desktop's published-port proxy plus the
`internal: true` app network (ADR-0001).

## 5. STRIDE per boundary

Control IDs refer to the catalogue in §8.

### B1 Document → ingest, scan, parse

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| S | Letterhead or name impersonates a known vendor (A07) | Vendor resolved by TRN against the master, not by name; lookalike check (SEC-15); P1 (SEC-08) | `fraud/test_rules.py`; `policy/test_authorize_properties.py` |
| T | Hidden content makes models read something different from what humans see (A01-A06) | Hidden-content scan, fail-closed (SEC-02); OCR-based parse does not see invisible text, so OCR support drops (SEC-06, SEC-07) | `scan/test_<technique>.py`; M6.6 |
| T | Malformed PDF, decompression bomb, huge page count | Type sniffing, ≤ 20 MB, ≤ 30 pages, encrypted PDFs rejected; parser in its own container with no egress (SEC-01, SEC-19) | `integration/test_upload_hardening.py` (M11 #4) |
| R | Vendor denies sending a document | sha256 content-addressed original kept; ingest audit event (SEC-16, SEC-24) | `ingest/test_dedupe.py`; audit tamper test |
| I | PDF JavaScript, attachments or links try to phone home | Never executed; scan flags them (SEC-02); no egress (SEC-19) | `scan/test_active_content.py`; `test_no_egress.py` |
| D | Upload flood or page bombs | Size and page caps; one workflow per sha256 (SEC-01, SEC-24) | M5.5 dedupe test |
| E | Filename path traversal or filename injection (A11) | Content-addressed paths; filename never used in a path or instruction, only as `UntrustedText` `ingest:filename` (SEC-01, SEC-13) | `ingest/test_filename_untrusted.py` |

### B2 Parse → extractors and arbiter

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| E | Injected text makes the model call tools or act | `StructuredRequest` has no `tools` field (SEC-04) | *exists*: `test_interfaces.py::test_structured_request_has_no_tools_field` |
| T | Injection changes extracted values (A01-A12) | A/B agreement, grounding and trust gate (SEC-06, SEC-07); the invoice IBAN only selects among admin-verified accounts, never adds a payee (SEC-08, ADR-0014) | M7.5 simulation; M6.6 |
| D | Context bomb exhausts VRAM or time | Page cap, token budget, timeouts; `ModelServingError` → retry → REVIEW | M5.2 retry tests |
| I | Cross-invoice leakage through the model | Stateless calls; one invoice per request | Prompt-builder unit test |

### B3 Extractor output → typed fields

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| S | Model fakes where a value came from (boxes) | `RawField` has no boxes; boxes come only from parse words (SEC-06, I1) | *exists*: `test_interfaces.py::test_raw_field_cannot_carry_boxes`; `extract/test_grounder.py` |
| T | Extra keys (for example `"approved": true`) or oversized free text | Grammar schema, `extra="forbid"`, `maxLength`; failure → REVIEW (SEC-05, T1) | `extract/test_output_schema.py` |
| T | Plausible but wrong values | Learn-then-Test error bound α (SEC-07); validation and 3-way match | `trust/test_ltt_simulation.py` |
| R | Which model produced a value? | `llm_calls` rows (model, version, trace id); A, B and arbiter outputs kept (SEC-16) | M10.3 dashboard SQL check |
| D | Invalid JSON | `ModelOutputError` → `valid_json=False` → REVIEW, never AUTO | `extract/test_output_schema.py` |

### B4 Typed fields → policy

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| E | AUTO pays an unverified or unknown account | P1 in both `route` and `authorize` (SEC-08) | `policy/test_route_properties.py` |
| T | Amounts shaped to slip under limits (split invoices, just below the limit) | P3 (SEC-10); split rule (SEC-15); the most severe outcome wins (SEC-11) | `fraud/test_rules.py`; route properties |
| T | Stale input: bank change after routing | `authorize` re-reads current state; Grant digest (SEC-14) | `policy/test_grant.py` |
| R | Why was this invoice auto-approved? | `RoutingDecision.reasons` audited (SEC-16) | M9.3 `explain_decision` cites audit ids |
| D | Policy or scan error | Typed error → retry → REVIEW; never AUTO (K7) | route property test with a failed scan |

### B5 Policy → human

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| S | Impersonating an approver | JWT auth; actor always taken from the token (SEC-17) | `api/test_authz_matrix.py` |
| T | Approve, then edit the amount (TOCTOU) | Approval bound to `subject_digest`; writer refuses a stale digest (SEC-14) | `policy/test_grant.py` |
| T | Human deceived by the rendered document (A06 bidi, A09 stamp) | IBAN and amounts rendered isolated LTR with controls stripped; payee shown from the master; system status visually separate from document content; flag explanations (SEC-18, M8.3) | Playwright `security-ui.spec.ts` |
| R | Approver denies approving | Grant carries an audit entry id; denials also audited (SEC-16) | `policy/test_grant.py`; audit tamper test |
| E | Reviewer approves above the limit; bank changer approves or verifies | P3, P2 SoD; verifier ≠ changer (SEC-10, SEC-09, SEC-26) | M5.3 authz test; `policy/test_sod.py`; `integration/test_bank_master.py` |
| S, E | A raw Temporal `review` signal (Temporal UI, CLI) approves a payment with no approval record | `confirm_review` requires an `approvals` row whose `invoice_digest` matches the current invoice, or a recorded rejection; otherwise audited and ignored; Temporal UI unpublished by default and read-only under `devports` (SEC-27, SEC-19) | `integration/test_review_guard.py`, `unit/test_compose_security.py` |
| I | Reviewers browse full bank details | Role-scoped responses; IBAN masked except for admins (SEC-17, SEC-22) | `api/test_authz_matrix.py` |

### B6 Agent and ops chat

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| E | Indirect prompt injection steers the agent into acting (A08, A12) | Tools only read or propose; `apx.agent` has no Grant or writer (SEC-12, K3) | `agent/test_no_write_imports.py`; `policy/test_proposals.py` |
| T | Document text read as instructions | Typed/free-text split, spotlighting, caps, `SPOTLIGHT_RULE` (SEC-13, T2-T5) | `common/test_untrusted.py`; `agent/test_context_canary.py` |
| S | Agent proposal passed off as a human decision | `Proposal.created_by` in `agent`, `chat`, `mcp`; UI label; audit (SEC-12, SEC-16) | `policy/test_proposals.py` |
| I | Agent leaks data into chat or email drafts (G3) | Masked IBAN, scoped queries, drafts never sent, CSP (SEC-22, SEC-18) | `agent/test_tool_outputs.py` |
| R | Unsupported claims in answers | Citations to audit ids (M9.3) | Playwright "why was X blocked" |
| D | Tool loops, context flooding | Tool-call limit per turn; `UNTRUSTED_PROMPT_BUDGET_CHARS` (SEC-13) | `common/test_untrusted.py` |
| T | SQL injection through tool arguments | Parameterised SQL only; argument schema validation (SEC-22) | M9.2 tool tests; M11 #4 |

### B7 MCP

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| S | Unauthenticated local client | Local transport tied to an authenticated user; any HTTP transport bound to 127.0.0.1 with a JWT (M9.1 ADR) (SEC-21) | `integration/test_mcp_tools.py` |
| E | Client calls a write or execute tool | Registry exposes only `read` and `propose` tools; no execute tool (SEC-21, SEC-12) | M9.4 client test lists tools |
| I | Bulk export of vendor or bank data | Masking, row limits, single-entity scope (SEC-22) | `agent/test_tool_outputs.py` |
| R | Who proposed what? | `created_by="mcp"` plus user id in the audit (SEC-16) | `policy/test_proposals.py` |
| D | Request flood | Per-session rate and tool-call limits | M9.4 test |

### B8 Web/API

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| S | Password guessing on `POST /auth/login` | Per (client IP, email) token bucket, burst 5, refill 5/min, 429 with `Retry-After`; LRU-bounded buckets; constant-time verify for unknown emails (SEC-29) | `unit/test_api_login_throttle.py` |
| S | Token forgery or theft | JWT with a pinned algorithm, secret from `.env` (missing or short → `ConfigError`), short expiry; bearer header rather than cookies, so no CSRF (SEC-17, SEC-23) | `api/test_authz_matrix.py` |
| S | DNS rebinding against `localhost` | Host-header allowlist plus auth (SEC-17) | `api/test_trusted_host.py` |
| T | SQL injection, mass assignment | Pydantic bodies with `extra="forbid"`; parameterised SQL (SEC-17) | M11 #4 SQL injection attempts |
| E | Role check only in the UI | Server-side `PolicyGuard` on every state change (SEC-14, SEC-17) | `api/test_authz_matrix.py` |
| I | XSS through document text in the UI | React escaping, no raw HTML, CSP `default-src 'self'`, pdf.js scripting off (SEC-18) | Playwright `security-headers.spec.ts` |
| D | Upload bombs, SSE connection exhaustion | Upload limits (SEC-01); raster caps on page size, render DPI and pixels and embedded-image pixels at ingest, scan and parse (SEC-28); connection caps | M11 #4; `unit/test_raster_caps.py` |
| R | Disputed action | Audit with the actor taken from the token (SEC-16) | Audit tamper test |

### B9 Gateway / edge network

| STRIDE | Threat | Controls | Proof |
|---|---|---|---|
| I, T | LAN or host reaches Postgres, Temporal, Phoenix, MLflow or llama-swap on published ports | Only the gateway publishes, on 127.0.0.1; Postgres (5432) and the Temporal UI (8233) are published only by the opt-in `devports` profile, whose Temporal UI proxy passes GET/HEAD only; `POSTGRES_PASSWORD`, `AGENT_RO_PASSWORD` and `API_JWT_SECRET` are generated into `.env` by `make env`, Compose has no fallback, and no code or config ships a default password (SEC-19, SEC-23; G-1, G-2 closed in 6971630) | `unit/test_compose_security.py`, `unit/test_db_passwords.py`, `unit/test_db_session.py`, `unit/test_init_env.py` |
| I | Runtime egress to the internet | `internal: true` app network (ADR-0001) (SEC-19). **Test missing: G-4** | `integration/test_no_egress.py` |
| T | Tampered model or image at setup time | `models.lock.json` revision plus sha256; read-only mounts; `uv.lock` (SEC-20) | *exists*: `test_download_models.py` |
| E | Container breakout from a parser exploit | No privileged containers; non-root app users; read-only model mounts (SEC-19) | `test_compose_security.py` |
| D | Anything on the host floods llama-swap | Loopback-only port; the GPU lock serialises stages (ADR-0002) | `test_compose_security.py` |

## 6. Attacker goals and attack trees

Notation: `[OR]` means any child achieves the parent; `[AND]` means all children are needed.
Each leaf lists its attack categories (§7), then the control that stops it (✗) and its test.

### G1 Redirect a payment (O1)

```
G1 Redirect a payment to an attacker account                                    [OR]
├── G1.1 Get an attacker IBAN used as payee                                     [OR]
│   ├── G1.1a IBAN in the document (A08 visible; A01-A04 hidden; A05/A06 obfuscated)
│   │         ✗ SEC-08 P1: payee = select_payee(vendor master, invoice IBAN); the invoice IBAN
│   │           only selects among admin-verified accounts, never adds one (ADR-0014)
│   │           SEC-08 invoice IBAN ≠ sole verified account, or several verified → REVIEW
│   │           SEC-15 invoice IBAN ≠ master → BANK_CHANGE flag → REVIEW (P4)
│   │           proof: policy/test_payee.py (property), M6.6 per-category ASR
│   ├── G1.1b Make an extractor emit a payee or "approved" field
│   │         ✗ SEC-05 schema has no such field, extra="forbid" → test_output_schema.py
│   └── G1.1c Impersonate a known vendor (A07 homoglyph, A05 zero-width in name)
│             ✗ SEC-15 TRN-based match + confusable-skeleton lookalike → VENDOR_IMPERSONATION
│               SEC-08 a new vendor has no verified account → REVIEW, approval denied
│               proof: fraud/test_rules.py, policy/test_authorize_properties.py
├── G1.2 Change the vendor bank master                                          [OR]
│   ├── G1.2a Agent, chat or MCP makes the change (A08, A12 aimed at the agent)
│   │         ✗ SEC-12 tools read/propose only; change_bank not in AGENT_PROPOSAL_KINDS
│   │           proof: policy/test_proposals.py, agent/test_no_write_imports.py
│   ├── G1.2b Social-engineer an admin (BEC "our bank changed" letter)          [AND]
│   │   ├── admin records the change (CHANGE_BANK_DETAILS; stored unverified)
│   │   ├── admin records a callback verification (residual R-1)
│   │   └── a different approver approves while P2 forces REVIEW
│   │         ✗ SEC-09 P2 REVIEW for 30 days + SoD; SEC-15 bank-change flag; SEC-16 audit
│   ├── G1.2c Insider admin changes, verifies and self-approves
│   │         ✗ SEC-09 P2 SoD bars the changer for 30 days → policy/test_sod.py
│   │         ✗ SEC-26 the verifier must differ from the changer (BankMasterService and
│   │           PaymentStateWriter.verify_bank_account) → integration/test_bank_master.py
│   │           (collusion: residual R-2)
│   └── G1.2d Write the database directly (published Postgres port, default password)
│             ✗ G-1, G-2 closed: 5432 unpublished by default, generated passwords, no default
│               G-3 verification must reference a bank_verifications row
│               proof: test_compose_security.py, test_db_passwords.py, audit verifier
└── G1.3 Swap payee or amount after approval (TOCTOU)
          ✗ SEC-14 approval bound to subject_digest; stale grant refused → policy/test_grant.py
          ✗ SEC-08 payee pinned at routing (RoutingDecision.payee); build_einvoice re-selects and
            refuses a different payee (PayeeChangedError, non-retryable, audited; ADR-0014 §7)
            → test_money_path_review_fixes.py
```

### G2 Get a fraudulent invoice approved (O2)

```
G2 Approve a fake, duplicate, inflated or blocked invoice                        [OR]
├── G2.1 Straight-through AUTO                                                  [AND, all must pass]
│   ├── every field accepted by the trust gate        ✗ SEC-07 (hidden totals A01-A04 lack OCR support)
│   ├── validation + 3-way match against PO/GRN      ✗ validate (M4)
│   ├── fraud flags below threshold                  ✗ SEC-15 duplicate, split, new vendor high amount
│   ├── scan flags and injection score below review  ✗ SEC-02, SEC-03 (A01-A06, A10, A11, A12)
│   └── P1 verified payee, P2 no recent change, P3 ≤ limit ✗ SEC-08, SEC-09, SEC-10
│       "pre-approved" claims (A09 stamp, A12 text, A10 metadata) have no path into
│       routing: SEC-05 no approval field; SEC-11 route reads only typed reports
│       proof: policy/test_route_properties.py, M6.6
├── G2.2 Get a human to approve                                                  [OR]
│   ├── Deceive the reviewer with document content (A09 stamp, A06 bidi display)
│   │         ✗ SEC-18 system status and master payee shown separately; bidi isolation;
│   │           flag explanations (M8.3) → Playwright security-ui.spec.ts
│   ├── Agent recommends or proposes approval because of injection (A12)
│   │         ✗ SEC-12 proposal only; human with APPROVE_PAYMENT must click; K5 still applies
│   │           (click-through: residual R-3, bounded by P1-P3)
│   ├── Reviewer approves above the limit            ✗ SEC-10 P3 → M5.3 authz test
│   └── Approve a BLOCKed invoice                    ✗ K5 denies when route = BLOCK; no override
├── G2.3 Bypass policy through the API or MCP        ✗ SEC-14, SEC-17, SEC-21 → api/test_authz_matrix.py
└── G2.4 Replay or double payment (resubmit, retries)
          ✗ SEC-24 sha256 dedupe, DUPLICATE rule, idempotency keys, exactly-once ASP
            proof: integration/test_workflow_crash.py, ingest/test_dedupe.py
```

### G3 Exfiltrate data (O3)

```
G3 Get invoice, vendor or bank data off the host                                 [OR]
├── G3.1 Runtime egress (PDF JS, model, library telemetry)
│         ✗ SEC-19 internal network (ADR-0001) → integration/test_no_egress.py (G-4)
├── G3.2 Agent output rendered as a remote image or link carrying data (A12)
│         ✗ SEC-18 CSP img-src 'self' data: blob:, connect-src 'self'; markdown without raw
│           HTML or remote images; no egress anyway → security-headers.spec.ts
├── G3.3 Drafted vendor email carries data ("include all IBANs") (A12)
│         ✗ SEC-22 drafts never sent (no SMTP at runtime); tool outputs masked and scoped
│           → agent/test_tool_outputs.py
├── G3.4 MCP client bulk-reads data                    ✗ SEC-21, SEC-22 → integration/test_mcp_tools.py
├── G3.5 LAN reads service ports (Postgres, Phoenix traces with prompts, MLflow)
│         ✗ G-1, G-2 (open) → test_compose_security.py
├── G3.6 Data or secrets committed to git              ✗ SEC-23 .gitignore, pre-commit, gitleaks CI (C5)
└── G3.7 Logs and traces copied off the host           ✗ SEC-25 IBAN redaction; Phoenix loopback only
```

## 7. Attack categories (M6.3) → controls → proof

M6.6 reports, for every category, the route distribution and the payment-changing attack success
rate (ASR). It reports both with the guard on (the gate) and with the guard off (ADR-0008
evidence). Payment-changing ASR must be 0 in both runs.

| ID | M6.3 category | Goal | First line | Backstops (deterministic) | Unit proof |
|---|---|---|---|---|---|
| A01 | Hidden white text (invisible fill) | G1, G2 | SEC-02 invisible fill → REVIEW | SEC-06/07 no OCR support; SEC-08 P1 | `scan/test_invisible_fill.py` |
| A02 | Tiny font (< 4 pt) | G1, G2 | SEC-02 font size | SEC-07; SEC-08 | `scan/test_tiny_font.py` |
| A03 | Off-page or clipped text | G1, G2 | SEC-02 off-page/clip | SEC-07; SEC-08 | `scan/test_off_page.py` |
| A04 | Text behind an image | G1, G2 | SEC-02 covered by image | SEC-07; SEC-08 | `scan/test_behind_image.py` |
| A05 | Zero-width characters | G1 | SEC-02 zero-width flag | `to_untrusted` strip (SEC-13); normalised matching (SEC-15); SEC-08 | `scan/test_zero_width.py`; `common/test_untrusted.py` |
| A06 | Bidi override | G1, G2 | SEC-02 bidi control (normal Arabic RTL never flags) | SEC-18 LTR isolation of IBAN/amounts; SEC-08 | `scan/test_bidi.py`; `scan/test_arabic_clean.py` |
| A07 | Homoglyph vendor | G1 | SEC-15 VENDOR_IMPERSONATION | TRN-based vendor match; SEC-08 (no verified account) | `fraud/test_rules.py::test_vendor_impersonation` |
| A08 | IBAN-swap instruction | G1 | SEC-03 guard; SEC-15 BANK_CHANGE; SEC-08 invoice IBAN ≠ sole verified account → REVIEW | SEC-08 P1 (the invoice IBAN only selects among admin-verified accounts); SEC-12 no agent `change_bank`; SEC-09 P2 | `policy/test_payee.py`; `policy/test_proposals.py` |
| A09 | Fake approval stamp | G2 | SEC-05 no approval field | SEC-14 approval only by Grant; SEC-18 system status separate | `extract/test_output_schema.py`; `policy/test_grant.py` |
| A10 | Metadata injection (Info/XMP, annotations, attachments, JS) | G2, G3 | SEC-02 metadata/attachment/JS flags | Metadata never in extractor prompts; `UntrustedText` `scan:metadata` (SEC-13); SEC-19 | `scan/test_active_content.py` |
| A11 | Filename or email-body injection | G2, G3 | SEC-03 guard on `ingest:*` text | SEC-01 content-addressed paths; SEC-13 source pattern and fencing | `ingest/test_filename_untrusted.py`; `common/test_untrusted.py` |
| A12 | Instruction inside a line-item description | G1, G2, G3 | SEC-03 guard; SEC-13 free text fenced | SEC-04 tool-free extractors; SEC-12 proposal only; K5 | `agent/test_context_canary.py`; M9.5 agent attack table |

### M6.6 results

`make eval-attacks` (`scripts/attacks_e2e.py`, `evals/attacks_e2e/`) drops every case into the
running stack's inbox, approves each review-routed case through the API as the seeded approver
(the "approver clicks approve" path: P1 must still pay only a verified account), and writes
[`reports/attacks_e2e.md`](../reports/attacks_e2e.md) and `.json`: ASR overall and per category,
route and outcome counts, detection rate per control, what the approver saw, and every case by id.

- Cases are rendered from the committed manifests against the ERP `make seed` loads (seed 2026),
  so each attack names a real vendor with a verified account. The M6.3 renders in `data/attacks/`
  use a separate suite ERP (seed 6300) whose vendors the seeded master does not know, which would
  hold every case as an unknown vendor and test nothing past P1.
- Success is read from Postgres only, never from model text: the payee pinned at routing (exact
  account where it is resolvable, else its masked suffix) against the verified accounts at the
  run's start, the attacker IBANs in the payload, the paid vendor and total against the genuine
  invoice, approval amounts against the genuine total, approvals by anyone but the harness
  approver, executed money proposals, and any vendor bank-master write since the run's snapshot.
- Held means provably held: blocked, left in review, a typed refusal of the approver's click
  (403/409/422 with a string `detail`), or a typed money-path refusal after approval (audited
  `payee_changed`). A workflow `failed` without that (the crash path), an untyped approve error,
  a timeout or an invoice ingested before the drop (stale) is `inconclusive`.
- Detection rates count only attack-specific fraud rules (`bank_detail_change`, `shared_iban`)
  and are reported next to a clean baseline (`--baseline 24` unhooked factory invoices of the
  same vendors, dropped and approved the same way) and net of it. After `make seed`,
  `new_vendor_high_amount` fires on every first invoice per vendor above 10,000 AED (the seed
  loads no invoice history) and `price_variance_po` fires on most invoices (factory line prices
  are not drawn from the PO, and the invoice builder links line *i* to PO line *i* by position).
- Rows are keyed by the run identity: the host's defending files, the running api/worker/parse/
  page-embed/llama-swap image ids and the served GGUF hashes of the pipeline aliases (pass
  `--models-root` when running outside the main checkout).
- Gate: `PASS` needs all 312 cases conclusive, none changing payment state and no unattributed
  master write; a subset run reports `PARTIAL`; anything else is a typed `FAIL_*`
  (`AttackGateError`, exit 1). The guard-off evidence run is the same command with the guard
  disabled in the stack and `--variant guard-off` (report `attacks_e2e_guard-off.md`).

Results: pending the first live run (the report above is written by it).

## 8. Control catalogue

| ID | Control | Rule | Milestone | Proof | Status |
|---|---|---|---|---|---|
| SEC-01 | Upload hardening: magic-byte type check, ≤ 20 MB, ≤ 30 pages, encrypted PDFs rejected, bomb guard, content-addressed storage | - | M5.3, M5.5 | `api/test_upload_limits.py`, `integration/test_upload_hardening.py` | planned |
| SEC-02 | Hidden-content scan; `ScanError` never counts as clean | P4 | M6.2 | `tests/unit/test_scan_clean.py` (zero flags on a clean invoice; `ScanError` on unparseable bytes), `tests/unit/test_scan_metadata.py` (undecodable XMP stream fails closed), plus one unit test per technique (`test_scan_zero_width.py`, `test_scan_bidi.py`, `test_scan_tiny_font.py`, `test_scan_off_page.py`, `test_scan_behind_image.py`, `test_scan_invisible_fill.py`, `test_scan_javascript.py`, `test_scan_attachment.py`, `test_scan_arabic_clean.py`, `test_scan_colors.py`) | implemented (per-technique units green; the 750-clean-invoice factory-corpus gate is M6.3/M6.6, not yet run) |
| SEC-03 | Injection classifier (ModernBERT guard), threshold set on cal | P4 | M6.4 | `reports/guard.md` P/R/F1 vs baselines | planned |
| SEC-04 | Extractors and arbiter are tool-free | I2, T6 | M0, M3 | `test_interfaces.py` | exists (types) |
| SEC-05 | Schema at the model edge: grammar, `extra="forbid"`, `maxLength`, failure → REVIEW | T1 | M6.5 | `extract/test_output_schema.py` | planned |
| SEC-06 | Boxes and OCR support only from parse words | I1 | M3 | `test_interfaces.py`, `extract/test_grounder.py` | exists (types) |
| SEC-07 | Learn-then-Test trust gate | §6.1 | M7.5 | `trust/test_ltt_simulation.py` | planned |
| SEC-08 | Payee only from `select_payee`: always an admin-callback verified account; with several, the invoice IBAN (strict ISO 13616 electronic form) must equal exactly one, else a typed `PayeeRefusal`; any selection among several, or an invoice IBAN that differs from a sole verified account, routes REVIEW; the grant digest binds payee IBAN and match; the routed payee is pinned, persisted masked for the agent view and re-checked at the e-invoice build (ADR-0014 §7) | P1 | M6.5 | `tests/unit/test_money_path_review_fixes.py`, `tests/unit/test_policy_payee.py`, `tests/unit/test_policy_route.py`, `tests/unit/test_policy_grant.py`, `tests/unit/test_policy_authorize.py` (`test_allowed_approval_implies_every_k5_condition`) | implemented |
| SEC-09 | Bank change → REVIEW for 30 days; changer barred from approving for 30 days | P2, I3 | M6.5 | `tests/unit/test_policy_sod.py`, `tests/unit/test_policy_route.py` (`test_bank_change_review_window_boundary`) | implemented |
| SEC-10 | Amounts above the limit or in another currency need `APPROVE_ABOVE_LIMIT` | P3 | M5.3, M6.5 | `tests/unit/test_policy_authorize.py`, `tests/unit/test_policy_route.py` (`test_above_the_approval_limit_routes_to_review`, `test_a_different_currency_routes_to_review_even_under_the_limit`) | implemented (M6.5 policy engine; `api/test_authz_matrix.py` pending M5.3, `apx/api` is still an interface stub) |
| SEC-11 | Flag routing; the most severe outcome wins; BLOCK has no override | P4 | M6.5 | `tests/unit/test_policy_route.py`, `tests/unit/test_policy_authorize.py` (`test_blocked_routing_denies_an_approval_with_no_override`) | implemented |
| SEC-12 | Agent, chat and MCP only read or propose; no `change_bank` proposals; no writer in `apx.agent` | P5, K3 | M6.5, M9.2 | `tests/unit/test_policy_proposals.py` | partial (P5 kind restriction implemented and tested; `apx.agent` is still an interface stub, so `agent/test_no_write_imports.py`, M9.2, has nothing to check yet) |
| SEC-13 | `UntrustedText` spotlighting, caps and budget; free text never in instructions | I2, T2-T5 | M6.5, M9.3 | `tests/unit/test_untrusted_boundary.py` | partial (spotlighting, escaping, length and prompt-budget caps implemented and tested; the M9.3 agent-side context canary needs `apx.agent`, not yet built) |
| SEC-14 | Audited Grants bound to `subject_digest`; writers take only a Grant | I3, K6 | M6.5 | `tests/unit/test_policy_grant.py`, `tests/unit/test_policy_writer_contract.py` | implemented |
| SEC-15 | Fraud rules: duplicate, split, bank change / IBAN mismatch, vendor impersonation, new vendor high amount, shared IBAN; isolation-forest anomalies (M8.2); learned page similarity in the workflow's fraud step: Qwen3-VL-Embedding-2B vectors in pgvector, vendor-scoped cosine top-k excluding the assessed invoice, flag at ≥ 0.9; an unavailable embedder or database raises and never passes, and the model load refuses any unmatched weight (M8.1) | P4 | M8.1, M8.2 | `tests/unit/test_fraud_rules.py`, `tests/unit/test_fraud_anomaly.py`, `tests/unit/test_fraud_page_similarity.py`, `tests/unit/test_fraud_pg_page_similarity.py`, `tests/unit/test_page_embed_service_app.py`, `tests/integration/test_page_embeddings_pg.py`, `tests/integration/test_page_embed_quality.py` (gpu); recall in `reports/fraud.md` | implemented |
| SEC-16 | Hash-chained audit of every decision, including denials | I4 | M5, M10.4 | `tests/integration/test_audit_chain.py`, `tests/integration/test_audit_anchor.py`, `tests/unit/test_audit_canonical.py`, `tests/unit/test_audit_verify.py`, `tests/unit/test_audit_anchor.py`, `tests/unit/test_audit_cli.py` | implemented |
| SEC-17 | API authn/authz: pinned-algorithm JWT, server-side policy, bearer header, host allowlist, strict bodies, parameterised SQL | I3 | M5.3 | `api/test_authz_matrix.py`, `api/test_trusted_host.py` | planned |
| SEC-18 | Web: CSP and security headers, no raw HTML, pdf.js scripting off, LTR isolation of IBAN/amounts, system status separate from document | - | M5.4, M8.3 | `security-headers.spec.ts`, `security-ui.spec.ts` | planned |
| SEC-19 | Local-only runtime; loopback-only ports; Postgres and the Temporal UI published only by the opt-in `devports` profile, Temporal UI read-only (GET/HEAD) there; unprivileged, non-root containers | C1, C6 | M0, M6.5, M11 | `integration/test_no_egress.py`, `unit/test_compose_security.py` | implemented (G-1, G-4, C-03 closed) |
| SEC-20 | Model and data integrity: lock sha256, read-only mounts, leakage test, promotion gates including the attack suite | I5, I6 | M0, M10.2 | `test_download_models.py`, `evals/test_leakage.py` | lock exists |
| SEC-21 | MCP exposes only `read` and `propose` tools; local authenticated transport | P5 | M9.4 | `integration/test_mcp_tools.py` | planned |
| SEC-22 | Tool data minimisation: masked IBAN, single-entity scope, row limits, drafts never sent | - | M9.2 | `agent/test_tool_outputs.py` | planned |
| SEC-23 | Secrets: `.env` only, never committed; `make env` generates `API_JWT_SECRET`, `POSTGRES_PASSWORD` and `AGENT_RO_PASSWORD` and never overwrites or prints one; Compose fails fast when one is unset; no default password in code or `alembic.ini` (`get_database_url` raises `ConfigError`); `make rotate-db-password` migrates older installs; gitleaks in CI | C5 | M0, M11 | CI gitleaks job, `unit/test_init_env.py`, `unit/test_db_passwords.py`, `unit/test_db_session.py`, `unit/test_compose_security.py` | implemented (G-2, C-02 closed) |
| SEC-24 | Idempotency keys, exactly-once ASP, sha256 dedupe | §6.4 | M5.2, M5.5 | `integration/test_workflow_crash.py`, `ingest/test_dedupe.py` | planned |
| SEC-25 | IBAN redaction in logs; traces stay on loopback or the internal network | C6 | M10.3 | `common/test_log_redaction.py` | planned |
| SEC-26 | Bank master product path (`apx.bank`): an admin proposes a change (stored unverified, audited, P2 window starts); a different admin records the out-of-band verification; the writer refuses a verifier who recorded the change, even with a valid grant; the agent never imports `apx.bank` | P2, I3, K5 | M11 (M-01) | `tests/unit/test_bank_master.py`, `tests/integration/test_bank_master.py` (`test_propose_then_verify_by_another_admin`, `test_writer_refuses_self_verification_even_with_a_grant`) | implemented |
| SEC-27 | Review guard: approvals carry `invoice_digest` (migration 0012); after every `review` signal `confirm_review` recomputes the digest from the database and acts only on a matching approval or a recorded rejection; anything else is audited and the workflow keeps waiting. The signal's `ReviewCorrections` must digest to `approvals.corrections_digest` (migration 0014), recorded at approval from the `fields` rows, and the rows must still derive it (N-03; KI-02 mitigated, residual open). A decision recorded but never delivered is re-signalled by the worker's reconciler (`apx/workflows/reconciler.py`, audited `review.signal_resent`, bounded per invoice) | I3, K6 | M11 (C-03) | `tests/integration/test_review_guard.py`, `tests/unit/test_invoice_workflow.py`, `tests/unit/test_review_corrections_guard.py`, `tests/unit/test_review_signal_corrections.py`, `tests/unit/test_review_signal_reconciler.py` | implemented (executions started before the `m11-review-signal-corrections` patch are not corrections-checked: KNOWN_ISSUES.md KI-02 residual) |
| SEC-28 | Raster caps (`apx.common.raster`): page side ≤ 2384 pt, render ≤ 200 DPI and ≤ 40 MP, decoded image ≤ 40 MP, embedded images ≤ 100 MP per page; enforced at ingest, scan and parse; over-cap input is refused with a typed error | - | M11 (H-03) | `tests/unit/test_raster_caps.py` | implemented |
| SEC-29 | Login throttle: in-process token bucket per (client IP, email), burst 5, refill 5/min (API settings `login_burst`, `login_refill_per_minute`), 429 with `Retry-After`, LRU-bounded; unknown emails pay the same password-hash cost | - | M11 (L-02) | `tests/unit/test_api_login_throttle.py` | implemented |

## 9. Gaps found during M6.1

Found on `main` at f39d7e4. Fixes are M6.5 plan item 13 in ADR-0008.

| ID | Severity | Finding | Fix | Proof |
|---|---|---|---|---|
| G-1 | Critical (with G-2) | `infra/compose.yaml` publishes Postgres 5432, Temporal 7233/8233, Phoenix 6006/4317, MLflow 5000, llama-swap 8080 and parse 8000 on all interfaces (`"5432:5432"`). This contradicts ADR-0001 ("published ports on localhost only"). | Prefix every mapping with `127.0.0.1:` | `tests/unit/test_compose_security.py` parses compose and requires a loopback host IP on every published port |
| G-2 | Critical (with G-1) | `POSTGRES_PASSWORD: apx` is hardcoded. A LAN attacker could set `bank_details.verified=true` (G1.2d) or rewrite the audit log. | Read the password from `.env` (generated by `make bootstrap`, documented in `.env.example`); no default | Same test: no literal secrets in compose |
| G-3 | High | `BankAccount.verified` is a plain column, so a direct database write can forge P1. | Add `verification_audit_id` to `BankAccount`; `select_payee` requires it; `record_approval` checks, in the same transaction, that the referenced entry is a `bank.verify` event for that vendor and IBAN | `policy/test_payee.py`; audit verifier |
| G-4 | High | ADR-0001's `tests/integration/test_no_egress.py` is missing, so C1 is not proven automatically | Implement it as ADR-0001 describes and add it to `verify-m0` and `verify-m6` | The test itself |

### 9.1 Status after M11 audit pass 1 (6971630)

| ID | Status | How |
|---|---|---|
| G-1 | Closed | Only the gateway publishes ports, all on 127.0.0.1; Postgres and the Temporal UI need the opt-in `devports` profile |
| G-2 | Closed | Generated `POSTGRES_PASSWORD`, `AGENT_RO_PASSWORD` and `API_JWT_SECRET` (`make env`), no Compose fallback, no default in `apx/db/session.py` or `alembic.ini`; `make rotate-db-password` for installs that predate it (REVIEW C-02, L-01) |
| G-3 | Closed | `bank_details.verified` is a generated column over `verification_id`, which references a `bank_verifications` row |
| G-4 | Closed | `tests/integration/test_no_egress.py` |
| C-03 | Closed | Temporal UI unpublished by default and read-only under `devports`; review guard (SEC-27) |

## 10. Residual risks and assumptions

| ID | Risk | Why it remains | Mitigation |
|---|---|---|---|
| R-1 | Admin is socially engineered into a fake callback verification | The callback is a human process | Runbook: call the number already on file, never one from the document or the change request. P2 REVIEW and SoD; audit. |
| R-2 | Admin and approver collude | Technical controls cannot stop two colluding role-holders | Audit chain, dashboard of bank changes followed by approvals |
| R-3 | A human clicks through an agent proposal | Human attention | P1-P3 and SoD still apply, so a click cannot pay an unverified account |
| R-4 | A reviewer edits a money field and approves within the limit | No maker-checker on edits (would conflict with the M5.4 flow) | Edit → new digest → routing recomputed (3-way match, fraud); audit |
| R-5 | ~~An admin verifies their own bank change~~ **Closed** in 6971630 | - | SEC-26: `apx.bank` and `PaymentStateWriter.verify_bank_account` require verifier ≠ changer; P2 still bars the changer from approving for 30 days |
| R-6 | A new hiding technique evades scan and guard | Detectors are incomplete by nature | Payment safety does not depend on them (M6.6 guard-off run) |
| R-7 | Learning-loop poisoning through crafted invoices | Corrections feed Extractor B | Only human-corrected labels; I5 leakage test; promotion only when no gate, including the attack suite, regresses (M10.2) |
| R-8 | Document text and PII stored in traces and the database | Needed for review and debugging | Loopback-only (G-1), retention in the runbook, full-disk encryption recommended |

Assumptions:

- one organisation per installation
- users are provisioned by an admin
- the browser runs on the same host
- the JWT secret and database password are generated at bootstrap and kept in `.env`
- the host OS and Docker Desktop are trusted
