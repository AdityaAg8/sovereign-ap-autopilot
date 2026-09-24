# ADR-0014: P1 with several verified accounts: the invoice IBAN selects, never introduces (amends ADR-0008 K5)

- Status: accepted
- Date: 2026-09-24
- Deciders: architect (T3, money path)
- PRD: §2, §6.2 I3, §6.3 P1, §8 (money path)
- Amends: ADR-0008 K5 ("exactly one account with `verified=True`") and its consequence "a vendor
  with two verified accounts routes to REVIEW until an admin retires one". Related: ADR-0003,
  `docs/security.md` I3, SEC-08, SEC-14, A08.
- Interfaces: `apx/common/errors.py`, `apx/policy/interfaces.py`, `apx/policy/engine.py`,
  `apx/policy/guard.py`, `apx/policy/writer.py`, `apx/workflows/activities.py`

## Context and problem statement

P1 (`PolicyRule.P1_VERIFIED_IBAN`) pays only an account with `verified=True` set by an admin
callback, never by extraction. Today two functions decide who the payee is, and they disagree:

| | `select_payee(vendor)` (policy: `route`, `authorize`, `subject_digest`, so guard and writer) | `verified_payment_iban(vendor, claimed)` (workflow: UBL `PaymentMeans`) |
|---|---|---|
| Counts as verified | `verified and source == "admin_callback"` | `verified` only |
| Reads the invoice IBAN | never | yes, raw string `in` |
| 0 verified | `PolicyDeniedError` | `None` |
| 1 verified, invoice IBAN missing or different | pays that account | pays that account |
| 2+ verified, invoice IBAN equals one (raw) | `PolicyDeniedError` | pays the claimed one |
| 2+ verified, no match or missing | `PolicyDeniedError` | `None` |

So a vendor with two admin-verified accounts (a common real case: AED and USD accounts, or two
banks) can never be approved, while the e-invoice path already picks an account using a raw
compare that `"ae07 0331..."` defeats. The e-invoice path also accepts `verified=True` with
`source="unverified"`, which the policy path rejects. Two sources of truth for the payee is a
risk in itself: the approved payee and the payee written into the UBL can differ.

The refusal is also untyped. Engine, guard and UI get one string ("no unique admin-verified
account") and cannot tell "no verified account" apart from "invoice IBAN missing".

## Decision drivers

1. I3 / ADR-0008: an unverified IBAN is never paid, under any input.
2. `Invoice.iban` is untrusted extracted text. It may select among the vendor's verified accounts
   but never adds a payee.
3. The single-verified-account case must not get looser than today.
4. One function decides the payee. Engine, guard, subject digest, writer and UBL path all call it.
5. Refusals are typed (enum) and fail closed. No silent fallback (PRD §8).
6. Use the simplest control that stops a forged invoice from steering money unseen.

## Considered options

- **(a) Status quo: exactly one verified account.** Simple and never reads the invoice. But
  vendors with several accounts can never be paid until an admin retires an account, which pushes
  admins into bank-master churn. That churn is itself the P2 attack surface. The UBL path stays a
  second, looser source of truth.
- **(b) With several verified accounts, pay the one whose normalized IBAN exactly equals the
  invoice IBAN. Zero or several matches refuse with a typed reason.** The invoice can only pick
  from a set the admin already verified. It needs a strict normalization and a steering control.
- (c) Pay a "primary" or most-recently-used account and ignore the invoice. Rejected: the vendor
  master has no primary flag or payment history, so this needs a schema change and a new
  admin-controlled field, and it would still disagree with the invoice.

## Decision outcome

Chosen: **(b)**, plus a REVIEW control on every non-trivial selection.

### 1. Single source of truth

`select_payee` stays in `apx/policy/interfaces.py`, keeps its name and takes a new **required
keyword** argument. mypy then flags every caller that has not been updated.
`verified_payment_iban` becomes a thin wrapper over it. No other code may filter
`bank_accounts` for payment.

```python
# apx/common/errors.py
class PayeeRefusal(StrEnum):
    """Why P1 could not select a payee (ADR-0014). Carried by PayeeRefusedError and PolicyReason."""

    UNKNOWN_VENDOR = "unknown_vendor"  # vendor is None
    MASTER_IBAN_MALFORMED = "master_iban_malformed"  # a verified account fails electronic_iban
    NO_VERIFIED_ACCOUNT = "no_verified_account"  # 0 accounts with verified and admin_callback
    INVOICE_IBAN_MISSING = "invoice_iban_missing"  # 2+ verified, invoice IBAN None or blank
    INVOICE_IBAN_MALFORMED = (
        "invoice_iban_malformed"  # 2+ verified, invoice IBAN fails electronic_iban
    )
    NO_MATCH = "no_match"  # 2+ verified, none equals the invoice IBAN
    AMBIGUOUS_MATCH = "ambiguous_match"  # 2+ verified records equal the invoice IBAN


class PayeeRefusedError(PolicyDeniedError):
    """P1 refused to select a payee; `refusal` is the typed reason (ADR-0014)."""

    def __init__(self, refusal: PayeeRefusal, message: str) -> None: ...

    refusal: PayeeRefusal


# apx/policy/interfaces.py
class PayeeMatch(StrEnum):
    MATCHED = "matched"  # invoice IBAN equals the selected account
    SOLE_UNCLAIMED = "sole_unclaimed"  # exactly 1 verified, invoice IBAN None or blank
    SOLE_MISMATCH = "sole_mismatch"  # exactly 1 verified, invoice IBAN differs or is malformed


class Payee(Frozen):
    account: BankAccount  # always one of vendor.bank_accounts, verified and admin_callback
    iban: str  # electronic_iban(account.iban): the value paid, digested and put in UBL
    match: PayeeMatch
    verified_count: int  # number of verified admin_callback account records (>= 1)


def electronic_iban(raw: str) -> str | None:
    """ISO 13616 electronic format, or None if `raw` is not a well-formed IBAN (see §2)."""


def select_payee(vendor: Vendor | None, *, invoice_iban: str | None) -> Payee:
    """P1 (ADR-0014): the payee among the vendor's admin-verified accounts. Raises
    PayeeRefusedError with a PayeeRefusal. The invoice IBAN only selects, never introduces."""


# apx/workflows/activities.py (kept, now a wrapper)
def verified_payment_iban(vendor: Vendor | None, claimed: str | None) -> str | None:
    """select_payee(vendor, invoice_iban=claimed).iban, or None on PayeeRefusedError so the UBL
    builder keeps refusing an Invoice without one (EInvoiceError, unchanged)."""
```

`PolicyReason` gains two optional fields, `payee_refusal: PayeeRefusal | None = None` and
`payee_match: PayeeMatch | None = None`. `rule` stays `PolicyRule.P1_VERIFIED_IBAN`. `message`
stays a system template that may include the enum value but never the document text (ADR-0008).
The guard's `_reasons_payload` writes both fields into the audit payload when they are set.

### 2. Normalization (`electronic_iban`)

1. The input must be ASCII only (`raw.isascii()`). The ASCII check runs before case-folding, so
   `"ß".upper() == "SS"` and similar can never apply.
2. Remove U+0020 SPACE only. That is the only separator ISO 13616 print format uses.
3. Upper-case (ASCII letters only, which step 1 guarantees).
4. The result must match `^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$` (length 15-34).

Anything else returns `None`. Rejected, not stripped: tab, newline, NBSP U+00A0, other Unicode
spaces, hyphens, dots, slashes, zero-width characters, Arabic-Indic and full-width digits, and
Cyrillic or Greek lookalike letters. Stripping or transliterating would let untrusted text turn
into a verified IBAN through characters a reviewer cannot see. The looser
`apx.fraud.normalize.normalize_iban` stays for fraud matching. It is never used for payment.

**Mod-97 is not checked here.** Selection is exact equality with an admin-verified record, so a
checksum adds no safety. A mistyped invoice IBAN simply fails to match (NO_MATCH, or
SOLE_MISMATCH and so REVIEW). Checking it would also refuse existing master data: several test
IBANs, for example `AE070331234567890000001`, fail mod-97. Mod-97 belongs at vendor-master write
time (`change_bank_details`). That is follow-up work, outside T3.

Master accounts go through the same function. A verified account whose stored IBAN returns `None`
refuses the whole vendor with `MASTER_IBAN_MALFORMED`. The account is not silently dropped: that
could turn a two-account vendor into a single-account one.

### 3. Selection algorithm (in order)

1. `vendor is None` → `UNKNOWN_VENDOR`.
2. `verified = [a for a in vendor.bank_accounts if a.verified and a.source == "admin_callback"]`.
   Records are counted, not distinct IBANs. Any `electronic_iban(a.iban) is None` →
   `MASTER_IBAN_MALFORMED`.
3. `len(verified) == 0` → `NO_VERIFIED_ACCOUNT`. An unverified account that matches the invoice
   changes nothing.
4. `len(verified) == 1` → pay that account (see §4). Set `match` as follows:
   - `MATCHED` if `electronic_iban(invoice_iban)` equals it.
   - `SOLE_UNCLAIMED` if the invoice IBAN is None or blank (`strip() == ""`).
   - Otherwise `SOLE_MISMATCH`. This includes a malformed invoice IBAN and one that equals an
     unverified account.
5. `len(verified) >= 2`:
   - None or blank → `INVOICE_IBAN_MISSING`.
   - `electronic_iban` is None → `INVOICE_IBAN_MALFORMED`.
   - Otherwise count the verified records equal to it: 0 → `NO_MATCH`, 2 or more →
     `AMBIGUOUS_MATCH` (duplicate verified rows are a data-integrity fault, so refuse),
     1 → `MATCHED`.

### 4. Single-verified-account case: still pays that account (not looser, and route gets stricter)

Today both functions pay the one verified account whatever the invoice says. `select_payee`
never reads the invoice. `verified_payment_iban` returns `verified[0]` when the claim does not
match. **Decision: resolution is unchanged.** A missing or mismatching invoice IBAN still resolves
to the sole verified account. There is nothing to select between, and the money can only go to
the admin-verified account. Refusing would block every legitimate invoice that prints a stale
IBAN, and the only way to clear it would be a bank-master change, which is exactly the P2 path an
attacker wants to force.

**Stricter than today:** `route` adds a REVIEW reason (P1, `payee_match=SOLE_MISMATCH`) when the
invoice IBAN is present and does not match. An invoice that says "pay a different account" is the
classic BEC signal (A08) and must never go AUTO. `SOLE_UNCLAIMED` adds no reason, as today.
`authorize` does not deny `SOLE_MISMATCH`: a human approving it is the review.

### 5. Steering threat and its control

With several verified accounts, a forged or altered invoice can move payment from one verified
account of the vendor to another. **Why this is acceptable:** every candidate was verified by an
admin callback as the vendor's own account. The attacker cannot introduce an account (I3 holds).
The worst case is a misdirected payment between the vendor's own accounts, which is recoverable,
not a loss to a third party. **Extra control:** `route` adds a REVIEW reason (P1,
`payee_match=MATCHED`, message "vendor has several admin-verified accounts; the invoice IBAN chose
the payee") whenever `verified_count >= 2`. A vendor with several accounts is therefore never
paid AUTO. A human always sees the masked selected payee before approving. This needs no new
schema: there is no "primary" or "last used" field to compare against. A refusal (§3) routes
REVIEW, as today, and `authorize` denies approval with the typed reason.

### 6. Grant binding (TOCTOU)

For approval actions, `subject_digest` binds
`subject["payee_iban"] = payee.iban` and `subject["payee_match"] = payee.match.value`, where
`payee = select_payee(vendor, invoice_iban=invoice.iban if invoice else None)`. Both the guard
(grant time) and the writer (write time) build the request from the same projection
(`invoice_for_digest` includes `row.iban`; `load_vendor` loads the current accounts), so both
sides call the one function. Between approve and execute, any of the following changes the digest
and `record_approval` raises `STALE_GRANT`:
- adding, removing or verifying an account in a way that changes the selection,
- editing the invoice IBAN so that it matches a different account,
- a match flipping to a mismatch.

If the selection now refuses, `subject_digest` raises `PayeeRefusedError` (a
`PolicyDeniedError`) and the writer lets it propagate. The typed reason reaches the caller and
nothing is written. Non-approval actions still never call `select_payee` and carry neither key.

### 7. Payee pin: routing to build (amendment, 2026-09-24, CR-money on PR #48)

The grant digest (§6) covers approve → write, but `build_einvoice` used to re-derive the payee
from the live master, so a master change between approval and build changed who is paid.
Now:
- `DefaultPolicyEngine.route` sets `RoutingDecision.payee = pin_payee(select_payee(...))`, a
  `RoutedPayee` with the masked IBAN, match, verified count and `iban_sha256` (never the raw
  IBAN), or None when P1 refused (the P1 reason carries `payee_refusal`).
- The workflow (patch `adr14-payee-pin`) passes `routing.payee` to `build_einvoice`. The
  activity re-runs `select_payee` on the live master and builds only if the live result has the
  same digest and match (or both refuse). Anything else is audited as
  `activity:payee_changed` and raised as a non-retryable `ApplicationError` of type
  `PayeeChangedError`. Nothing is built, so neither account is paid. The pin is the routed
  payee, which is also what the reviewer and agent see, so any master change after routing
  (even one an approver later approved) refuses the build. The invoice must then be
  re-processed. This is deliberately conservative.
- `SqlInvoiceRecords.routed` writes a `policy_route` decision whose reasons end with one
  `{"payee": {...}}` element (`routing_record`: masked IBAN, match and count, or the refusal).
  Migration `0008_agent_routed_payee` redefines `v_agent_invoices` to read the payee from that
  row: `payee_masked`, `document_iban_matches_payee` (`match == "matched"`), and new
  `payee_match` and `payee_refusal` columns. The payee element is filtered out of
  `reasons_json`, and the `apx_agent_ro` grant is re-issued. Persisting at route time needed no
  table change: the `decisions` row already existed for this purpose, and a new column would
  also need an `apx/db/models.py` change.
- The agent prompt drops `payee.iban_sha256` from the routing it quotes.
- **Deploy with workflows in flight (drain runbook).** A workflow routed before this change has no
  pin (`routing.payee is None`). If it reaches `build_einvoice` afterwards, the build fails closed
  with `PayeeChangedError`: no wrong payment, but the invoice never completes. A build already
  scheduled against the old one-argument activity fails with a `TypeError` until its retries run
  out. Before deploying: let in-flight invoices finish, or reject/re-ingest the ones still in
  AWAITING_REVIEW (`temporal workflow list --query 'ExecutionStatus="Running"'`). An invoice that
  hits the refusal after the deploy is re-ingested so it is re-routed and pinned. Pre-v0.1.0 no
  such workflows exist outside dev stacks.

## Consequences

- Good: vendors with several verified accounts become payable, but only through REVIEW. One
  function decides the payee for policy, grant and UBL. The e-invoice path now also requires
  `source == "admin_callback"`, which is stricter than today. Refusals are typed and audited.
- Good: single-account resolution is unchanged, and a mismatching invoice IBAN now forces REVIEW
  where before it could go AUTO.
- Bad: `Invoice.iban` now influences P1. The docs/security.md claim "P1 ignores the invoice IBAN"
  (SEC-08, T row, B4 tree) is replaced by "the invoice IBAN only selects among admin-verified
  accounts". SEC-08 and A08 in docs/security.md must be updated in the same change.
- Bad: more REVIEW volume for multi-account vendors and stale-IBAN invoices. This is accepted:
  conservative on the money path.
- Resolved by §7: `v_agent_invoices` used to show the most recently changed verified account as
  `payee_masked` and compare IBANs raw, so for multi-account vendors it could name a different
  account from the one P1 selects. It now shows the routed selection. Invoices routed before
  migration 0008 have no persisted payee and show None. `v_agent_vendor_accounts.payee_masked`
  (vendor level) is still the newest verified account and is documented as display only. Mod-97
  at bank-master write time is still follow-up work.
- Bad, and accepted (CR-money P2-3): a reviewer's IBAN edit never reaches `select_payee`.
  `edit_field` only appends a `fields` row, while the policy, grant and build read the
  extracted `invoices.iban`. So a multi-account vendor whose IBAN was misread by OCR
  (`INVOICE_IBAN_MISSING`, `INVOICE_IBAN_MALFORMED`, `NO_MATCH`) can only be rejected, not
  corrected in review. This is deliberate: a human edit selecting the payee would make one
  reviewer able to steer payment among accounts. The P1 refusal message says so
  (`apx.policy.engine.IBAN_EDIT_HINT`), and it reaches the reviewer in the approval's 403 and in
  the route reasons. The remedy is a corrected invoice from the supplier.
- Neutral: G-3 (`verification_audit_id`) is unaffected. When it lands it goes into step 2 of §3.

## Proof (tests)

New tests, in `tests/unit/test_policy_payee.py` unless noted:
- **Hypothesis property:** generate 0..N verified admin_callback accounts (IBANs drawn from a small
  pool so duplicates occur), 0..M unverified accounts (including `verified=True,
  source="unverified"` forgeries), and any invoice IBAN. The invoice IBAN is drawn from: None,
  blank, a verified or unverified IBAN with random case and spaces, the same with a NBSP, hyphen,
  Arabic-Indic digit or Cyrillic lookalike, and arbitrary `st.text()`. Assert that either
  `PayeeRefusedError` is raised, or `payee.account` is in `vendor.bank_accounts` with `verified`
  and `source == "admin_callback"` and `payee.iban == electronic_iban(payee.account.iban)`. With
  `verified_count >= 2`, also assert `electronic_iban(invoice_iban) == payee.iban` and that
  exactly one verified record has that IBAN. With exactly one, assert the payee is that account
  for every invoice IBAN (the old property still holds). The paid IBAN is never an unverified
  account's IBAN unless a verified record has the same IBAN.
- Unit tests:
  - one match: two verified accounts, invoice equals the second → second, `MATCHED`.
  - no match → `NO_MATCH`.
  - unverified-only match: vendor with only an unverified account equal to the invoice →
    `NO_VERIFIED_ACCOUNT`. Verified A plus unverified B, invoice B, several verified → `NO_MATCH`.
    Single verified A plus unverified B, invoice B → A with `SOLE_MISMATCH`.
  - case and space normalization: `"ae07 0331 2345 6789 0123 456"` matches `MATCHED`.
  - rejected separators: NBSP, tab, hyphen, Arabic-Indic digit, Cyrillic `А` →
    `INVOICE_IBAN_MALFORMED` (several verified) or `SOLE_MISMATCH` (one).
  - ambiguous duplicate: two verified records with the same IBAN, invoice equals it →
    `AMBIGUOUS_MATCH`.
  - missing invoice IBAN: None and `"  "` with two verified → `INVOICE_IBAN_MISSING`. With one →
    `SOLE_UNCLAIMED`.
  - malformed master IBAN → `MASTER_IBAN_MALFORMED`.
  - `electronic_iban` table test.
- `tests/unit/test_policy_route.py`:
  - two verified plus a match → REVIEW with a P1 reason and `payee_match=MATCHED`.
  - one verified plus a mismatching invoice IBAN → REVIEW (`SOLE_MISMATCH`).
  - one verified with no invoice IBAN → no P1 reason.
  - a refusal → REVIEW with `payee_refusal` set.
- `tests/unit/test_policy_authorize.py`:
  - a refusal denies approval with a P1 reason carrying `payee_refusal`.
  - two verified plus a match is allowed when the other K5 conditions hold.
- `tests/unit/test_policy_grant.py`: the digest changes when a second verified account is added
  and the invoice IBAN now selects it, and when the invoice IBAN changes from matching to
  mismatching. Covered at the writer level by `tests/unit/test_policy_writer_contract.py` (stale
  grant).

Existing tests that encode "exactly one verified account" or the old signature:
- `test_policy_payee.py:15`: `select_payee(vendor)` becomes
  `select_payee(vendor, invoice_iban=None)`. Assert `.account == account` and `SOLE_UNCLAIMED`.
- `test_policy_payee.py:21, :36, :47`: still refused. Now assert `PayeeRefusedError` with
  `NO_VERIFIED_ACCOUNT`.
- `test_policy_payee.py:28`: **behaviour change.** "Two verified accounts are denied" becomes
  `INVOICE_IBAN_MISSING` when the invoice IBAN is None, plus a match case that pays the matched
  account. Rename the test accordingly.
- `test_policy_payee.py:42`: now asserts `UNKNOWN_VENDOR`.
- `test_policy_payee.py:55`: the property still holds for a single verified account, now passing
  `invoice_iban=document_iban` explicitly. Its docstring ("has no `invoice` parameter") and the
  module docstring (line 1, "unique admin-verified account") are rewritten.
- `test_policy_authorize.py:84`: becomes `select_payee(vendor, invoice_iban=invoice.iban)`,
  which must not raise.
- `test_policy_grant.py:151`: the monkeypatched stub gets the new signature
  `(vendor, *, invoice_iban)`. `expected_subject` also asserts `"payee_match"` is absent.
- `test_workflow_components.py:267-274`: all five assertions keep their results (the old wrapper
  behaviour for these inputs is preserved). Add:
  - two verified with a lower-case, spaced claim → the claimed account (raw compare gave None).
  - two verified with a third IBAN → None.
  - `verified=True, source="unverified"` as the sole account → None (this was the verified IBAN).

Gate: `uv run pytest tests/unit`, `uv run mypy`, `uv run ruff check .`,
`uv run python scripts/check_headers.py --all`. Run locally: `make e2e`, plus
`uv run pytest -m integration tests/integration/test_api_authz.py tests/integration/test_workflow_crash.py`.
These confirm that live approvals and the UBL still resolve the same payee for single-account
vendors.
