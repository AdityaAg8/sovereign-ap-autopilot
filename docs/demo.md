# Demo walkthrough

`make demo` runs a small, realistic accounts-payable day on your machine: it brings up the stack,
loads the factory ERP (vendors, verified bank accounts, POs, users), starts the web app against the
live API, and drops nine supplier invoices into the inbox a few seconds apart, checking each one's
outcome. Nothing leaves the machine: the models, OCR, database and workflow engine run in local
containers.

## Before you start

- A fresh clone with `make bootstrap` and `make models` done once (models are downloaded on the
  host, never by the running stack).
- An NVIDIA GPU with 12 GB (the extraction models share it one stage at a time).
- The factory test split in `data/uae/test` (`make factory`) and the attack set in `data/attacks`.

## Run it

```bash
make demo
```

The terminal shows one row per document as it moves through the pipeline, and checks each one
against the outcome its scenario promises (one row per scenario shown; times are illustrative):

```
Scenario     Document      Status           Route   Result  Time   Detail
-----------------------------------------------------------------------------------------------------------------
clean        0082e402ae77  archived         review  pass     212s  archived after approval
bilingual    0298f9afdc61  archived         review  pass     340s  archived after approval
poisoned     07f7ed21c0aa  blocked          block   pass      41s  blocked by the scan
bank change  b95321b393fe  awaiting_review  review  pass     118s  held by P1
duplicate    011e5a82b846  awaiting_review  review  pass     126s  flagged: duplicate_invoice
```

Colour marks only the Result cell (green pass, red FAIL) and is off when `NO_COLOR` is set, when
`TERM=dumb`, or when the output is not a terminal (then a row is printed only when it changes).

The run always ends. Each scenario has its own deadline (30 minutes from its drop; change it with
`make demo DEMO_ARGS="--timeout 900"`). A scenario that misses its outcome prints one typed line
and the command exits 1:

| Line starts with | Meaning |
|---|---|
| `timeout:` | No verdict before the deadline; the line names the last status the API reported. |
| `workflow_failed:` | The workflow failed; the line carries the error `GET /invoices/{id}` recorded. |
| `wrong_outcome:` | The invoice settled somewhere the scenario forbids, e.g. an attack on the payment path. |
| `approval_refused:` | The API refused the demo's approval of a clean invoice (status and reason shown). |
| `already_ingested:` | The document was in the database before the drop; run `make seed` first. |

`setup_error:` or `api_error:` (exit 2) means the local data or the API is missing before anything
is dropped.

Open **http://localhost:3000/operations** and sign in with the reviewer/approver account the
terminal names. The password is in `data/seed_credentials.json`, which `make seed` generates and
git ignores.

## What each scenario shows

| Scenario | Expected outcome (asserted) | What happens | Where to look |
|---|---|---|---|
| Clean (3) | Archived | Extracted by two models, grounded to the words on the page, validated and checked for fraud. Invoices from vendors with one verified account the invoice matches. Without a calibrated trust gate each one is held for review; the demo approves it through the API as the seeded approver, and the workflow builds the PINT AE e-invoice, validates it, sends it to the mock ASP and archives it. | Operations: the archived count. Invoice page: each field's value, box and confidence. |
| Bilingual (2) | Archived | Arabic-first invoices, approved the same way. Party names printed only in Arabic keep their Arabic script. | Invoice page: the supplier field in Arabic, boxed on the page image. |
| Poisoned (2) | Blocked, or held with a scan reason; never on the payment path | Attack PDFs with hidden instructions, picked because the rule-based scanner alone scores them at the block threshold; the runtime injection guard adds its own flags on top. Above the threshold the invoice is blocked and can never be paid. The extraction models read every document tool-free, so hidden instructions have nothing to act on. | Operations: the blocked tile. Invoice page: the scan reason. |
| Bank change | Held for review by P1 | The vendor switched its IBAN and now has several admin-verified accounts; the invoice IBAN may only choose among them, never add one, so P1 payee selection holds it for a person (ADR-0014). The pinned payee is what an approval pays. When it was issued inside 30 days of the switch the `bank_detail_change` flag shows too. | Invoice page: the P1 reason and the payee. |
| Duplicate | Held with a duplicate flag | The first clean invoice resubmitted as a new file: same number, vendor and amount (`duplicate_invoice`), and the same page (`near_duplicate_page` / `near_duplicate_embedding`). | Invoice page: the duplicate flag and its explanation. |

## Things to try

- **Decide the held ones** on the review page. Approving the bank-change invoice pays the
  payee P1 pinned, never an IBAN typed on the invoice; the duplicate and poisoned ones are best
  rejected.
- **Ask the assistant** (press `c`): "Why was the bank-change invoice held?" The answer cites the
  audit events it relied on; each citation opens that event in the invoice's audit trail. Ask it to
  pay an invoice and it declines: it can read and draft, never approve or pay.
- **Check the audit chain**: `cd infra && docker compose exec worker python -m apx.audit verify`
  recomputes every hash in the log and its anchors, and reports the first broken link if any.

## Reset

`make seed` restores the ERP and clears all invoices. `make down` stops the stack. Model weights and
the database volume stay on disk.
