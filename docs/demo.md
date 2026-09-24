# Demo walkthrough

`make demo` runs a small, realistic accounts-payable day on your machine: it brings up the stack,
loads the factory ERP (vendors, verified bank accounts, POs, users), starts the web app against the
live API, and drops nine supplier invoices into the inbox a few seconds apart. Nothing leaves the
machine: the models, OCR, database and workflow engine run in local containers.

## Before you start

- A fresh clone with `make bootstrap` and `make models` done once (models are downloaded on the
  host, never by the running stack).
- An NVIDIA GPU with 12 GB (the extraction models share it one stage at a time).
- The factory test split in `data/uae/test` (`make factory`) and the attack set in `data/attacks`.

## Run it

```bash
make demo
```

The terminal shows one row per document as it moves through the pipeline:

```
Scenario     Document       Status            Route    Flags
------------------------------------------------------------------------
clean        00ddefcafb24   awaiting_review   review
bilingual    0365433bcbd2   awaiting_review   review
poisoned     0097d3bdc9f5   blocked           block
bank change  1e99a1450f19   awaiting_review   review   bank_detail_change
duplicate    de0827e5d1ff   awaiting_review   review   duplicate_invoice
```

Colour is used only for the route (green for paid, amber for held, red for blocked) and is off
when `NO_COLOR` is set or the output is not a terminal.

Open **http://localhost:3000/operations** and sign in with the reviewer/approver account the
terminal names. The password is in `data/seed_credentials.json`, which `make seed` generates and
git ignores.

## What each scenario shows

| Scenario | What happens | Where to look |
|---|---|---|
| Clean (3) | Extracted by two models, grounded to the words on the page, validated and checked for fraud. With no calibrated trust gate yet (M7.4), every invoice is held for a person rather than paid automatically. | Review queue: each field shows its value, its box on the page and a confidence. |
| Bilingual (2) | Arabic-first invoices. Party names printed only in Arabic keep their Arabic script. | Invoice page: the supplier field in Arabic, boxed on the page image. |
| Poisoned (2) | PDFs carrying hidden instructions to the model. The scanner flags text a reader cannot see, and above the block threshold the invoice is blocked and can never be paid. The extraction models read every document tool-free, so hidden instructions have nothing to act on. | Operations: the blocked tile. Invoice page: the scan flags and evidence. |
| Bank change | The vendor switched its IBAN days before issuing this invoice. The fraud rule flags it, and payment always goes to an admin-verified account in the vendor master; the IBAN printed on the invoice can only pick among those, never add one (ADR-0014). | Invoice page: the `bank_detail_change` flag and its plain-language explanation. |
| Duplicate | The first clean invoice resubmitted as a new file. Same number, vendor and amount, so it is flagged as a duplicate. | Invoice page: the `duplicate_invoice` flag and its explanation. |

## Things to try

- **Approve a held invoice** on the review page. The workflow resumes: it builds the PINT AE
  e-invoice, validates it against the official schematron, sends it to the mock ASP and archives
  it. Watch the invoice move to *Archived* on the operations page.
- **Ask the assistant** (press `c`): "Why was the bank-change invoice held?" The answer cites the
  audit events it relied on; each citation opens that event in the invoice's audit trail. Ask it to
  pay an invoice and it declines: it can read and draft, never approve or pay.
- **Check the audit chain**: `cd infra && docker compose exec worker python -m apx.audit verify`
  recomputes every hash in the log and its anchors, and reports the first broken link if any.

## Reset

`make seed` restores the ERP and clears all invoices. `make down` stops the stack. Model weights and
the database volume stay on disk.
