# End-to-end pipeline run (M5.6)

20 documents dropped into the inbox of the running stack; review-routed invoices approved through the API by a seeded approver.

- Factory invoices archived (UBL built, schematron passed, sent, archived): **15/18**
- Attack PDFs kept out of payment: **2/2**
- Seconds from drop to routing decision: median 1169, max 1651

| Document | Kind | Route | Final status | Drop to route (s) | Fields vs truth | Notes |
|---|---|---|---|---|---|---|
| `5faddaad4fbf` | EN digital | review | archived | 1169 | ok |  |
| `60a940fdb6d0` | EN digital | review | archived | 1165 | ok |  |
| `60ed0c3e2644` | EN digital | review | archived | 1165 | ok |  |
| `6332cab97c6e` | EN digital | review | archived | 1169 | ok |  |
| `6c09772c6f17` | EN digital | review | archived | 1162 | ok |  |
| `6d0a24dc1e99` | EN digital | review | archived | 1169 | ok |  |
| `6df4966b0856` | EN digital | review | archived | 1169 | ok |  |
| `6fc9ec9bac8a` | EN digital | review | archived | 1614 | ok |  |
| `754577b7a71b` | EN digital | review | archived | 1169 | ok |  |
| `7a2a733afb4a` | AR-first digital | review | archived | 1169 | ok |  |
| `8437459bab79` | AR-first digital | review | awaiting_review | 1611 | ok | approve refused: 422 {"detail":"cannot approve: correct these fields first: invoice_type_code, line_items[1].vat_category, line_items[2].vat_ |
| `910a128e8c12` | AR-first digital | review | archived | 1614 | ok |  |
| `91a18faeb811` | AR-first digital | review | archived | 1169 | ok |  |
| `135a92c2fb5d` | scanned | review | archived | 1614 | ok |  |
| `80783d9249a8` | scanned | review | awaiting_review | 1468 | mismatch: supplier_trn | corrected: supplier_address; approve refused: 403 {"detail":"action approve_payment denied: no payee could be selected from the admin-verified accounts: unknown_vendor"} |
| `165d017b0a90` | scanned | review | awaiting_review | 1614 | mismatch: lines (23 of 24) | approve refused: 422 {"detail":"cannot approve: validation found amounts that do not add up (lines[11].net_amount, lines[12].net_amount, subt |
| `7d4ea1cc560d` | scanned | review | archived | 1611 | ok |  |
| `0950ca9cded7` | zero-rated | review | archived | 1165 | ok |  |
| `0980a593c78f` | attack | block | blocked | 725 | n/a |  |
| `0a4fb7e1dc2a` | attack | block | blocked | 1651 | n/a |  |
