# End-to-end pipeline run (M5.6)

20 documents dropped into the inbox of the running stack; review-routed invoices approved through the API by a seeded approver.

- Factory invoices archived (UBL built, schematron passed, sent, archived): **15/18**
- Attack PDFs kept out of payment: **0/2**
- Seconds from drop to routing decision: median 5003, max 5433

| Document | Kind | Route | Final status | Drop to route (s) | Fields vs truth | Notes |
|---|---|---|---|---|---|---|
| `0082e402ae77` | EN digital | review | archived | 4996 | ok |  |
| `043273b51399` | EN digital | review | archived | 4996 | ok |  |
| `054296095715` | EN digital | review | archived | 4993 | ok |  |
| `0d5bb4492cb8` | EN digital | review | archived | 5003 | ok |  |
| `0dc1c271348c` | EN digital | review | archived | 5000 | ok |  |
| `10d75d791852` | EN digital | review | archived | 5003 | ok |  |
| `11eb6f7b9bed` | EN digital | review | archived | 5000 | ok |  |
| `15910e215775` | EN digital | review | archived | 5429 | ok |  |
| `18c083f314e9` | EN digital | review | archived | 5432 | ok |  |
| `0298f9afdc61` | AR-first digital | review | archived | 4996 | ok |  |
| `0764f24b3e16` | AR-first digital | review | archived | 4993 | ok |  |
| `1b5ce207483b` | AR-first digital | review | awaiting_review | 5429 | ok | approve refused: 403 {"detail":"action approve_payment denied: no payee could be selected from the admin-verified accounts: invoice_iban_miss |
| `1cf8f48b3769` | AR-first digital | review | archived | 5429 | ok |  |
| `13e9e6552e94` | scanned | review | approved | 5433 | ok |  |
| `b1f2a9e3f8f2` | scanned | review | archived | 5426 | ok |  |
| `fbf2cc1d12b3` | scanned | review | approved | 5429 | ok |  |
| `1c17cf651a9c` | scanned | review | archived | 5429 | ok |  |
| `01c6cd2203a2` | zero-rated | review | archived | 5004 | ok |  |
| `0097d3bdc9f5` | attack | — | pending | — | n/a |  |
| `0370ed80a145` | attack | — | pending | — | n/a |  |
