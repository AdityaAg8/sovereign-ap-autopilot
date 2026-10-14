# End-to-end pipeline run

20 documents dropped into the inbox of the running stack; review-routed invoices approved through the API by a seeded approver.

- Factory invoices archived (UBL built, schematron passed, sent, archived): **18/18**
- Attack PDFs kept out of payment: **2/2** (1 of them only because the workflow failed before routing)
- Seconds from drop to routing decision: median 1459, max 2232

| Document | Kind | Route | Final status | Drop to route (s) | Fields vs truth | Notes |
|---|---|---|---|---|---|---|
| `77e699564ccd` | EN digital | review | archived | 1459 | ok |  |
| `7a118e853028` | EN digital | review | archived | 1936 | ok |  |
| `7a3e2a8786c3` | EN digital | review | archived | 1428 | ok |  |
| `7d843cf02049` | EN digital | review | archived | 1429 | ok |  |
| `7d96873af072` | EN digital | review | archived | 1936 | ok |  |
| `846599920cb1` | EN digital | review | archived | 1459 | ok |  |
| `867d05616f8e` | EN digital | review | archived | 1435 | ok |  |
| `8893bd2c5f37` | EN digital | review | archived | 1936 | ok |  |
| `8f493dc761f4` | EN digital | review | archived | 1460 | ok |  |
| `92a565fe5d4d` | AR-first digital | review | archived | 1933 | ok |  |
| `a1f931809628` | AR-first digital | review | archived | 1435 | ok |  |
| `a3a051e968d1` | AR-first digital | review | archived | 1432 | ok |  |
| `c36e1da8de6c` | AR-first digital | review | archived | 1435 | ok |  |
| `683d90a54929` | scanned | review | archived | 2232 | ok |  |
| `d3fb3c81eb2a` | scanned | review | archived | 2229 | ok |  |
| `3cbaf0497746` | scanned | review | archived | 1936 | ok |  |
| `6cbf1fe055ac` | scanned | review | archived | 2232 | ok |  |
| `0d54fd15e6e4` | zero-rated | review | archived | 1429 | ok |  |
| `0d0e8d0a8777` | attack | review | awaiting_review | 650 | n/a |  |
| `0d2f782a95cc` | attack | — | failed | — | n/a | failed: ActivityError: Activity task failed <- ModelOutputError: ModelOutputError: required invoice fields missing: subtotal, vat_total, total |
