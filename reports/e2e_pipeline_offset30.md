# End-to-end pipeline run (M5.6)

14 documents dropped into the inbox of the running stack; review-routed invoices approved through the API by a seeded approver.

- Factory invoices archived (UBL built, schematron passed, sent, archived): **11/12**
- Attack PDFs kept out of payment: **2/2**
- Seconds from drop to routing decision: median 368, max 580

| Document | Kind | Route | Final status | Drop to route (s) | Fields vs truth | Notes |
|---|---|---|---|---|---|---|
| `6645b8cfb980` | EN digital | review | archived | 395 | ok |  |
| `675e7349a9c4` | EN digital | review | archived | 553 | ok |  |
| `6e1c2887a3e4` | EN digital | review | archived | 241 | ok |  |
| `6f44800546da` | EN digital | review | archived | 398 | ok |  |
| `74151010fffd` | EN digital | review | archived | 260 | ok |  |
| `75e4241ac1ff` | EN digital | review | archived | 205 | ok |  |
| `76b94a0076af` | EN digital | review | archived | 345 | ok |  |
| `7a695d9ac513` | EN digital | review | archived | 580 | ok |  |
| `7fc6bf815630` | EN digital | review | archived | 568 | ok |  |
| `d792ff9cdce0` | scanned | review | archived | 199 | ok |  |
| `f27eeb9ce1f9` | scanned | review | awaiting_review | 208 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `6343783eb72e` | zero-rated | review | archived | 392 | ok |  |
| `19ed89ef24e0` | attack | review | awaiting_review | 205 | n/a |  |
| `1a72c320065f` | attack | review | awaiting_review | 410 | n/a |  |
