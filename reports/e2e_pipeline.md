# End-to-end pipeline run (M5.6)

20 documents dropped into the inbox of the running stack; review-routed invoices approved through the API by a seeded approver.

- Factory invoices archived (UBL built, schematron passed, sent, archived): **10/18**
- Attack PDFs kept out of payment: **2/2**
- Seconds from drop to routing decision: median 1384, max 1393

| Document | Kind | Route | Final status | Drop to route (s) | Fields vs truth | Notes |
|---|---|---|---|---|---|---|
| `1b646a0b272a` | EN digital | review | awaiting_review | 1382 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `4149f2c72c37` | EN digital | review | archived | 1388 | ok |  |
| `6639482345a9` | EN digital | review | archived | 1377 | ok |  |
| `8596c8bd3c14` | EN digital | review | archived | 1379 | ok |  |
| `8690e5cfac04` | EN digital | review | awaiting_review | 1392 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `8a8e32f11788` | EN digital | review | archived | 1378 | ok |  |
| `8bbc5bc20b63` | EN digital | review | awaiting_review | 1382 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `8e7c09d203a6` | EN digital | review | archived | 1387 | ok |  |
| `9070cee968bf` | EN digital | review | archived | 1375 | ok |  |
| `2d8d80c267c2` | AR-first digital | review | awaiting_review | 1386 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `49758be16c17` | AR-first digital | review | archived | 1384 | ok |  |
| `4cc167f4d676` | AR-first digital | review | awaiting_review | 1386 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `5078851300fb` | AR-first digital | review | awaiting_review | 1383 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `23c4bf3523c6` | scanned | review | archived | 1389 | ok |  |
| `2475ada75265` | scanned | review | archived | 1375 | ok |  |
| `3dcc7a45007f` | scanned | review | archived | 1392 | ok |  |
| `4555e119d731` | scanned | review | awaiting_review | 1390 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `0704c3d4c17f` | zero-rated | review | awaiting_review | 1381 | ok | approve refused: 403 {"detail":"action approve_payment denied: vendor has no unique admin-verified account"} |
| `0429b9c520cb` | attack | review | awaiting_review | 1385 | n/a |  |
| `054e8eb11d1b` | attack | review | awaiting_review | 1393 | n/a |  |
