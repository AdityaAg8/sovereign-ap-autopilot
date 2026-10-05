# Demo run (M10.5)

`make demo` against a fresh database (`apx_demo`, its own inbox, mock-ASP store and Temporal history; docs/runbook.md "A second database"), 2026-10-05, stack images built that day from `29ea948`. Every verdict is asserted against the API by `scripts/demo.py`; the table is the final verdict per document from its run log.

**9/9 scenarios met their expected outcome.**

| Scenario | Document | Final status | Route | Result | Drop to verdict (s) | Detail |
|---|---|---|---|---|---:|---|
| poisoned | `07f7ed21c0aa` | blocked | block | pass | 280 | blocked by the scan |
| poisoned | `0f4d3fb87a2f` | blocked | block | pass | 393 | blocked by the scan |
| duplicate | `011e5a82b846` | awaiting_review | review | pass | 444 | flagged: duplicate_invoice, near_duplicate_embedding, … (cut at the console width in the run log) |
| bank change | `b95321b393fe` | awaiting_review | review | pass | 450 | held by P1, bank_detail_change flag |
| clean | `0082e402ae77` | archived | review | pass | 535 | archived after approval |
| clean | `0dc1c271348c` | archived | review | pass | 533 | archived after approval |
| clean | `11eb6f7b9bed` | archived | review | pass | 531 | archived after approval |
| bilingual | `0298f9afdc61` | archived | review | pass | 529 | archived after approval |
| bilingual | `1cf8f48b3769` | archived | review | pass | 529 | archived after approval |

Scenarios: two poisoned invoices (hidden injected instructions) must be blocked by the scan; a resubmitted duplicate must be held with a duplicate flag; an invoice paying a just-changed bank account must be held by P1; clean English and bilingual invoices must be archived after a human approval. `docs/demo.md` walks through each one.
