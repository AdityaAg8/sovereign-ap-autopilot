# Smoke test report (PRD §10 M0.5)

Generated: 2026-09-22T11:14:41.492502+00:00
VRAM budget per stage: 11.5 GB (PRD C2)

| Model | Stage | Reachable | Peak VRAM (GB) | Latency (ms) | Checks | Result |
|---|---|---|---|---|---|---|
| parse | parse | no | 2.45 | 0 | load=FAIL (transformers not installed: No module named 'torch') | FAIL |
| injection_guard_base | scan | no | 2.45 | 0 | load=FAIL (transformers not installed: No module named 'torch') | FAIL |
| extractor_a | extract | no | 2.45 | n/a | reachable=FAIL (ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it) | FAIL |
| extractor_b_base | extract | no | 2.45 | n/a | reachable=FAIL (ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it) | FAIL |
| arbiter | arbitrate | no | 2.45 | n/a | reachable=FAIL (ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it) | FAIL |
| text_embeddings | fraud | no | 2.45 | 0 | load=FAIL (transformers not installed: No module named 'torch') | FAIL |
| page_similarity | fraud_page | no | 2.45 | 0 | load=FAIL (transformers not installed: No module named 'torch') | FAIL |

## Per-stage VRAM budget

| Stage | Peak VRAM (GB) | Within budget |
|---|---|---|
| arbitrate | 2.45 | yes |
| extract | 2.45 | yes |
| fraud | 2.45 | yes |
| fraud_page | 2.45 | yes |
| parse | 2.45 | yes |
| scan | 2.45 | yes |
