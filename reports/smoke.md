# Smoke test report

Generated: 2026-09-22T14:02:10.125534+00:00
VRAM budget per stage: 11.5 GB

| Model | Stage | Reachable | Peak VRAM (GB) | Latency (ms) | Checks | Result |
|---|---|---|---|---|---|---|
| parse | parse | yes | 3.16 | 75233 | load=PASS; ocr_output=PASS | PASS |
| injection_guard_base | scan | yes | 1.41 | 5906 | load=PASS | PASS |
| extractor_a | extract | yes | 7.90 | 104984 | reachable=PASS; json_schema_output=PASS; thinking_disabled=PASS; logprobs=PASS; tool_call=PASS | PASS |
| extractor_b_base | extract | yes | 5.56 | 170402 | reachable=PASS; json_schema_output=PASS; thinking_disabled=PASS; logprobs=PASS | PASS |
| arbiter | arbitrate | yes | 9.74 | 109212 | reachable=PASS; json_schema_output=PASS; thinking_disabled=PASS; logprobs=PASS | PASS |
| text_embeddings | fraud | yes | 3.09 | 768276 | load=PASS; embeddings=PASS | PASS |
| page_similarity | fraud_page | yes | 5.94 | 271815 | load=PASS; embeddings=PASS | PASS |

## Per-stage VRAM budget

| Stage | Peak VRAM (GB) | Within budget |
|---|---|---|
| arbitrate | 9.74 | yes |
| extract | 7.90 | yes |
| fraud | 3.09 | yes |
| fraud_page | 5.94 | yes |
| parse | 3.16 | yes |
| scan | 1.41 | yes |

## Amendment 2026-09-23 (ADR-0013): extract group made sequential

Under real 5-page load, A + B co-resident took 10.3-10.6 GB (11.74-11.84 GB whole card with
a 1.1-1.4 GB desktop), which breaks the budget with the usual ~2.3 GB desktop. Per ADR-0002's
fallback rule the `extract` group now swaps (one of A/B loaded). Measured stage peaks, whole
card: A stage 8.42 GB, B stage 5.56 GB (desktop ~1.2 GB). See docs/adr/0013.
