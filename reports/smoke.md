# Smoke test report (PRD §10 M0.5)

Generated: 2026-09-22T14:02:10.125534+00:00
VRAM budget per stage: 11.5 GB (PRD C2)

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
