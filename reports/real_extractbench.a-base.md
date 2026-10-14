# Real-data eval: extractbench eval, Extractor A base

n = 252, extractbench eval, Extractor A `base` (extractor-a), zero-shot

- Headline on the 245 `source:real` documents, then all 252.
- Predictions carry no citations, so grounding F1 is the true score of an answer that cites nothing.
- Generated at 2026-10-05T15:20:48.279073+00:00; normaliser `04d4eb8964f09d79`; git 82fdd889e65b6fc4218f46c250ca7f66afd6513f

| Metric | Value | 95% CI | n |
|---|---|---|---|
| value_f1.real | 70.39% | 66.54% to 73.88% (bootstrap) | 245 |
| grounding_f1.real | 0.00% | 0.00% to 0.00% (bootstrap) | 168 |
| value_f1.all | 70.26% | 66.67% to 73.57% (bootstrap) | 252 |
| grounding_f1.all | 0.00% | 0.00% to 0.00% (bootstrap) | 175 |
| valid_json_rate | 91.67% | 87.60% to 94.49% (wilson) | 252 |

Valid-JSON target (≥ 99%): FAIL. Failures: 21 of 252 (invalid_json 21); every failure counts in every denominator.
Latency: p50 16.1 s, p95 79.1 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Qwen3.5-9B one-shot value F1 (all 252) | 92.31% | no | authors' harness; ours differs in parser, 12 k-token slot, schema digest | ExtractBench authors' published result (PRD §5) |

Identity: model=extractor-a, gguf_sha256=6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293, prompt_version=5cf5b095f6253911, serving_args_sha256=3513a676929ccffe, parser_identity=66f415c5490d9e5b
