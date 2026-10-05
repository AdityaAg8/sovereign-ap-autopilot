# Real-data eval: cord test, Extractor A base

n = 100, cord test, Extractor A `base` (extractor-a), zero-shot

- Scored on the 7 mapped CORD keys only (ADR-0019 §2).
- Generated at 2026-10-05T15:20:42.670149+00:00; normaliser `04d4eb8964f09d79`; git ea6f25c5febbcbd2d246beedcfa932be0bdb7a50
- Mapped gold share: 76.02% of gold field instances

| Metric | Value | 95% CI | n |
|---|---|---|---|
| field_f1 | 46.16% | 38.51% to 52.91% (bootstrap) | 100 |
| nted_accuracy | 37.16% | 32.26% to 42.05% (bootstrap) | 100 |
| valid_json_rate | 100.00% | 96.30% to 100.00% (wilson) | 100 |

Valid-JSON gate (PRD ≥ 99%): PASS. Failures: 0 of 100; every failure counts in every denominator.
Latency: p50 3.9 s, p95 8.2 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Donut field F1 | 84.10% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |
| Donut TED accuracy | 90.90% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |

Identity: model=extractor-a, gguf_sha256=6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293, prompt_version=cdcd2832034c55e3, serving_args_sha256=3513a676929ccffe, parser_identity=a8832889e65c7fd5
