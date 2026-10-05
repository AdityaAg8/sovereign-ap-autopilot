# Real-data eval: cord test, Extractor B v1

n = 100, cord test, Extractor B `v1` (extractor-b-v1), zero-shot

- Scored on the 7 mapped CORD keys only (ADR-0019 §2).
- Generated at 2026-10-05T05:16:08.499522+00:00; normaliser `04d4eb8964f09d79`; git b6b0e1c422857f9f5c96912270a9fa7c237d2d59
- Mapped gold share: 76.02% of gold field instances

| Metric | Value | 95% CI | n |
|---|---|---|---|
| field_f1 | 64.12% | 59.11% to 68.49% (bootstrap) | 100 |
| nted_accuracy | 58.95% | 53.20% to 64.56% (bootstrap) | 100 |
| valid_json_rate | 100.00% | 96.30% to 100.00% (wilson) | 100 |

Valid-JSON gate (PRD ≥ 99%): PASS. Failures: 0 of 100; every failure counts in every denominator.
Latency: p50 3.3 s, p95 5.9 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Donut field F1 | 84.10% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |
| Donut TED accuracy | 90.90% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |

Identity: model=extractor-b-v1, gguf_sha256=5269de7351ff4571385f2dfd7e7863ebb68b8f97939c3dccc770e1964ae8a866, prompt_version=6a9d2adb7b011729, serving_args_sha256=8bf08066be67b0b8, parser_identity=a8832889e65c7fd5
