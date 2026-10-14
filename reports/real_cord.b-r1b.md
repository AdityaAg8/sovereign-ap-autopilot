# Real-data eval: cord test, Extractor B r1b

n = 100, cord test, Extractor B `r1b` (extractor-b-r1b), in-distribution

- Scored on the 7 mapped CORD keys only (ADR-0019 §2).
- Generated at 2026-10-05T15:20:43.262039+00:00; normaliser `04d4eb8964f09d79`; git 00846e9371703f60d181283d2c54e93f85951d1d
- Mapped gold share: 76.02% of gold field instances

| Metric | Value | 95% CI | n |
|---|---|---|---|
| field_f1 | 90.70% | 87.64% to 93.35% (bootstrap) | 100 |
| nted_accuracy | 89.49% | 85.22% to 93.15% (bootstrap) | 100 |
| valid_json_rate | 100.00% | 96.30% to 100.00% (wilson) | 100 |

Valid-JSON target (≥ 99%): PASS. Failures: 0 of 100; every failure counts in every denominator.
Latency: p50 2.4 s, p95 5.0 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Donut field F1 | 84.10% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |
| Donut TED accuracy | 90.90% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |

Identity: model=extractor-b-r1b, gguf_sha256=366648ea66478d6c1baff5348efd6fc56e0478503bda04366229b0cda3e0c784, prompt_version=bd67d50c985fc8c7, serving_args_sha256=a39af6b2a9f7dcd9, parser_identity=a8832889e65c7fd5
