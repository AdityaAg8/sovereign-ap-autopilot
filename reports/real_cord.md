# Real-data eval: cord test, Extractor B v1

n = 100, cord test, Extractor B `v1` (extractor-b-v1), zero-shot

- Scored on the 7 mapped CORD keys only (ADR-0019 §2).
- Generated at 2026-10-05T15:20:43.600927+00:00; normaliser `04d4eb8964f09d79`; git b6b0e1c422857f9f5c96912270a9fa7c237d2d59
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

# Real-data eval: cord test, Extractor B r1

n = 100, cord test, Extractor B `r1` (extractor-b-r1), in-distribution

- Scored on the 7 mapped CORD keys only (ADR-0019 §2).
- Generated at 2026-10-05T15:20:42.974609+00:00; normaliser `04d4eb8964f09d79`; git 073e6c3d8d49e808d74f61032b8f972fcc229ce0
- Mapped gold share: 76.02% of gold field instances

| Metric | Value | 95% CI | n |
|---|---|---|---|
| field_f1 | 90.52% | 87.29% to 93.17% (bootstrap) | 100 |
| nted_accuracy | 88.69% | 83.97% to 92.66% (bootstrap) | 100 |
| valid_json_rate | 100.00% | 96.30% to 100.00% (wilson) | 100 |

Valid-JSON gate (PRD ≥ 99%): PASS. Failures: 0 of 100; every failure counts in every denominator.
Latency: p50 2.6 s, p95 5.5 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Donut field F1 | 84.10% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |
| Donut TED accuracy | 90.90% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |

Identity: model=extractor-b-r1, gguf_sha256=11ce0660b74abe95f64fee509396b1ec8b78888ef7d43c2bf42c85e2f3eea246, prompt_version=6a9d2adb7b011729, serving_args_sha256=0b433ec65c6548bf, parser_identity=a8832889e65c7fd5

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

Valid-JSON gate (PRD ≥ 99%): PASS. Failures: 0 of 100; every failure counts in every denominator.
Latency: p50 2.4 s, p95 5.0 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Donut field F1 | 84.10% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |
| Donut TED accuracy | 90.90% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |

Identity: model=extractor-b-r1b, gguf_sha256=366648ea66478d6c1baff5348efd6fc56e0478503bda04366229b0cda3e0c784, prompt_version=bd67d50c985fc8c7, serving_args_sha256=a39af6b2a9f7dcd9, parser_identity=a8832889e65c7fd5

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
