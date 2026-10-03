# Real-data before/after (Extractor B)

## cord test (n = 100, identical documents `5fbbcfa8ba7e`)

| Arm | Distribution | field_f1 | nted_accuracy |
|---|---|---|---|
| B v1 | zero-shot | 64.12% | 58.95% |
| B r1 | in-distribution | 90.52% | 88.69% |
| A base | zero-shot | 46.16% | 37.16% |

| Paired difference | Value | 95% CI | n |
|---|---|---|---|
| field_f1:b-r1 - b-v1 | +26.40% | 21.52% to 31.40% (paired_bootstrap) | 100 |
| field_f1:a-base - b-v1 | -17.95% | -26.23% to -10.06% (paired_bootstrap) | 100 |
