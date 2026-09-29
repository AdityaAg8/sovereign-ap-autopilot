"""Entry point for `python -m evals.real run|parse|report|before-after|reality-template`
(ADR-0019 §8); the logic lives in evals.real.run."""

from evals.real.run import main

raise SystemExit(main())
