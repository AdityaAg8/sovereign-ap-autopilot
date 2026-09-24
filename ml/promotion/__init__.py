"""M10.2 retrain-and-promote pipeline (PRD §10 M10.2, ADR-0011): the pure champion-vs-challenger
promotion decider over `evals.run` reports and the injected-dependency orchestration that runs
corrections export, training, shadow eval, the decision and the `champion` alias move."""
