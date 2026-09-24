"""attacks/generate.py: `python -m attacks.generate` builds the M6.3 attack-suite manifests
deterministically from one suite seed and writes them to `attacks/manifests/<category>.json`
(PRD §10 M6.3). The default run uses only `attacks.textsource.TemplateTextSource` and never
touches a GPU or the network; `--redteam` additionally asks the uncensored Qwen3.5-9B red-team
model (`attacks.textsource.LlamaSwapRedTeamTextSource`) to enrich every case's payload text and
writes the result straight back into the manifest, so it is fetched at most once per case
(PRD §10 M6.3: "cache outputs to manifests")."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from attacks.manifest import (
    CATEGORIES,
    CATEGORIES_BY_NAME,
    AttackCase,
    derive_goal_and_lang,
    derive_seed,
    save_cases,
)
from attacks.textsource import LlamaSwapRedTeamTextSource, RedTeamTextSource, TemplateTextSource
from factory.layouts import TaxCase, layout_ids

SUITE_SEED = 6300
CASES_PER_CATEGORY = 26
N_VENDORS = 12  # must match attacks.render.SUITE_N_VENDORS
_MAX_LINES = 10  # small on purpose: attack cases don't need factory's full 1-40 line range


def build_case(
    category: str, index: int, *, text_source: RedTeamTextSource, source_label: str
) -> AttackCase:
    """One deterministic case for `category` at `index` (0-based). Every field but
    `payload_text`/`source` is a pure function of `(SUITE_SEED, category, index)`, so swapping
    `text_source` (the default `TemplateTextSource` vs. an enrichment pass's
    `LlamaSwapRedTeamTextSource`) changes only the payload text, never which invoice a case
    renders against (PRD §10 M6.3: "deterministic from a seed")."""
    spec = CATEGORIES_BY_NAME[category]
    seed = derive_seed(SUITE_SEED, category, index)
    goal, lang = derive_goal_and_lang(category, seed)
    payload_text = text_source.generate(category=category, goal=goal, lang=lang, seed=seed)

    ids = layout_ids()
    tax_cases = list(TaxCase)
    return AttackCase(
        id=f"{spec.attack_id}-{index + 1:03d}",
        attack_id=spec.attack_id,
        category=category,
        goal=goal,
        lang=lang,
        payload_text=payload_text,
        placement=spec.placement,
        expected_detection_surface=spec.expected_detection_surface,
        seed=seed,
        source=source_label,
        layout_id=ids[derive_seed(seed, "layout") % len(ids)],
        vendor_index=derive_seed(seed, "vendor") % N_VENDORS,
        tax_case=tax_cases[derive_seed(seed, "tax") % len(tax_cases)].value,
        n_lines=1 + derive_seed(seed, "lines") % _MAX_LINES,
    )


def build_category_cases(
    category: str, *, text_source: RedTeamTextSource, source_label: str, n_cases: int
) -> list[AttackCase]:
    return [
        build_case(category, i, text_source=text_source, source_label=source_label)
        for i in range(n_cases)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m attacks.generate")
    parser.add_argument(
        "--redteam",
        action="store_true",
        help="enrich payload text via the llama-swap red-team model (needs a free GPU slot)",
    )
    parser.add_argument("--served-model", default="red-team")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--cases-per-category", type=int, default=CASES_PER_CATEGORY)
    args = parser.parse_args(argv)

    text_source: RedTeamTextSource
    source_label: str
    redteam_client: LlamaSwapRedTeamTextSource | None = None
    if args.redteam:
        redteam_client = LlamaSwapRedTeamTextSource(
            base_url=args.base_url, served_model=args.served_model
        )
        text_source, source_label = redteam_client, "redteam"
    else:
        text_source, source_label = TemplateTextSource(), "template"

    try:
        total = 0
        for spec in CATEGORIES:
            cases = build_category_cases(
                spec.name,
                text_source=text_source,
                source_label=source_label,
                n_cases=args.cases_per_category,
            )
            path = save_cases(spec.name, cases)
            print(f"{spec.name}: {len(cases)} cases -> {path}")
            total += len(cases)
        print(f"total: {total} cases across {len(CATEGORIES)} categories (source={source_label})")
        return 0
    except Exception as exc:
        if args.redteam:
            print(
                "attacks.generate --redteam failed (GPU busy or llama-swap unreachable): "
                f"{exc}\nRerun without --redteam for template-only manifests, or fix the "
                "llama-swap endpoint and retry.",
                file=sys.stderr,
            )
            return 1
        raise
    finally:
        if redteam_client is not None:
            redteam_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
