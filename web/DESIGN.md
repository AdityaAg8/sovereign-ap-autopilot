# Design system — "Ledger"

Locked design direction for the Sovereign AP Autopilot review UI (M5.4). Read this before adding
any screen; every color and font declaration in the app must reference a token defined here.

## Process note (Hallmark adaptation)

Hallmark's macrostructure/nav-archetype/footer-archetype apparatus targets marketing pages (hero,
footer, public nav). This is an authenticated, dense, keyboard-first **operator app**, not a
landing page — there is no hero and no footer. Per Hallmark's own component-scope guidance, the
disciplines that still apply in full are: locked tokens (no ad-hoc hex/OKLCH in components),
typography purity (no italic headers), full interaction-state coverage, honest copy, motion
restraint, contrast/focus/keyboard rigor, and the 320–768px responsiveness floor. Those are what
the slop-test pass below checks. Frontend-design and ui-ux-pro-max were used for the token system,
type pairing and component patterns (dense tables, command palette, detail/side-panel review).

## Direction

**Ledger**: a precise, technical, restrained finance-operator system — closer to a Bloomberg
terminal or Mercury Bank's ops console than a generic SaaS dashboard. No warm-cream/serif
(AI-default #1), no near-black+neon-accent cliché (#2), no broadsheet hairlines (#3): those are
marketing-page defaults, and this is a control surface for money. The single signature move:
**synchronized grounding** — selecting a field in the field list simultaneously highlights its
literal bounding box on the source document and its confidence readout, in the same cobalt accent
and the same instant — because the product's whole trust claim is that every value traces back to
a literal box in the source page, never a hallucinated one. Every other surface stays quiet so
that synchronization reads clearly.

## Type

- **UI / display face:** IBM Plex Sans — a grotesk built for data-dense technical interfaces
  (originally IBM's own product typeface), not a marketing sans.
- **Arabic pairing:** IBM Plex Sans Arabic — drawn by the same foundry program as Plex Sans, so
  x-height, stroke contrast and weight steps match instead of the common mismatched
  Latin-sans + Noto-Arabic pairing. Applied automatically whenever `dir="rtl"` / `lang="ar"`.
  Every place invoice content is bilingual (vendor names, line items) sets `lang`/`dir` per span so
  Arabic text always renders in Plex Sans Arabic even inside an LTR shell.
- **Data / mono face:** IBM Plex Mono — tabular figures for amounts, dates, TRNs, IBANs, invoice
  IDs and timestamps, so columns of numbers align and never reflow on hover/selection. Mono is
  for those values only — labels, names, descriptions and prose stay in Plex Sans.

### Numbers: Latin digits, always

Every amount and count goes through one formatter (`src/lib/format.ts`): `en-AE` grouping (comma
thousands, period decimals, two decimals for money) pinned to Latin digits (`nu-latn`), whatever
the browser locale or the invoice's language. Arabic-Indic input digits are read (with the same
comma-vs-period rule as `apx.common.normalize.normalize_amount`) but always displayed in Latin
digits, so an Arabic invoice's totals line up with the English inbox. Money and IDs render
`dir="ltr"` so bidi never reorders them. Text that does not parse cleanly, or has more than two
decimals, is shown as extracted — never rounded or reinterpreted into a different number.

### Scale (rem, 16px root; UI defaults to compact `sm` size for table density)

| Token         | Size             | Use                                  |
| ------------- | ---------------- | ------------------------------------ |
| `--text-2xs`  | 0.6875rem / 11px | timestamps, table meta, kbd hints    |
| `--text-xs`   | 0.75rem / 12px   | captions, badges, secondary labels   |
| `--text-sm`   | 0.8125rem / 13px | default table/list row text          |
| `--text-base` | 0.875rem / 14px  | body text, form inputs, field values |
| `--text-md`   | 1rem / 16px      | section headers, emphasized labels   |
| `--text-lg`   | 1.125rem / 18px  | page titles                          |
| `--text-xl`   | 1.375rem / 22px  | dialog / panel headers               |
| `--text-2xl`  | 1.75rem / 28px   | dashboard hero numbers               |
| `--text-3xl`  | 2.25rem / 36px   | login brand mark                     |

Line-height 1.4 for UI text, 1.6 for prose (help overlay, empty states). Weights: 400 body, 500
labels/emphasis, 600 headers, 700 reserved for the brand mark only.

## Color (OKLCH, semantic tokens only — see `src/app/globals.css` for the source of truth)

Light and dark are designed together, not inverted. Both pass WCAG 2.1 AA (4.5:1 body text, 3:1
large text/UI outlines) — verified in the accessibility pass below.

| Token                 | Light                 | Dark                   | Role                                          |
| --------------------- | --------------------- | ---------------------- | --------------------------------------------- |
| `--background`        | oklch(0.99 0.002 95)  | oklch(0.17 0.009 260)  | app canvas                                    |
| `--foreground`        | oklch(0.19 0.012 260) | oklch(0.94 0.006 260)  | primary text                                  |
| `--card`              | oklch(1 0 0)          | oklch(0.205 0.010 260) | panels, rows on hover                         |
| `--muted`             | oklch(0.96 0.004 260) | oklch(0.24 0.011 260)  | subtle fills                                  |
| `--muted-foreground`  | oklch(0.47 0.014 260) | oklch(0.68 0.013 260)  | secondary text                                |
| `--border`            | oklch(0.90 0.006 260) | oklch(0.30 0.013 260)  | hairlines                                     |
| `--primary`           | oklch(0.32 0.06 260)  | oklch(0.80 0.09 260)   | primary buttons                               |
| `--accent` / `--ring` | oklch(0.55 0.19 260)  | oklch(0.74 0.15 260)   | focus, active, links, the grounding highlight |
| `--destructive`       | oklch(0.48 0.19 25)   | oklch(0.70 0.18 25)    | reject, block, errors                         |
| `--success`           | oklch(0.42 0.12 155)  | oklch(0.72 0.15 155)   | auto-route, approved, the Approve button fill |
| `--warning`           | oklch(0.44 0.13 75)   | oklch(0.78 0.14 80)    | review-route, low confidence                  |

Approve is a solid `--success` fill with `--success-foreground` text in both themes; only its
disabled state (50% opacity) may look dimmed. Shortcut hints on action buttons (`A`, `F`, `R`) are
text: they keep the label's full color (≥ 4.5:1) and are set apart by a faint keycap border, never
by reduced opacity.

Route/status color is **never** the only signal: every badge pairs color with an icon and text
(`Auto`, `Review`, `Blocked`), per WCAG SC 1.4.1.

## Spacing, radii, layout

4px base scale (Tailwind default `--spacing`). Dense-table rows use `py-1.5 px-3`; detail panels
use `p-4`/`p-6`; page gutters `px-6` desktop / `px-4` at ≤640px. `--radius: 0.5rem` (8px) — precise
without being sharp (brutalist) or soft (consumer-app). Sidebar fixed at 240px (56px collapsed);
content max-width unconstrained (operators want the full data-dense width).

## Motion

`--dur-fast: 120ms`, `--dur-base: 180ms`, `--dur-slow: 240ms`. `--ease-out` for entrances/hover,
`--ease-in` for exits, `--ease-in-out` for state toggles — never the browser default `ease`, never
spring/bounce on UI chrome. Only `transform`/`opacity` animate. `prefers-reduced-motion: reduce`
collapses everything to an instant or ≤1ms crossfade. The one deliberate motion moment is the
grounding highlight: the PDF field box and its list row both transition `outline-color`/`opacity`
over `--dur-base` in lockstep — everything else on the page is static.

## Focus states

`:focus-visible` only (never on mouse click) — 2px solid `--ring`, 2px offset, `border-radius`
matching the element. Never animated in (must appear the instant focus lands). Every interactive
element (row, button, tab, field, command item) has a visible focus ring.

**Rows inside scrolling lists use an inset ring instead** (`outline-offset: -2px`): a 2px-outside
ring on a row would be clipped by the list's overflow edge and cut through its text. The inbox
list keeps DOM focus on the listbox (`aria-activedescendant`) and draws the ring on the active row
— 1px while it is only the hovered/active row, 2px inset while the list has keyboard focus — so
keyboard focus and the visual "active row" stay identical. Invoice field rows get the same 2px
inset ring on `:focus-visible`. (Tailwind v4's `outline-none` sets the outline style to none, so
any ring layered on top of it must re-state `outline-solid`.)

## Components

shadcn/ui installed via the CLI (`components.json`, Radix primitives, `radix-nova` base style,
RTL enabled) — see `src/components/ui/*`. Tokens above override the generated defaults in
`src/app/globals.css`; components never hardcode a color or font, only Tailwind utility classes
that resolve to the tokens (`bg-background`, `text-muted-foreground`, `font-mono`, etc).

## Accessibility and performance budgets

WCAG 2.1 AA on every screen; Lighthouse performance + accessibility ≥ 95 on `/login` and `/inbox`
(production build, measured with the Lighthouse CLI — see the M5.4 report for numbers); zero
layout shift (skeletons reserve space, fonts use `display: swap` with metric-matched fallbacks);
lean bundles (route-level code, no client bundle for the fixture data unless the page needs it).
