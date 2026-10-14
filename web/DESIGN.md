# Design system: "Ledger"

The visual system for the Sovereign AP review UI. Read this before adding a screen. Every color,
shadow and font in the app resolves to a token in `src/app/globals.css`; components use the
Tailwind utilities those tokens generate (`bg-card`, `text-muted-foreground`, `shadow-panel`)
and never a raw hex or OKLCH value.

## Direction

A financial control room: calm, precise, dense where the data lives. The reference points are
the best finance and ops tools (Linear, Mercury, Ramp), not a dashboard template. One teal
accent, taken from the rule color of the invoice documents themselves, ties the interface to
the paper it is checking. Status colors appear only where they carry meaning.

**Signature: the accountant's double rule.** The invoice total sits on a teal double underline
(`.ledger-total`), the mark bookkeepers draw under a final figure, and the brand mark is an
auditor's tick over the same rule. It is used in those two places only.

## Type

| Role | Face                      | Use                                                          |
| ---- | ------------------------- | ------------------------------------------------------------ |
| UI   | Instrument Sans, variable | labels, names, headings, KPI values (with `tabular-nums`)    |
| Data | Geist Mono, variable      | amounts, invoice numbers, TRNs, IBANs, dates and times       |
| AR   | IBM Plex Sans Arabic      | bilingual vendor names (`lang="ar"`), loaded on vendors only |

Both Latin faces are self-hosted through `next/font` (no runtime CDN). The data face is not
preloaded because it never renders the LCP text. `.tabular-mono` is the one class for figures.

Scale (rem, 16px root, compact for table density): `2xs` 11px for meta and keycaps, `xs` 12px
for labels and secondary text, `sm` 13px for rows and body, `base` 14px for page titles and the
invoice total, `xl` 20px for the operations headline and login heading, 26px for KPI values.
Weights: 400 body, 500 labels and values, 600 headings. Headings track at -0.01em to -0.025em.

### Numbers: Latin digits, always

Every amount and count goes through `src/lib/format.ts`: `en-AE` grouping pinned to Latin
digits, two decimals for money, `dir="ltr"` on figures. Text that does not parse cleanly is shown
as extracted, never rounded. `<Amount>` sets the currency code back in the muted tone without
changing the text content.

## Color (OKLCH)

Light and dark are designed as a pair. In dark mode elevation reads as lightness
(sidebar < canvas < panel < popover) and shadows are dropped.

| Token                | Light                  | Dark                   | Role                                 |
| -------------------- | ---------------------- | ---------------------- | ------------------------------------ |
| `--background`       | oklch(0.979 0.004 205) | oklch(0.172 0.012 232) | canvas                               |
| `--card`             | oklch(1 0 0)           | oklch(0.206 0.014 232) | panels                               |
| `--foreground`       | oklch(0.2 0.02 235)    | oklch(0.94 0.008 210)  | primary text                         |
| `--muted-foreground` | oklch(0.47 0.02 232)   | oklch(0.71 0.016 220)  | secondary text                       |
| `--border`           | oklch(0.915 0.008 215) | oklch(0.285 0.016 232) | hairlines                            |
| `--input`            | oklch(0.665 0.014 220) | oklch(0.51 0.018 228)  | control borders (3:1)                |
| `--accent`, `--ring` | oklch(0.48 0.085 192)  | oklch(0.79 0.1 185)    | teal: selection, focus, links, bars  |
| `--primary`          | oklch(0.46 0.085 192)  | oklch(0.79 0.1 185)    | primary button (Sign in)             |
| `--success`          | oklch(0.47 0.11 158)   | oklch(0.76 0.13 160)   | approved, Auto route, Approve button |
| `--warning`          | oklch(0.52 0.12 66)    | oklch(0.81 0.12 80)    | needs review, review route           |
| `--destructive`      | oklch(0.51 0.19 27)    | oklch(0.72 0.16 25)    | blocked, reject, errors              |
| `--sidebar`          | oklch(0.235 0.03 222)  | oklch(0.142 0.012 232) | the ink-teal spine (dark in both)    |
| `--sidebar-primary`  | oklch(0.8 0.1 185)     | oklch(0.79 0.1 185)    | brand tile, active nav marker        |

All text pairs pass WCAG 2.1 AA (4.5:1; most are above 6:1) and control borders reach 3:1. The
sidebar is dark in both themes, so `[data-surface="spine"]` switches the focus ring to the
lifted teal. Status is never color alone: every dot, icon or bar sits beside a word or a number.

## Surfaces and elevation

- `panel` utility: card background, 1px border, `--radius-xl` (12.6px) and `shadow-panel` (a
  1px shadow in light mode, none in dark). It is the only container style on the canvas.
- `shadow-bar` lifts the sticky review action bar; `shadow-paper` lifts the invoice page.
- `--radius` is 0.5625rem: buttons and inputs at 9px, panels at 12.6px.
- Page gutters are `p-3` on phones and `p-5` from `sm`; panels pad `px-4`/`px-5`.

## Components

- **App shell.** 240px spine (56px icon rail below `md`) with grouped navigation (Invoices,
  Records, System), live Inbox and Review queue counts fetched when the browser is idle, and the
  account menu at the bottom. The topbar holds the page title, search (⌘K), Ask (C) and the
  theme toggle.
- **Tables.** Tinted header row in `2xs`, hairline row separators, 44px rows, right-aligned mono
  figures. Status is a 6px dot plus label, route an icon plus label, flags a compact red count
  badge. Active rows take a faint teal tint; keyboard focus draws a 2px inset ring.
- **Confidence.** Percent plus a 48px bar, teal when high, warning or danger when low.
- **Invoice review.** Document left, field column right (400px). Each field row is label and
  review state, value with an edit button, then source and confidence. Rows that need review
  are tinted, and only their boxes are outlined on the document (dashed); the selected field's
  box and row highlight together in teal. Underline tabs; the action bar is raised and shows
  `A`, `F`, `R` keycaps at full label contrast.
- **Operations.** A live band (headline sentence plus an in-flight trace), five KPI cards that
  draw only what the snapshot backs (share bars, median against p95), the stage flow with bars
  sized against the busiest station beside the settled breakdown, then activity beside the
  models on the GPU.
- **Login.** Split layout: product summary on the spine, sign-in form and the demo account
  picker on the canvas. On phones only the form remains, under the brand mark.

shadcn/ui primitives live in `src/components/ui/*`; menu items highlight with `bg-muted`, never
the accent.

## Motion

`--duration-fast` 120ms, `--duration-base` 180ms, `--duration-slow` 240ms; hover and selection
color changes run at 150ms. Custom `--ease-out`, `--ease-in` and `--ease-in-out` curves replace
the browser default. Only color, opacity and transform animate, never layout. Motion that
carries meaning: the live dot and loaded models ping softly, and an activity row glows once
when its invoice moves. `prefers-reduced-motion: reduce` removes the pings and collapses every
other transition to an instant change.

## Focus

`:focus-visible` only, 2px solid `--ring` with a 2px offset, never animated in. Rows inside
scrolling lists use an inset ring (`outline-offset: -2px`) so the scroll edge cannot clip it.
Tailwind v4's `outline-none` sets the outline style to none, so a ring layered on top must
re-state `outline-solid`.

## Browser surfaces

`color-scheme` follows the theme, text selection uses a teal tint, the caret is teal and
scrollbars use the border color. The tab icon (`src/app/icon.svg`) is the brand mark.
