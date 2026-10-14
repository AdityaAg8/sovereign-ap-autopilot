/** One citation rendered inline in assistant text or a review note: a small mono
 * chip naming the cited record ("Audit 41"). It links to the record when it has a page, so every
 * claim the assistant makes is one click from its evidence. */
import Link from "next/link";
import type { CitationSource } from "@/lib/api/types";
import { citationLabel, splitCitations } from "@/lib/assistant/citations";

const CHIP =
  "mx-0.5 inline-flex items-baseline rounded border px-1 py-px align-baseline font-mono text-[0.6875rem] leading-none whitespace-nowrap";

export function CitationChip({
  source,
  citedRef,
  href,
}: {
  source: CitationSource;
  citedRef: string;
  href: string | null;
}) {
  const label = citationLabel(source, citedRef);
  if (!href) {
    return <span className={`${CHIP} border-border text-muted-foreground`}>{label}</span>;
  }
  return (
    <Link
      href={href}
      className={`${CHIP} border-accent/40 text-accent hover:bg-accent/10 focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none`}
      aria-label={`Open ${label.toLowerCase()}`}
    >
      {label}
    </Link>
  );
}

/** Text with its `[source:ref]` markers rendered as chips. `hrefFor` supplies each chip's link
 * (from the API's resolved citations, or computed for review notes on the audit tab). */
export function CitedText({
  text,
  hrefFor,
}: {
  text: string;
  hrefFor: (source: CitationSource, ref: string) => string | null;
}) {
  return (
    <>
      {splitCitations(text).map((part, index) =>
        part.kind === "text" ? (
          <span key={index}>{part.text}</span>
        ) : (
          <CitationChip
            key={index}
            source={part.source}
            citedRef={part.ref}
            href={hrefFor(part.source, part.ref)}
          />
        ),
      )}
    </>
  );
}
