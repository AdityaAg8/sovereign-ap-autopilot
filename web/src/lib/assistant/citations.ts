/** Inline citation markers in assistant text (PRD §10 M9.3, ADR-0010 CITATION_PATTERN): splits a
 * reply into plain text and `[source:ref]` markers so the panel can render each marker as a link
 * to the record it cites. The pattern mirrors apx.agent.interfaces.CITATION_PATTERN exactly. */
import type { ChatCitation, CitationSource } from "@/lib/api/types";

const CITATION_RE = /\[(invoice|audit|vendor|policy|document|proposal):([A-Za-z0-9_:.-]{1,120})\]/g;

export type ReplyPart =
  { kind: "text"; text: string } | { kind: "citation"; source: CitationSource; ref: string };

export function splitCitations(text: string): ReplyPart[] {
  const parts: ReplyPart[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION_RE)) {
    const at = match.index ?? 0;
    if (at > last) parts.push({ kind: "text", text: text.slice(last, at) });
    parts.push({ kind: "citation", source: match[1] as CitationSource, ref: match[2]! });
    last = at + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

const SOURCE_LABEL: Record<CitationSource, string> = {
  invoice: "Invoice",
  audit: "Audit",
  vendor: "Vendor",
  policy: "Rule",
  document: "Document",
  proposal: "Proposal",
};

/** Short visible label for a citation chip, e.g. "Audit 41" or "Rule P4". */
export function citationLabel(source: CitationSource, ref: string): string {
  const shortRef = source === "invoice" || source === "proposal" ? ref.slice(0, 8) : ref;
  return `${SOURCE_LABEL[source]} ${shortRef}`;
}

export function findCitation(
  citations: readonly ChatCitation[],
  source: CitationSource,
  ref: string,
): ChatCitation | undefined {
  return citations.find((c) => c.source === source && c.ref === ref);
}
