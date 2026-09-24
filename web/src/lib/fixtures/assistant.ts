/** Fixture-mode stand-ins for the M9.3 audit trail and ops assistant (PRD §10 M9.3): each
 * invoice gets a deterministic audit trail derived from its status, route and fraud flags, and a
 * scripted assistant answers "why was X blocked / held" from that trail with real citations, so
 * the chat panel and its e2e test run without apx/api or a GPU. */
import type {
  AuditEventView,
  ChatCitation,
  ChatReply,
  ChatStep,
  ChatTurn,
  InvoiceDetail,
} from "@/lib/api/types";
import { splitCitations } from "@/lib/assistant/citations";

/** apx.agent.interfaces.REFUSAL_TEMPLATES["action_not_permitted"], verbatim. */
const ACTION_NOT_PERMITTED =
  "I can't do that. I can only read records and draft proposals; a user with the right role " +
  "must approve, pay, execute or change bank details in the app.";

const ACTION_RE =
  /\b(pay|approve|release|execute|wire|transfer|send (the )?payment|change (the )?(bank|iban))\b/i;

function stableId(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 90000) + 10000;
}

export function invoiceNumber(invoice: InvoiceDetail): string {
  return (
    invoice.fields.find((f) => f.name === "invoice_number")?.raw ?? invoice.summary.id.slice(0, 12)
  );
}

function reviewRule(invoice: InvoiceDetail): string {
  return invoice.flags[0]?.rule ?? "low_confidence";
}

/** The invoice's audit trail, newest first. Ids are stable per invoice. */
export function fixtureAudit(invoice: InvoiceDetail): AuditEventView[] {
  const base = stableId(invoice.summary.id) * 10;
  const start = Date.parse(invoice.summary.receivedAt);
  const at = (seconds: number) => new Date(start + seconds * 1000).toISOString();
  const events: AuditEventView[] = [
    { id: base + 1, ts: at(2), actor: "workflow", action: "activity:scan", facts: {}, note: null },
    {
      id: base + 2,
      ts: at(40),
      actor: "workflow",
      action: "activity:extract",
      facts: {},
      note: null,
    },
    {
      id: base + 3,
      ts: at(44),
      actor: "workflow",
      action: "activity:validate",
      facts: {},
      note: null,
    },
  ];
  invoice.flags.slice(0, 4).forEach((flag, index) => {
    events.push({
      id: base + 4 + index,
      ts: at(46 + index),
      actor: "workflow",
      action: "fraud.flag",
      facts: { rule: flag.rule, kind: flag.severity },
      note: null,
    });
  });
  if (invoice.summary.route) {
    events.push({
      id: base + 8,
      ts: at(50),
      actor: "workflow",
      action: "activity:route",
      facts: {
        route: invoice.summary.route,
        rule: invoice.summary.route === "auto" ? null : reviewRule(invoice),
      },
      note: null,
    });
  }
  if (invoice.summary.status === "awaiting_review") {
    const first = invoice.flags[0];
    events.push({
      id: base + 9,
      ts: at(75),
      actor: "agent",
      action: "agent.review_note",
      facts: {},
      note:
        `Held for review by the ${reviewRule(invoice).replace(/_/g, " ")} check [audit:${base + 8}]. ` +
        (first
          ? `Check this first: ${first.explanation} [audit:${base + 4}]`
          : "Check the fields marked for review before deciding."),
    });
  }
  return events.reverse();
}

function withLinks(text: string, invoice: InvoiceDetail | null): ChatCitation[] {
  const seen = new Set<string>();
  const out: ChatCitation[] = [];
  for (const part of splitCitations(text)) {
    if (part.kind !== "citation") continue;
    const key = `${part.source}:${part.ref}`;
    if (seen.has(key)) continue;
    seen.add(key);
    let href: string | null = null;
    if (part.source === "invoice") href = `/invoices/${part.ref}`;
    if (part.source === "audit" && invoice) {
      href = `/invoices/${invoice.summary.id}?tab=audit&event=${part.ref}`;
    }
    out.push({ source: part.source, ref: part.ref, href });
  }
  return out;
}

function explain(invoice: InvoiceDetail): string {
  const number = invoiceNumber(invoice);
  const trail = fixtureAudit(invoice);
  const route = trail.find((e) => e.action === "activity:route");
  const flags = trail.filter((e) => e.action === "fraud.flag").reverse();
  const top = invoice.flags[0];
  switch (invoice.summary.status) {
    case "blocked":
      return (
        `${number} was blocked [audit:${route?.id}]. ` +
        (top && flags[0] ? `${top.explanation} [audit:${flags[0].id}]` : "") +
        (flags.length > 1
          ? ` It also tripped ${flags.length - 1} other check${flags.length > 2 ? "s" : ""} ` +
            flags
              .slice(1)
              .map((f) => `[audit:${f.id}]`)
              .join(" ") +
            "."
          : "") +
        " A blocked invoice can only be rejected; nothing will be paid."
      );
    case "awaiting_review":
      return (
        `${number} is waiting for a reviewer because of the ` +
        `${reviewRule(invoice).replace(/_/g, " ")} check [audit:${route?.id}].` +
        (top && flags[0] ? ` ${top.explanation} [audit:${flags[0].id}]` : "")
      );
    case "approved":
    case "sent":
    case "archived":
      return `${number} passed every check and was approved automatically [audit:${route?.id}].`;
    default:
      return `${number} is still being processed; it has reached the ${invoice.summary.status.replace(/_/g, " ")} stage [invoice:${invoice.summary.id}].`;
  }
}

/** Scripted assistant over the fixture store. Emits the same steps the real agent would. */
export async function fixtureAssistant(
  history: readonly ChatTurn[],
  invoices: Iterable<InvoiceDetail>,
  onStep: (step: ChatStep) => void,
  pause: (ms: number) => Promise<void>,
): Promise<ChatReply> {
  const question = [...history].reverse().find((t) => t.role === "user")?.text ?? "";
  if (ACTION_RE.test(question) && !/\bwhy\b/i.test(question)) {
    return {
      text: ACTION_NOT_PERMITTED,
      refusal: "action_not_permitted",
      citations: [],
      proposals: [],
    };
  }
  await pause(350);
  onStep({ tool: "search_invoices", label: "Searching invoices", outcome: "ok" });
  const lower = question.toLowerCase();
  const all = [...invoices];
  const match = all.find(
    (inv) =>
      lower.includes(invoiceNumber(inv).toLowerCase()) ||
      lower.includes(inv.summary.id.toLowerCase()),
  );
  if (!match) {
    const example = all.find((inv) => inv.summary.status === "blocked") ?? all[0];
    return {
      text:
        "I couldn't find an invoice matching that. Give me its invoice number" +
        (example ? `, for example ${invoiceNumber(example)}.` : "."),
      refusal: null,
      citations: [],
      proposals: [],
    };
  }
  await pause(500);
  onStep({ tool: "explain_decision", label: "Checking the audit trail", outcome: "ok" });
  await pause(400);
  const text = explain(match);
  return { text, refusal: null, citations: withLinks(text, match), proposals: [] };
}
