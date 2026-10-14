/** Fixture-mode stand-ins for the audit trail and ops assistant: each
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

/** The invoice's audit trail, newest first, in the vocabulary apx writes: one `activity:<step>`
 * event per workflow step (apx.workflows.activities `_audit`, one `activity:fraud` per invoice
 * with its flag count), then the exception agent's note or the delivery steps. Ids are stable
 * per invoice and increase along the trail, like the hash chain's. */
export function fixtureAudit(invoice: InvoiceDetail): AuditEventView[] {
  let nextId = stableId(invoice.summary.id) * 20;
  const start = Date.parse(invoice.summary.receivedAt);
  const events: AuditEventView[] = [];
  const add = (
    seconds: number,
    action: string,
    facts: AuditEventView["facts"] = {},
    actor = "workflow",
    note: string | null = null,
  ): number => {
    const id = ++nextId;
    events.push({
      id,
      ts: new Date(start + seconds * 1000).toISOString(),
      actor,
      action,
      facts,
      note,
    });
    return id;
  };
  const { status, route } = invoice.summary;

  add(2, "activity:scan", { flags: 0 });
  add(9, "activity:parse", { pages: invoice.pageCount });
  add(40, "activity:extract");
  add(42, "activity:trust_gate");
  add(43, "activity:build_invoice");
  add(44, "activity:validate");
  const reached = route !== null || status === "fraud_checked";
  if (!reached) return events.reverse();

  const rules = invoice.flags.map((f) => f.rule);
  const fraudId = add(46, "activity:fraud", {
    flags: invoice.flags.length,
    rules: rules.length > 0 ? [...new Set(rules)].join(", ") : null,
  });
  if (route === null) return events.reverse();

  const routeId = add(50, "activity:route", {
    route,
    rule: route === "auto" ? null : reviewRule(invoice),
  });
  if (status === "awaiting_review") {
    const first = invoice.flags[0];
    add(
      75,
      "agent.review_note",
      {},
      "agent",
      `Held for review by the ${reviewRule(invoice).replace(/_/g, " ")} check [audit:${routeId}]. ` +
        (first
          ? `Check this first: ${first.explanation} [audit:${fraudId}]`
          : "Check the fields marked for review before deciding."),
    );
  }
  const cleared = status === "approved" || status === "sent" || status === "archived";
  if (route === "review" && cleared) {
    // A reviewer approved it: the policy check, the decision and the payment-state write
    // (apx.policy.guard, apx.api.service, apx.policy.writer), in the order apx records them.
    add(90, "authorize:approve_payment", { allowed: true }, "reviewer");
    add(91, "decision.approve", {}, "reviewer");
    add(91, "payment_state:approve", {}, "reviewer");
  }
  const delivered = route === "review" ? 95 : 52;
  if (status === "sent" || status === "archived") {
    add(delivered, "activity:build_einvoice");
    add(delivered + 3, "activity:send_to_asp");
  }
  if (status === "archived") add(delivered + 8, "activity:archive");
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
  const fraud = trail.find((e) => e.action === "activity:fraud");
  const top = invoice.flags[0];
  const others = invoice.flags.length - 1;
  switch (invoice.summary.status) {
    case "blocked":
      return (
        `${number} was blocked [audit:${route?.id}]. ` +
        (top && fraud ? `${top.explanation} [audit:${fraud.id}]` : "") +
        (others > 0 ? ` It also tripped ${others} other check${others > 1 ? "s" : ""}.` : "") +
        " A blocked invoice can only be rejected; nothing will be paid."
      );
    case "awaiting_review":
      return (
        `${number} is waiting for a reviewer because of the ` +
        `${reviewRule(invoice).replace(/_/g, " ")} check [audit:${route?.id}].` +
        (top && fraud ? ` ${top.explanation} [audit:${fraud.id}]` : "")
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
