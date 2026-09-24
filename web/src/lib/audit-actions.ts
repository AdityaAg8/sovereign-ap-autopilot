/** Human labels for audit_log actions as apx writes them (apx.agent.interfaces.AuditAction:
 * `activity:<step>`, `authorize:<cap>`, `decision.<action>`, `proposal.submit|accept`,
 * `payment_state:<...>`, `field_edit`, `agent.review_note*`, `mcp.read|refusal`), used by the invoice
 * Audit tab (PRD §10 M9.3). An action this table doesn't know yet still gets a readable label. */

/** Workflow steps, in pipeline order (apx.workflows.activities `activity.defn` names). */
const ACTIVITY_LABEL: Readonly<Record<string, string>> = {
  scan: "Scanned for hidden instructions",
  parse: "Document parsed",
  extract: "Fields extracted",
  trust_gate: "Confidence checked",
  build_invoice: "Invoice assembled",
  validate: "Validated against PINT AE",
  apply_review: "Reviewer corrections applied",
  fraud: "Fraud checks run",
  route: "Routed",
  build_einvoice: "E-invoice built",
  send_to_asp: "Sent to the ASP",
  archive: "Archived",
};

/** apx.policy.interfaces.Capability values. */
const CAPABILITY_LABEL: Readonly<Record<string, string>> = {
  edit_field: "edit a field",
  route_exception: "route an exception",
  approve_payment: "approve payment",
  approve_above_limit: "approve above the limit",
  change_bank_details: "change bank details",
  verify_bank_details: "verify bank details",
  manage_users: "manage users",
  execute_proposal: "execute a proposal",
};

/** apx.api decision actions (lib/api/types.ts `DecisionAction`). */
const DECISION_LABEL: Readonly<Record<string, string>> = {
  approve: "Approved",
  reject: "Rejected",
  send_to_review: "Sent back to review",
  execute_proposal: "Proposal executed",
};

/** apx.policy.writer payment-state transitions. */
const PAYMENT_STATE_LABEL: Readonly<Record<string, string>> = {
  approve: "Payment approval recorded",
  bank_change: "Bank details change recorded",
  verify_bank: "Bank details verified",
  execute_proposal: "Proposal execution recorded",
};

const EXACT_LABEL: Readonly<Record<string, string>> = {
  "proposal.submit": "Proposal drafted",
  "proposal.accept": "Proposal accepted",
  field_edit: "Field corrected",
  "agent.review_note": "Assistant review note",
  "agent.review_note_refused": "Review note withheld (not verifiable)",
  "agent.review_note_unavailable": "Review note unavailable (model offline)",
  "mcp.refusal": "Assistant request refused",
  "mcp.read": "Assistant read records",
};

function humanize(text: string): string {
  const words = text.replace(/[._:]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "";
}

/** The label an audit event is shown under. Never throws: an unknown action or step falls back
 * to a sentence-cased reading of the raw action, and an empty one to "Audit event". */
export function auditActionLabel(action: string): string {
  const exact = EXACT_LABEL[action];
  if (exact) return exact;
  const [, family, rest] = /^([a-z_]+)[.:](.+)$/.exec(action) ?? [];
  if (family !== undefined && rest !== undefined) {
    switch (family) {
      case "activity":
        return ACTIVITY_LABEL[rest] ?? `Workflow step: ${humanize(rest).toLowerCase()}`;
      case "authorize":
        return `Permission checked: ${CAPABILITY_LABEL[rest] ?? humanize(rest).toLowerCase()}`;
      case "decision":
        return DECISION_LABEL[rest] ?? `Decision: ${humanize(rest).toLowerCase()}`;
      case "payment_state":
        return PAYMENT_STATE_LABEL[rest] ?? `Payment state: ${humanize(rest).toLowerCase()}`;
      case "proposal":
        return `Proposal ${humanize(rest).toLowerCase()}`;
    }
  }
  return humanize(action) || "Audit event";
}
