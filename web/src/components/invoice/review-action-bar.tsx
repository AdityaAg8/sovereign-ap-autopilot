/** Approve / Fix / Reject action bar for the review queue, bound to a/f/r keyboard shortcuts (PRD
 * §10 M5.4). Approve is disabled — with the reason stated, never just greyed out silently — when
 * the signed-in role can't clear this invoice's amount (PRD §1 roles; the real authorization
 * boundary is apx/api's policy engine once M5.3/M6.5 land, this is UX gating only). A refused
 * approval (422: ADR-0012 fields still unresolved) is announced inline through a live region. */

"use client";

import { Check, Pencil, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { classifiedLabel, fieldLabel } from "@/lib/invoice-fields";
import { useRegisterShortcuts } from "@/lib/shortcuts/registry";

const SHORTCUTS = [
  { combo: "a", description: "Approve" },
  { combo: "f", description: "Jump to fix the first field needing review" },
  { combo: "r", description: "Reject" },
] as const;

export interface ApproveError {
  /** The server's `detail`, shown verbatim when it names no fields. */
  detail: string;
  /** Unresolved field names parsed from `detail`. */
  fields: readonly string[];
}

export function ReviewActionBar({
  disabled,
  approveBlockedReason,
  pending,
  approveError,
  onApprove,
  onFix,
  onReject,
}: {
  disabled: boolean;
  approveBlockedReason: string | null;
  pending: boolean;
  approveError: ApproveError | null;
  onApprove: () => void;
  onFix: () => void;
  onReject: () => void;
}) {
  useRegisterShortcuts("Review actions", SHORTCUTS);
  useHotkeys([
    {
      combo: "a",
      description: "Approve",
      handler: () => !disabled && !approveBlockedReason && onApprove(),
    },
    { combo: "f", description: "Fix", handler: () => !disabled && onFix() },
    { combo: "r", description: "Reject", handler: () => !disabled && onReject() },
  ]);

  const approveButton = (
    <Button
      variant="success"
      disabled={disabled || pending || Boolean(approveBlockedReason)}
      onClick={onApprove}
      aria-describedby={approveError ? "approve-error" : undefined}
      className="gap-1.5"
    >
      <Check className="size-4" /> Approve <kbd className="ms-1 opacity-60">A</kbd>
    </Button>
  );

  const names = approveError?.fields.map((f) => classifiedLabel(f, true) ?? fieldLabel(f)) ?? [];

  return (
    <div className="border-border bg-background border-t">
      {/* Always mounted so screen readers pick up the message when it is inserted. */}
      <div aria-live="assertive" aria-atomic="true">
        {approveError && (
          <p
            id="approve-error"
            className="text-destructive border-destructive/30 bg-destructive/5 border-b px-4 py-2 text-sm"
          >
            <span className="font-medium">Approval refused.</span>{" "}
            {names.length > 0
              ? `Correct ${names.length === 1 ? "this field" : "these fields"} first: ${names.join(", ")}.`
              : approveError.detail}
          </p>
        )}
      </div>
      <div className="flex items-center gap-2 px-4 py-3">
        {approveBlockedReason ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span>{approveButton}</span>
            </TooltipTrigger>
            <TooltipContent>{approveBlockedReason}</TooltipContent>
          </Tooltip>
        ) : (
          approveButton
        )}
        <Button
          variant="outline"
          disabled={disabled || pending}
          onClick={onFix}
          className="gap-1.5"
        >
          <Pencil className="size-4" /> Fix <kbd className="ms-1 opacity-60">F</kbd>
        </Button>
        <Button
          variant="destructive"
          disabled={disabled || pending}
          onClick={onReject}
          className="ms-auto gap-1.5"
        >
          <X className="size-4" /> Reject <kbd className="ms-1 opacity-60">R</kbd>
        </Button>
      </div>
    </div>
  );
}
