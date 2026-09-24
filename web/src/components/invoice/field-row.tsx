/** One extracted field: label, grounded value, confidence, source and the inline edit affordance
 * (PRD §10 M5.4 "field list with confidence + source + edit affordance"). Selecting or focusing a
 * row highlights its box on the document pane (web/DESIGN.md's synchronized-grounding signature).
 * ADR-0012 classified fields (document type, currency, line VAT category/rate) get constrained
 * editors: native selects for the code lists, a guided numeric input for the rate. */

"use client";

import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { Check, Pencil, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfidenceMeter } from "@/components/inbox/confidence-meter";
import type { FieldValue } from "@/lib/api/types";
import {
  CURRENCY_FIELD,
  fieldLabel,
  INVOICE_TYPE_FIELD,
  INVOICE_TYPE_OPTIONS,
  isMissingCurrency,
  isMoneyField,
  isMonoField,
  optionLabel,
  parseLineField,
  VAT_CATEGORY_OPTIONS,
} from "@/lib/invoice-fields";
import { formatGroupedAmount } from "@/lib/format";
import { cn } from "@/lib/utils";

const SOURCE_LABEL: Record<FieldValue["source"], string> = {
  extractor_a: "Extractor A",
  extractor_b: "Extractor B",
  arbiter: "Arbiter",
  reviewer: "Reviewer edit",
};

type Editor =
  | { kind: "select"; options: readonly { value: string; label: string }[] }
  | { kind: "rate" }
  | { kind: "currency" }
  | { kind: "text" };

function editorFor(name: string): Editor {
  if (name === INVOICE_TYPE_FIELD) return { kind: "select", options: INVOICE_TYPE_OPTIONS };
  if (name === CURRENCY_FIELD) return { kind: "currency" };
  const line = parseLineField(name);
  if (line?.attr === "vat_category") return { kind: "select", options: VAT_CATEGORY_OPTIONS };
  if (line?.attr === "vat_rate") return { kind: "rate" };
  return { kind: "text" };
}

/** The value as a reviewer reads it, or null when the field is effectively unresolved. */
function displayValue(field: FieldValue, editor: Editor, money: boolean): string | null {
  const raw = field.raw?.trim() ? field.raw : null;
  if (editor.kind === "currency" && isMissingCurrency(raw)) return null;
  if (raw === null) return null;
  if (editor.kind === "select") return optionLabel(editor.options, raw) ?? raw;
  if (editor.kind === "rate") return `${raw}%`;
  return money ? formatGroupedAmount(raw) : raw;
}

export function FieldRow({
  field,
  active,
  editing,
  unresolved = false,
  hint,
  onSelect,
  onStartEdit,
  onCancelEdit,
  onSave,
}: {
  field: FieldValue;
  active: boolean;
  editing: boolean;
  /** Named by the last rejected approval (422) and not yet corrected. */
  unresolved?: boolean;
  /** Guidance shown under the value (the allowed VAT rates for this line's category). */
  hint?: string;
  onSelect: () => void;
  onStartEdit: () => void;
  onCancelEdit: () => void;
  onSave: (value: string) => Promise<void>;
}) {
  const editor = editorFor(field.name);
  const initialDraft =
    editor.kind === "currency" && isMissingCurrency(field.raw) ? "" : (field.raw ?? "");
  const [draft, setDraft] = useState(initialDraft);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const selectRef = useRef<HTMLSelectElement>(null);
  const hintId = useId();
  const mono = isMonoField(field.name);
  const money = isMoneyField(field.name);
  const shown = displayValue(field, editor, money);
  const label = fieldLabel(field.name);
  const isSelect = editor.kind === "select";

  useEffect(() => {
    if (editing) {
      setDraft(initialDraft);
      if (isSelect) {
        selectRef.current?.focus();
      } else {
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    }
  }, [editing, initialDraft, isSelect]);

  async function commit() {
    setSaving(true);
    try {
      await onSave(draft.trim());
    } finally {
      setSaving(false);
    }
  }

  function onKeyDown(event: KeyboardEvent) {
    if (editing) {
      // The input and its wrapping row div share this handler, and the input is a descendant of
      // that div — without stopPropagation, Enter/Escape on the input bubbles straight into the
      // row's own listener and re-runs the same branch a second time (a real bug: it used to
      // save the field, and post the "field updated" toast, twice per keystroke).
      event.stopPropagation();
      if (event.key === "Enter") {
        event.preventDefault();
        void commit();
      } else if (event.key === "Escape") {
        event.preventDefault();
        onCancelEdit();
      }
      return;
    }
    if (event.key === "Enter" || event.key === "e") {
      event.preventDefault();
      onStartEdit();
    }
  }

  const describedBy = hint ? hintId : undefined;
  // The row takes its name from its visible content (WCAG 2.5.3 label in name: an aria-label
  // that omitted the status and source text failed axe's label-content-name-mismatch); the
  // Required marker and the rate guidance are also exposed as descriptions.
  const rowDescribedBy = [unresolved ? `${hintId}-req` : null, hint ? hintId : null]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={onKeyDown}
      data-field={field.name}
      aria-describedby={rowDescribedBy || undefined}
      className={cn(
        "border-border/70 group flex flex-col gap-1.5 border-b px-4 py-2.5 outline-none",
        "focus-visible:outline-ring focus-visible:outline-2 focus-visible:-outline-offset-2",
        active && "bg-accent/10 outline-ring outline-1 -outline-offset-1",
        unresolved && "border-s-destructive border-s-2",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-muted-foreground text-xs">{label}</span>
        <div className="flex items-center gap-2">
          {unresolved ? (
            <span id={`${hintId}-req`} className="text-destructive text-[0.6875rem] font-medium">
              Required
            </span>
          ) : (
            !field.accepted && (
              <span className="text-warning text-[0.6875rem] font-medium">Needs review</span>
            )
          )}
          <ConfidenceMeter value={field.confidence} />
        </div>
      </div>
      {editing ? (
        <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
          {editor.kind === "select" ? (
            <select
              ref={selectRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={saving}
              aria-label={`Edit ${label}`}
              aria-describedby={describedBy}
              className="border-input bg-background focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 h-7 w-full min-w-0 rounded-lg border px-2 text-sm outline-none focus-visible:ring-3 disabled:opacity-50"
            >
              <option value="" disabled>
                Choose…
              </option>
              {editor.options.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.value} · {o.label}
                </option>
              ))}
            </select>
          ) : (
            <Input
              ref={inputRef}
              value={draft}
              onChange={(e) =>
                setDraft(editor.kind === "currency" ? e.target.value.toUpperCase() : e.target.value)
              }
              onKeyDown={onKeyDown}
              disabled={saving}
              aria-label={`Edit ${label}`}
              aria-describedby={describedBy}
              inputMode={editor.kind === "rate" ? "decimal" : undefined}
              maxLength={editor.kind === "currency" ? 3 : undefined}
              placeholder={editor.kind === "currency" ? "AED" : undefined}
              autoComplete="off"
              spellCheck={false}
              className={cn("h-7 text-sm", mono && "tabular-mono")}
            />
          )}
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Save"
            disabled={saving}
            onClick={() => void commit()}
          >
            <Check />
          </Button>
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Cancel"
            disabled={saving}
            onClick={onCancelEdit}
          >
            <X />
          </Button>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2">
          {shown === null && editor.kind !== "text" ? (
            <span className="text-destructive text-sm font-medium">Missing</span>
          ) : (
            <span className={cn("truncate text-sm", mono && "tabular-mono")}>{shown ?? "—"}</span>
          )}
          <button
            type="button"
            aria-label={`Edit ${label}`}
            onClick={(e) => {
              e.stopPropagation();
              onStartEdit();
            }}
            className="text-muted-foreground hover:text-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
          >
            <Pencil className="size-3.5" />
          </button>
        </div>
      )}
      {hint && (
        <span id={hintId} className="text-muted-foreground text-[0.6875rem]">
          {hint}
        </span>
      )}
      <span className="text-muted-foreground text-[0.6875rem]">{SOURCE_LABEL[field.source]}</span>
    </div>
  );
}
