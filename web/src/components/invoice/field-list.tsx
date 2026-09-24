/** Scrollable list of extracted fields backing the invoice-detail review pane (PRD §10 M5.4).
 * Header fields come first, then the ADR-0012 `line_items[N].*` rows grouped under "Line N".
 * Exposes `focusFirstNeedsReview` (the "f" shortcut) and `focusField` (a rejected approval moves
 * focus to the first field it names) via ref. */

"use client";

import { forwardRef, useImperativeHandle, useMemo, useState } from "react";
import { toast } from "sonner";
import { FieldRow } from "@/components/invoice/field-row";
import type { FieldEdit, FieldValue } from "@/lib/api/types";
import { fieldLabel, lineFieldName, parseLineField, vatRateGuidance } from "@/lib/invoice-fields";

export interface FieldListHandle {
  focusFirstNeedsReview: () => void;
  /** Selects and focuses the row for `name`; false when no such row is rendered. */
  focusField: (name: string) => boolean;
}

function focusRow(name: string): boolean {
  const row = document.querySelector<HTMLElement>(`[data-field="${CSS.escape(name)}"]`);
  if (!row) return false;
  row.focus();
  row.scrollIntoView({ block: "nearest" });
  return true;
}

export const FieldList = forwardRef<
  FieldListHandle,
  {
    fields: readonly FieldValue[];
    activeField: string | null;
    /** Field names the last rejected approval reported as unresolved. */
    unresolved?: readonly string[];
    onSelectField: (name: string) => void;
    onEditField: (edit: FieldEdit) => Promise<void>;
  }
>(function FieldList({ fields, activeField, unresolved = [], onSelectField, onEditField }, ref) {
  const [editingField, setEditingField] = useState<string | null>(null);

  const { header, lines } = useMemo(() => {
    const head: FieldValue[] = [];
    const byLine = new Map<number, FieldValue[]>();
    for (const f of fields) {
      const line = parseLineField(f.name);
      if (!line) {
        head.push(f);
        continue;
      }
      const rows = byLine.get(line.index) ?? [];
      rows.push(f);
      byLine.set(line.index, rows);
    }
    const grouped = [...byLine.entries()]
      .sort(([a], [b]) => a - b)
      .map(([index, rows]) => ({
        index,
        // Category before rate: the rate's guidance depends on the category above it.
        rows: rows.sort((a, b) => a.name.localeCompare(b.name)),
      }));
    return { header: head, lines: grouped };
  }, [fields]);

  const valueOf = (name: string) => fields.find((f) => f.name === name)?.raw ?? null;

  useImperativeHandle(ref, () => ({
    focusFirstNeedsReview: () => {
      const target = fields.find((f) => !f.accepted);
      if (!target) {
        toast.success("Every field is auto-accepted — nothing left to fix.");
        return;
      }
      onSelectField(target.name);
      setEditingField(target.name);
    },
    focusField: (name: string) => {
      if (!fields.some((f) => f.name === name)) return false;
      setEditingField(null);
      onSelectField(name);
      return focusRow(name);
    },
  }));

  function renderRow(field: FieldValue) {
    const line = parseLineField(field.name);
    const hint =
      line?.attr === "vat_rate"
        ? vatRateGuidance(valueOf(lineFieldName(line.index, "vat_category")))
        : undefined;
    return (
      <FieldRow
        key={field.name}
        field={field}
        active={field.name === activeField}
        editing={field.name === editingField}
        unresolved={unresolved.includes(field.name)}
        hint={hint}
        onSelect={() => onSelectField(field.name)}
        onStartEdit={() => {
          onSelectField(field.name);
          setEditingField(field.name);
        }}
        onCancelEdit={() => {
          setEditingField(null);
          focusRow(field.name);
        }}
        onSave={async (value) => {
          await onEditField({ field: field.name, value, reason: "Reviewer correction" });
          setEditingField(null);
          const where = line ? `Line ${line.index} ` : "";
          toast.success(`${where}${fieldLabel(field.name)} updated`);
          // Keep keyboard users where they were once the editor closes.
          requestAnimationFrame(() => focusRow(field.name));
        }}
      />
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      {header.map(renderRow)}
      {lines.map(({ index, rows }) => (
        <section key={index} aria-labelledby={`line-${index}-heading`}>
          <h2
            id={`line-${index}-heading`}
            className="bg-muted/40 text-muted-foreground border-border/70 border-b px-4 py-1.5 text-xs font-medium"
          >
            Line {index}
          </h2>
          {rows.map(renderRow)}
        </section>
      ))}
    </div>
  );
});
