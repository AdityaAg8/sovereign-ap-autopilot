/** One inbox filter picker (status or route). Renders a stand-in that looks and reads exactly like
 * the Radix select trigger until the real select (filter-select-menu.tsx) loads on the first
 * interaction, so Radix Select and its positioning code stay off the inbox's critical path. */

"use client";

import { useId } from "react";
import { ChevronDownIcon } from "lucide-react";
import { selectTriggerClass, selectTriggerIconClass } from "@/components/ui/select-trigger-class";
import { useDeferredControl } from "@/lib/deferred-control";
import { cn } from "@/lib/utils";

export interface FilterSelectOption {
  value: string;
  label: string;
  className?: string;
}

export interface FilterSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  options: readonly FilterSelectOption[];
  ariaLabel: string;
  className?: string;
  placeholder: string;
}

const loadMenu = () => import("@/components/inbox/filter-select-menu");

export function FilterSelect(props: FilterSelectProps) {
  const { module, standInProps, openRequested } = useDeferredControl(loadMenu);
  // Like Radix, the closed trigger names the listbox it will control (not rendered until open).
  const listboxId = useId();

  if (module) {
    return <module.FilterSelectMenu {...props} defaultOpen={openRequested} />;
  }

  const selected = props.options.find((option) => option.value === props.value);
  // Mirrors the attributes Radix Select's closed trigger renders (role, state, size, slots), so
  // styling, accessible name and role are identical before and after the swap.
  return (
    <button
      type="button"
      role="combobox"
      aria-label={props.ariaLabel}
      aria-expanded={false}
      aria-controls={listboxId}
      aria-autocomplete="none"
      data-state="closed"
      data-slot="select-trigger"
      data-size="default"
      data-placeholder={selected ? undefined : ""}
      className={cn(selectTriggerClass, props.className)}
      {...standInProps}
    >
      <span data-slot="select-value" style={{ pointerEvents: "none" }}>
        {selected ? selected.label : props.placeholder}
      </span>
      <ChevronDownIcon aria-hidden className={selectTriggerIconClass} />
    </button>
  );
}
