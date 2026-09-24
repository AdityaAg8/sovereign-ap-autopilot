/** The real (Radix) select behind an inbox filter, loaded on the first interaction by
 * `filter-select.tsx`. Opens immediately when the stand-in trigger was already activated; a
 * focused stand-in is never swapped (see lib/deferred-control.ts). */

"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { FilterSelectProps } from "@/components/inbox/filter-select";

export function FilterSelectMenu({
  value,
  onValueChange,
  options,
  ariaLabel,
  className,
  placeholder,
  defaultOpen,
}: FilterSelectProps & { defaultOpen: boolean }) {
  return (
    <Select value={value} onValueChange={onValueChange} defaultOpen={defaultOpen}>
      <SelectTrigger className={className} aria-label={ariaLabel}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value} className={option.className}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
