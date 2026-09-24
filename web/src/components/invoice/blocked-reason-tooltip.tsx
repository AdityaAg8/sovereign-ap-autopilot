/** Hover hint that says why Approve is disabled (role or approval limit), the app's only tooltip.
 * Loaded on the first interaction by review-action-bar.tsx so Radix Tooltip and its positioning
 * code stay off the invoice page's critical path; the provider lives here with it. */

"use client";

import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

export function BlockedReasonTooltip({
  reason,
  children,
}: {
  reason: string;
  children: ReactNode;
}) {
  return (
    <TooltipProvider delayDuration={400}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span>{children}</span>
        </TooltipTrigger>
        <TooltipContent>{reason}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
