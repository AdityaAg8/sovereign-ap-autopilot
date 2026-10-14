/** The product mark: an auditor's tick over the double rule accountants draw under a final total,
 * on a teal tile ("checked, and it adds up"). Inline SVG drawn from theme tokens, shared by the
 * sidebar and the login screen. */
import { cn } from "@/lib/utils";

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden className={cn("size-7 shrink-0", className)}>
      <rect width="32" height="32" rx="7" className="fill-sidebar-primary" />
      <g
        className="stroke-sidebar-primary-foreground"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      >
        <path d="M10 13.5l3.6 3.6L22 8.8" strokeWidth="2.6" />
        <path d="M9 21.5h14M9 25h14" strokeWidth="1.7" />
      </g>
    </svg>
  );
}
