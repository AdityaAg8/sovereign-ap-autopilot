/** Layout for every authenticated screen (inbox, review queue, invoice detail, vendors): mounts
 * the app shell (sidebar, topbar, command palette, shortcut help) around the route's page, plus
 * the tooltip and toast providers only these screens use (kept off /login's bundle). */
import { AppShell } from "@/components/shell/app-shell";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

export default function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider delayDuration={400}>
      <AppShell>{children}</AppShell>
      <Toaster position="bottom-right" />
    </TooltipProvider>
  );
}
