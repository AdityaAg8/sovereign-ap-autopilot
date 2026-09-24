/** Layout for every authenticated screen (inbox, review queue, invoice detail, vendors): mounts
 * the app shell (sidebar, topbar, command palette, shortcut help) around the route's page, plus
 * the toast viewport only these screens use (kept off /login's bundle, and mounted once idle).
 * The tooltip provider lives beside the app's only tooltip (blocked-reason-tooltip.tsx). */
import { AppShell } from "@/components/shell/app-shell";
import { LazyToaster } from "@/components/shell/lazy-toaster";

export default function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <AppShell>{children}</AppShell>
      <LazyToaster />
    </>
  );
}
