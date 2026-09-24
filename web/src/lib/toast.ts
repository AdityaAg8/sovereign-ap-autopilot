/** Toast calls for the signed-in screens, with sonner imported on first use rather than bundled
 * into the invoice page's critical path. Same API subset the screens use (`toast.success`,
 * `toast.error`); the viewport is mounted by components/shell/lazy-toaster.tsx, and sonner
 * replays toasts created before it subscribes. */

type Sonner = typeof import("sonner");

let sonner: Promise<Sonner> | null = null;

function loadSonner(): Promise<Sonner> {
  if (!sonner) {
    const pending = import("sonner");
    sonner = pending;
    // A failed chunk load is not cached, so the next toast retries the import.
    pending.catch(() => {
      if (sonner === pending) sonner = null;
    });
  }
  return sonner;
}

function show(kind: "success" | "error", message: string): void {
  loadSonner().then(
    (m) => {
      m.toast[kind](message);
    },
    (err: unknown) => {
      // The action already happened; only its confirmation could not be shown. Say so.
      console.error(`Could not show the "${message}" notification:`, err);
    },
  );
}

export const toast = {
  success: (message: string) => show("success", message),
  error: (message: string) => show("error", message),
};
