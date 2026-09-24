/** Login screen: brand mark, one sentence of context and the sign-in form (PRD §10 M5.4). Kept
 * minimal — no imagery, no client JS beyond the form — to hold Lighthouse performance at ≥95. */
import { FileStack } from "lucide-react";
import { LoginForm } from "@/components/login/login-form";

export default function LoginPage() {
  return (
    <main className="flex min-h-dvh items-center justify-center px-4 py-12">
      <div className="flex w-full max-w-sm flex-col items-center gap-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <FileStack className="text-accent size-8" aria-hidden />
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Sovereign AP</h1>
            <p className="text-muted-foreground mt-1 text-sm">
              Review, validate and approve UAE e-invoices — grounded to the source page, every time.
            </p>
          </div>
        </div>
        <LoginForm />
      </div>
    </main>
  );
}
