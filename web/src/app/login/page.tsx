/** Login screen: a split layout with the product's one-line purpose and three plain facts on the
 * ink-teal panel, and the sign-in form on the canvas. No imagery and no client JS beyond the form,
 * to hold Lighthouse performance at 95 or above. */
import { BrandMark } from "@/components/shell/brand-mark";
import { LoginForm } from "@/components/login/login-form";

const FACTS = [
  "Every extracted value is tied to its box on the source page.",
  "Fraud and prompt-injection checks run before anything is paid.",
  "Runs on your own hardware, with no outbound calls.",
];

export default function LoginPage() {
  return (
    <div className="grid min-h-dvh lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <aside
        data-surface="spine"
        className="bg-sidebar text-sidebar-foreground hidden flex-col justify-between p-10 lg:flex xl:p-14"
      >
        <div className="flex items-center gap-2.5">
          <BrandMark />
          <span className="text-sidebar-accent-foreground text-sm font-semibold tracking-[-0.01em]">
            Sovereign AP
          </span>
        </div>
        <div className="max-w-md">
          <p className="text-sidebar-accent-foreground text-[1.75rem] leading-[1.15] font-semibold tracking-[-0.025em] text-balance">
            Supplier invoices, checked line by line and ready to pay.
          </p>
          <p className="text-sidebar-muted mt-4 text-sm leading-relaxed text-pretty">
            Reads English and Arabic invoices, builds UAE PINT AE e-invoices and puts anything
            uncertain in front of a person.
          </p>
          <ul className="mt-8 space-y-3">
            {FACTS.map((fact) => (
              <li key={fact} className="flex gap-3 text-sm">
                <span aria-hidden className="bg-sidebar-primary mt-2 h-px w-4 shrink-0" />
                <span>{fact}</span>
              </li>
            ))}
          </ul>
        </div>
        <p className="text-sidebar-muted text-2xs">Accounts payable for UAE businesses</p>
      </aside>
      <main className="flex items-center justify-center px-4 py-12 sm:px-8">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-2.5 lg:hidden">
            <BrandMark />
            <span className="text-sm font-semibold tracking-[-0.01em]">Sovereign AP</span>
          </div>
          <h1 className="text-xl font-semibold tracking-[-0.02em]">Sign in</h1>
          <p className="text-muted-foreground mt-1 mb-7 text-sm">
            Review, fix and approve the invoices waiting on you.
          </p>
          <LoginForm />
        </div>
      </main>
    </div>
  );
}
