/** Login form for the dev-fixture auth flow. Any password authenticates a listed
 * fixture email; in fixture mode the three role accounts are shown so a reviewer/approver/admin
 * can sign in without hunting for credentials in a README. In http mode those `@sovereign.local`
 * accounts don't exist on the real backend, so we point at `data/seed_credentials.json` instead
 * of listing them (and never render a password, generated or otherwise). */

"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/lib/auth/auth-context";
import { FIXTURE_USERS } from "@/lib/fixtures/users";
import { getApiMode } from "@/lib/api/client";
import { ApiError } from "@/lib/api/contract";
import { cn } from "@/lib/utils";

/** Only a typed 401/404 from the adapter means "wrong account"; anything else (a chunk that
 * failed to load, a network error, a 5xx) means the server could not be reached. */
function isRejectedCredentials(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 401 || err.status === 404);
}

export function LoginForm() {
  const { login } = useAuth();
  const fixtureMode = getApiMode() === "fixture";
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  // Until React hydrates, a click would submit the native form. The button stays disabled until
  // then, and the form is POST, so credentials never land in the URL or browser history.
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setPending(true);
    try {
      await login(email, password);
      router.replace("/inbox");
    } catch (err: unknown) {
      if (!isRejectedCredentials(err)) {
        setError("Couldn't reach the server, try again.");
      } else {
        setError(
          fixtureMode
            ? "No account matches that email. Try one of the demo accounts below."
            : "Those credentials were not accepted. Check data/seed_credentials.json.",
        );
      }
    } finally {
      setPending(false);
    }
  }

  return (
    <form
      method="post"
      onSubmit={onSubmit}
      noValidate
      className="w-full max-w-sm space-y-5"
      aria-describedby={error ? "login-error" : undefined}
    >
      <div className="space-y-1.5">
        <Label htmlFor="email">Email</Label>
        <Input
          id="email"
          name="email"
          type="email"
          autoComplete="username"
          autoFocus
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          aria-invalid={error ? true : undefined}
          className="bg-card h-9"
        />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="password">Password</Label>
        <Input
          id="password"
          name="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          aria-invalid={error ? true : undefined}
          className="bg-card h-9"
        />
      </div>
      {error && (
        <p id="login-error" role="alert" className="text-destructive text-sm">
          {error}
        </p>
      )}
      <Button type="submit" size="lg" className="w-full" disabled={pending || !hydrated}>
        {pending && <Loader2 className="animate-spin" aria-hidden />}
        Sign in
      </Button>
      {fixtureMode ? (
        <div className="border-border border-t pt-5">
          <p id="demo-accounts" className="text-sm font-medium">
            Demo accounts
          </p>
          <p className="text-muted-foreground mt-0.5 text-xs">Pick one to fill in the form.</p>
          <ul aria-labelledby="demo-accounts" className="panel mt-3 divide-y overflow-hidden">
            {FIXTURE_USERS.map((u) => {
              const picked = email === u.email;
              return (
                <li key={u.id}>
                  <button
                    type="button"
                    aria-pressed={picked}
                    onClick={() => {
                      setEmail(u.email);
                      setPassword("demo");
                    }}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left transition-colors duration-150 focus-visible:-outline-offset-2",
                      picked ? "bg-accent/[0.08]" : "hover:bg-muted/60",
                    )}
                  >
                    <span className="tabular-mono min-w-0 truncate text-xs">{u.email}</span>
                    <span className="text-muted-foreground shrink-0 text-2xs capitalize">
                      {u.roles.join(", ")}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : (
        <p className="border-border text-muted-foreground border-t pt-4 text-xs">
          Sign-in details for the seeded accounts are in{" "}
          <span className="tabular-mono">data/seed_credentials.json</span> (written by{" "}
          <span className="tabular-mono">make seed</span>).
        </p>
      )}
    </form>
  );
}
