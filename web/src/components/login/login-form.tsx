/** Login form for the dev-fixture auth flow (PRD §1 roles). Any password authenticates a listed
 * fixture email; in fixture mode the three role accounts are shown so a reviewer/approver/admin
 * can sign in without hunting for credentials in a README. In http mode those `@sovereign.local`
 * accounts don't exist on the real backend, so we point at `data/seed_credentials.json` instead
 * of listing them (and never render a password, generated or otherwise). */

"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/lib/auth/auth-context";
import { FIXTURE_USERS } from "@/lib/fixtures/users";
import { getApiMode } from "@/lib/api/client";
import { ApiError } from "@/lib/api/contract";

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
        />
      </div>
      {error && (
        <p id="login-error" role="alert" className="text-destructive text-sm">
          {error}
        </p>
      )}
      <Button type="submit" className="w-full" disabled={pending}>
        {pending && <Loader2 className="animate-spin" aria-hidden />}
        Sign in
      </Button>
      {fixtureMode ? (
        <div className="border-border space-y-2 border-t pt-4">
          <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
            Demo accounts
          </p>
          <ul className="space-y-1">
            {FIXTURE_USERS.map((u) => (
              <li key={u.id}>
                <button
                  type="button"
                  onClick={() => {
                    setEmail(u.email);
                    setPassword("demo");
                  }}
                  className="text-muted-foreground hover:text-foreground focus-visible:text-foreground text-left text-xs underline-offset-2 hover:underline"
                >
                  <span className="tabular-mono">{u.email}</span> — {u.roles.join(", ")}
                </button>
              </li>
            ))}
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
