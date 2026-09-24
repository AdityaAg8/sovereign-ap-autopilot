/** Root route: sends signed-in operators to the inbox and everyone else to /login. Auth state is
 * client-only (dev fixture session), so this redirect can't happen at the server. */

"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/auth-context";

export default function Home() {
  const { status } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (status === "authenticated") router.replace("/inbox");
    if (status === "unauthenticated") router.replace("/login");
  }, [status, router]);

  return (
    <div className="text-muted-foreground flex h-dvh items-center justify-center text-sm">
      Loading Sovereign AP Autopilot…
    </div>
  );
}
