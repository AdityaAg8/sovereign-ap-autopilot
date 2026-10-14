/** Root layout for the Sovereign AP Autopilot review UI: loads the Ledger direction's Latin/mono
 * fonts (IBM Plex Sans / Mono, web/DESIGN.md), theming and the auth session provider. The
 * paired Arabic face (IBM Plex Sans Arabic) is *not* loaded here: it only ever renders inside
 * the vendor screens' bilingual names (the only `lang="ar"` text in this app), so it is loaded
 * by `(app)/vendors/layout.tsx` instead — every other route, including /login and /inbox, never
 * downloads a font family it never displays (Lighthouse performance
 * budget). */
import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { AuthProvider } from "@/lib/auth/auth-context";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
  // Not preloaded: mono only styles figures/IDs, never the LCP text, so its three files should
  // not compete with the stylesheet and Plex Sans on the critical path. It still loads (and
  // swaps in) as soon as a rendered element uses it.
  preload: false,
});

export const metadata: Metadata = {
  title: "Sovereign AP Autopilot",
  description: "Local-only AP agent: review queue, invoice detail and vendor oversight.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fdfdfc" },
    { media: "(prefers-color-scheme: dark)", color: "#1c1f27" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${plexSans.variable} ${plexMono.variable}`}
    >
      {/* Font CSS variables are set here, not on <body>, so globals.css's `html { @apply
          font-sans }` rule (an ancestor of body) can actually see them — a variable defined on a
          descendant is invisible to its ancestors. */}
      <body className="antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {/* The toast viewport lives in (app)/layout.tsx and the tooltip provider beside its only
              tooltip: only signed-in screens use them, so /login does not ship Radix Popper or
              sonner on its critical path. */}
          <AuthProvider>{children}</AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
