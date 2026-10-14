/** Root layout for the Sovereign AP Autopilot review UI: loads the UI face (Instrument Sans) and
 * the data face (Geist Mono) via next/font, theming and the auth session provider. The paired
 * Arabic face is loaded only by `(app)/vendors/layout.tsx`, the one place `lang="ar"` text renders,
 * so /login and /inbox never download a font family they never display. */
import type { Metadata, Viewport } from "next";
import { Geist_Mono, Instrument_Sans } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { AuthProvider } from "@/lib/auth/auth-context";
import "./globals.css";

const uiFace = Instrument_Sans({
  variable: "--font-ui",
  subsets: ["latin"],
  display: "swap",
});

const dataFace = Geist_Mono({
  variable: "--font-data",
  subsets: ["latin"],
  display: "swap",
  // Not preloaded: mono only styles figures and IDs, never the LCP text, so its file should not
  // compete with the stylesheet and the UI face on the critical path. It still loads (and swaps
  // in) as soon as a rendered element uses it.
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
    { media: "(prefers-color-scheme: light)", color: "#f6f9f9" },
    { media: "(prefers-color-scheme: dark)", color: "#12171b" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${uiFace.variable} ${dataFace.variable}`}>
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
