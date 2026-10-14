/** Next.js build configuration for the review UI: sends a Content-Security-Policy and the standard
 * security headers on every route, sized for the pdf.js workers served
 * from /pdfjs and the apx API's REST + SSE origin.
 *
 * Tried and reverted: `experimental.inlineCss` (inlines each page's CSS as `<style>` instead of
 * a render-blocking `<link>`, aimed at the ~400ms LCP cost `render-blocking-insight` attributes
 * to Tailwind's stylesheet on /login and /inbox). Under this Turbopack build it rewrites
 * next/font's relative `@font-face` `url(../media/*.woff2)` incorrectly once the CSS is moved
 * into the HTML document, 404-ing every font file (visible in Lighthouse's `errors-in-console`,
 * which also dropped best-practices from 100) without even improving the LCP score. Revisit once
 * this experimental feature's interaction with next/font under Turbopack is fixed upstream. */
import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV !== "production";

/** Origins the browser may call for REST, the SSE feeds and the invoice PDF (fetched by the
 * pdf.js parser worker). Mirrors `BASE_URL` in src/lib/api/http-adapter.ts; the default API is
 * reachable as both localhost and 127.0.0.1, so both are allowed. */
function apiOrigins(): string[] {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (configured) return [new URL(configured).origin];
  return ["http://localhost:8000", "http://127.0.0.1:8000"];
}

/** Static policy (no per-request nonce): the App Router's inline hydration payloads and
 * next-themes' pre-paint script need 'unsafe-inline' for scripts, and React style props need it
 * for styles. What the policy still buys: no third-party script, frame, object or form target,
 * no framing of the review UI, and network access limited to this origin plus the apx API.
 * 'wasm-unsafe-eval' lets pdf.js instantiate its WebAssembly image decoders inside the workers
 * (workers get this same header on /pdfjs/* and /_next/static/*); dev adds 'unsafe-eval' for
 * React Refresh. */
function contentSecurityPolicy(): string {
  const scriptSrc = ["'self'", "'unsafe-inline'", "'wasm-unsafe-eval'"];
  if (isDev) scriptSrc.push("'unsafe-eval'");
  const directives: Record<string, string[]> = {
    "default-src": ["'self'"],
    "script-src": scriptSrc,
    "style-src": ["'self'", "'unsafe-inline'"],
    "img-src": ["'self'", "data:", "blob:"],
    "font-src": ["'self'", "data:"],
    "connect-src": ["'self'", ...apiOrigins()],
    "worker-src": ["'self'", "blob:"],
    "object-src": ["'none'"],
    "frame-src": ["'none'"],
    "frame-ancestors": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
  };
  return Object.entries(directives)
    .map(([name, values]) => `${name} ${values.join(" ")}`)
    .join("; ");
}

const securityHeaders = [
  { key: "Content-Security-Policy", value: contentSecurityPolicy() },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "no-referrer" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  },
];

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
