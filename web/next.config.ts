/** Next.js build configuration for the review UI; extend here as pages need rewrites, headers or
 * images config.
 *
 * Tried and reverted: `experimental.inlineCss` (inlines each page's CSS as `<style>` instead of
 * a render-blocking `<link>`, aimed at the ~400ms LCP cost `render-blocking-insight` attributes
 * to Tailwind's stylesheet on /login and /inbox). Under this Turbopack build it rewrites
 * next/font's relative `@font-face` `url(../media/*.woff2)` incorrectly once the CSS is moved
 * into the HTML document, 404-ing every font file (visible in Lighthouse's `errors-in-console`,
 * which also dropped best-practices from 100) without even improving the LCP score. Revisit once
 * this experimental feature's interaction with next/font under Turbopack is fixed upstream. */
import type { NextConfig } from "next";

const nextConfig: NextConfig = {/* config options here */};

export default nextConfig;
