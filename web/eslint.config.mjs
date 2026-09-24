/** ESLint flat config for the review UI: Next.js core-web-vitals and TypeScript rules, Prettier-compatible. */
import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript", "prettier"),
  {
    // public/pdfjs is vendored, pre-minified pdfjs-dist build output (see pdf-canvas.tsx's
    // provenance comment) — source-linting it is both meaningless and, for the minified worker,
    // slow enough to matter.
    ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "next-env.d.ts", "public/**"],
  },
  {
    // Adapter methods implement a shared interface (ApiClient) and intentionally leave some
    // parameters unused (e.g. the fixture adapter's `subscribeEvents`); underscore-prefixing
    // documents that instead of suppressing the check for every unused variable.
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
];

export default eslintConfig;
