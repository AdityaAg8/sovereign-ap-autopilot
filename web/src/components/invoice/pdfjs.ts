/** pdf.js for the invoice viewer, with the ES2026 polyfills it needs on browsers that lack them
 * (public/pdfjs/es-polyfills.mjs, shared with the worker entry) evaluated first. Imported only
 * by the render worker (pdf-render-worker.ts), so neither ever runs on the main thread. */
import "../../../public/pdfjs/es-polyfills.mjs";

export { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
