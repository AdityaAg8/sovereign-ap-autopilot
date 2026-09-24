/** pdf.js worker entry (the `workerSrc` pdf-canvas.tsx sets): installs the ES2026 polyfills the
 * modern pdf.js 6 build needs on browsers that lack them, then runs the vendored worker. Module
 * imports evaluate in order, so the polyfills are in place before any worker code runs. */
import "./es-polyfills.mjs";
import "./pdf.worker.min.mjs";
