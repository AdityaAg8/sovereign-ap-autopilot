/** Route-group layout for /vendors and /vendors/[id]: loads IBM Plex Sans Arabic only here,
 * since the vendor list and vendor detail screens are the only place this app renders bilingual
 * (`lang="ar"`) vendor names (web/DESIGN.md). Every other route inherits just the root layout's
 * Latin/mono fonts, so /login and /inbox never pay to download a font family they never
 * display (Lighthouse performance budget, .claude/rules/design.md). */
import { IBM_Plex_Sans_Arabic } from "next/font/google";

const plexSansArabic = IBM_Plex_Sans_Arabic({
  variable: "--font-plex-sans-arabic",
  subsets: ["arabic"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export default function VendorsLayout({ children }: { children: React.ReactNode }) {
  return <div className={plexSansArabic.variable}>{children}</div>;
}
