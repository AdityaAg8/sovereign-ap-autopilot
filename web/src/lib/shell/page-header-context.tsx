/** Lets a page set the shell topbar's title and action buttons without the App Router layout
 * needing per-route props (PRD §10 M5.4 app shell). */

"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

interface PageHeader {
  title: string;
  actions?: ReactNode;
}

interface PageHeaderContextValue {
  header: PageHeader;
  setHeader: (header: PageHeader) => void;
}

const DEFAULT_HEADER: PageHeader = { title: "Sovereign AP Autopilot" };

const PageHeaderContext = createContext<PageHeaderContextValue | null>(null);

export function PageHeaderProvider({ children }: { children: ReactNode }) {
  const [header, setHeader] = useState<PageHeader>(DEFAULT_HEADER);
  const value = useMemo(() => ({ header, setHeader }), [header]);
  return <PageHeaderContext.Provider value={value}>{children}</PageHeaderContext.Provider>;
}

export function usePageHeader(): PageHeader {
  const ctx = useContext(PageHeaderContext);
  if (!ctx) throw new Error("usePageHeader must be used within a PageHeaderProvider");
  return ctx.header;
}

/** Call from a page's top level with a stable object (memoize `actions` if it's not a primitive). */
export function useSetPageHeader(header: PageHeader): void {
  const ctx = useContext(PageHeaderContext);
  if (!ctx) throw new Error("useSetPageHeader must be used within a PageHeaderProvider");
  const { setHeader } = ctx;
  useEffect(() => {
    setHeader(header);
    return () => setHeader(DEFAULT_HEADER);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [header.title, header.actions, setHeader]);
}
