/** Central registry pages use to publish their keyboard shortcuts so the `?` help overlay
 * always reflects what's actually bound on screen, instead of a hand-maintained list
 * that drifts from the code. */

"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export interface ShortcutEntry {
  combo: string;
  description: string;
}

export interface ShortcutGroup {
  scope: string;
  entries: readonly ShortcutEntry[];
}

interface RegistryValue {
  groups: readonly ShortcutGroup[];
  register: (scope: string, entries: readonly ShortcutEntry[]) => () => void;
}

const RegistryContext = createContext<RegistryValue | null>(null);

export const GLOBAL_SHORTCUTS: readonly ShortcutEntry[] = [
  { combo: "Mod+K", description: "Open the command palette" },
  { combo: "?", description: "Show this shortcut reference" },
  { combo: "G then I", description: "Go to inbox" },
  { combo: "G then R", description: "Go to review queue" },
  { combo: "G then V", description: "Go to vendors" },
  { combo: "G then O", description: "Go to operations" },
  { combo: "C", description: "Ask the assistant" },
];

export function ShortcutsProvider({ children }: { children: ReactNode }) {
  const [scopes, setScopes] = useState<Map<string, readonly ShortcutEntry[]>>(new Map());

  const register = useCallback((scope: string, entries: readonly ShortcutEntry[]) => {
    setScopes((prev) => {
      const next = new Map(prev);
      next.set(scope, entries);
      return next;
    });
    return () => {
      setScopes((prev) => {
        const next = new Map(prev);
        next.delete(scope);
        return next;
      });
    };
  }, []);

  const groups = useMemo<readonly ShortcutGroup[]>(
    () => [
      { scope: "Global", entries: GLOBAL_SHORTCUTS },
      ...Array.from(scopes.entries()).map(([scope, entries]) => ({ scope, entries })),
    ],
    [scopes],
  );

  const value = useMemo<RegistryValue>(() => ({ groups, register }), [groups, register]);

  return <RegistryContext.Provider value={value}>{children}</RegistryContext.Provider>;
}

export function useShortcutGroups(): readonly ShortcutGroup[] {
  const ctx = useContext(RegistryContext);
  if (!ctx) throw new Error("useShortcutGroups must be used within a ShortcutsProvider");
  return ctx.groups;
}

/** Registers a page's shortcuts for the help overlay for as long as the calling component is
 * mounted. Call once per page with a stable `entries` array (memoize if it's computed). */
export function useRegisterShortcuts(scope: string, entries: readonly ShortcutEntry[]): void {
  const ctx = useContext(RegistryContext);
  if (!ctx) throw new Error("useRegisterShortcuts must be used within a ShortcutsProvider");
  const { register } = ctx;
  useEffect(() => register(scope, entries), [register, scope, entries]);
}
