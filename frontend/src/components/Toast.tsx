"use client";

import { X } from "lucide-react";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { TONE_ICON, toneClass, type Tone } from "@/lib/status";

type Toast = { id: number; tone: Tone; title: string; body?: string };
const Ctx = createContext<(t: Omit<Toast, "id">) => void>(() => undefined);

/** Short confirmations for things that happen in the background (a review finishing, a deletion). */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const next = useRef(1);
  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((t) => t.id !== id)), []);
  const push = useCallback(
    (t: Omit<Toast, "id">) => {
      const id = next.current++;
      setItems((xs) => [...xs.slice(-2), { ...t, id }]);
      window.setTimeout(() => dismiss(id), t.tone === "warn" ? 9000 : 5000);
    },
    [dismiss]
  );
  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="toasts" aria-live="polite" aria-relevant="additions">
        {items.map((t) => {
          const I = TONE_ICON[t.tone];
          return (
            <div key={t.id} className={`toast ${toneClass(t.tone)}`} role={t.tone === "warn" ? "alert" : "status"}>
              <I size={18} aria-hidden="true" />
              <div>
                <strong>{t.title}</strong>
                {t.body && <span className="muted">{t.body}</span>}
              </div>
              <button type="button" className="btn btn-ghost btn-sm btn-icon" aria-label="Dismiss" onClick={() => dismiss(t.id)}>
                <X size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </Ctx.Provider>
  );
}

export function useToast() {
  return useContext(Ctx);
}
