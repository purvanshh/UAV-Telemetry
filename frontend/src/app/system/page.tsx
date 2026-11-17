"use client";

import { useCallback, useEffect, useState } from "react";
import { RefreshCw, ScanSearch, Server } from "lucide-react";
import { Button, Card, PageHeader, Pill } from "@/components/ui";
import { api, type AiStatus } from "@/lib/api";

type State = { health: "ok" | "down" | null; ai: AiStatus | null | "down"; at: Date | null };

export default function SystemPage() {
  const [st, setSt] = useState<State>({ health: null, ai: null, at: null });
  const [busy, setBusy] = useState(false);

  const check = useCallback(async () => {
    setBusy(true);
    const [h, a] = await Promise.allSettled([api.health(), api.aiStatus()]);
    setSt({
      health: h.status === "fulfilled" ? "ok" : "down",
      ai: a.status === "fulfilled" ? a.value : "down",
      at: new Date(),
    });
    setBusy(false);
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const ai = st.ai === "down" || st.ai === null ? null : st.ai;
  return (
    <main className="page page-narrow">
      <PageHeader
        title="System status"
        lede="Whether ARJUNA can read logs and run the detailed review right now."
        actions={
          <Button size="sm" icon={RefreshCw} loading={busy} onClick={() => void check()}>
            Check again
          </Button>
        }
      />
      <Card pad={false}>
        <div className="service">
          <span className={`status-icon tone-${st.health === "ok" ? "go" : st.health === "down" ? "warn" : "neutral"}`} aria-hidden="true">
            <Server />
          </span>
          <div>
            <strong>Analysis service</strong>
            <p className="small muted">Reads flight logs, checks the recording, and runs every fault check and the physics model.</p>
          </div>
          {st.health === null ? (
            <Pill>Checking…</Pill>
          ) : st.health === "ok" ? (
            <Pill tone="go" dot>
              Online
            </Pill>
          ) : (
            <Pill tone="warn" dot>
              Unreachable
            </Pill>
          )}
        </div>
        <div className="service">
          <span className={`status-icon tone-${ai?.configured ? "go" : st.ai === null ? "neutral" : "caution"}`} aria-hidden="true">
            <ScanSearch />
          </span>
          <div>
            <strong>Detailed AI review</strong>
            <p className="small muted">
              {ai?.configured
                ? `Uses ${ai.model}. Without it, every automatic check still works.`
                : "Not configured on the server. Every automatic check still works. Set OPENAI_API_KEY to turn it on."}
            </p>
          </div>
          {st.ai === null ? (
            <Pill>Checking…</Pill>
          ) : ai?.configured ? (
            <Pill tone="go" dot>
              Ready
            </Pill>
          ) : (
            <Pill tone="caution" dot>
              Off
            </Pill>
          )}
        </div>
      </Card>
      {st.at && (
        <p className="tiny muted" style={{ marginTop: "0.75rem" }}>
          Last checked {st.at.toLocaleTimeString()}.
        </p>
      )}
    </main>
  );
}
