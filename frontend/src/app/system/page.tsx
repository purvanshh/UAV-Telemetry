"use client";

import { useEffect, useState } from "react";
import { AppNav } from "@/components/AppNav";
import { AiStatus, api } from "@/lib/api";

export default function SystemPage() {
  const [health, setHealth] = useState<"ok" | "down" | null>(null);
  const [ai, setAi] = useState<AiStatus | null>(null);

  useEffect(() => {
    api.health().then(() => setHealth("ok")).catch(() => setHealth("down"));
    api.aiStatus().then(setAi).catch(() => setAi(null));
  }, []);

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head">
          <h1>System status</h1>
        </div>
        <ul className="status-list">
          <li>
            <strong>Analysis service</strong>{" "}
            {health === null ? "checking…" : health === "ok" ? <span className="chip s-ok">reachable</span> : <span className="chip s-error">not reachable</span>}
            <p className="small muted">Reads ARJUNA logs, verifies integrity and runs every fault-code check and the physics model.</p>
          </li>
          <li>
            <strong>AI agent</strong>{" "}
            {ai ? (
              ai.configured ? <span className="chip s-ok">configured · {ai.model}</span> : <span className="chip s-caution">not configured</span>
            ) : (
              "—"
            )}
            <p className="small muted">
              Investigates each flight through code tools. Without it, every code finding is still shown. Set
              OPENAI_API_KEY (and optionally UAV_AGENT_MODEL) on the server.
            </p>
          </li>
        </ul>
      </main>
    </>
  );
}
