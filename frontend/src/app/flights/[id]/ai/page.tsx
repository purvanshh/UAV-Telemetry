"use client";

import { FormEvent, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { api, ApiError, AiStatus, Diagnostic, setLastFlightId } from "@/lib/api";

type Msg = { role: string; content: string };

export default function AiDiagnosticsPage() {
  const params = useParams();
  const flightId = String(params.id || "");
  const [aiStatus, setAiStatus] = useState<AiStatus | null>(null);
  const [diagnostic, setDiagnostic] = useState<Diagnostic | null>(null);
  const [trace, setTrace] = useState<string>("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<string>("");
  const [events, setEvents] = useState<{ event_id: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setLastFlightId(flightId);
        setAiStatus(await api.aiStatus());
        const flight = await api.getFlight(flightId);
        if (flight.has_report) {
          const t = await api.evidenceTrace(flightId);
          setTrace(t.formatted);
          setEvents(await api.events(flightId));
        }
        if (flight.has_ai_diagnostic) {
          setDiagnostic(await api.getDiagnostic(flightId));
        }
        const chat = await api.getChat(flightId);
        setMessages(chat.messages || []);
        if (chat.selected_event_id) setSelectedEvent(chat.selected_event_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load AI page");
      }
    })();
  }, [flightId]);

  async function runDiagnostic() {
    setBusy(true);
    setError(null);
    try {
      const ids = selectedEvent ? [selectedEvent] : undefined;
      const res = await api.runDiagnostic(flightId, ids);
      setDiagnostic(res.diagnostic);
      if (res.trace && typeof res.trace === "object") {
        const t = await api.evidenceTrace(flightId);
        setTrace(t.formatted);
      }
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "AI failed");
    } finally {
      setBusy(false);
    }
  }

  async function onChat(e: FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || !diagnostic) return;
    setChatBusy(true);
    setError(null);
    const userMsg = prompt.trim();
    setPrompt("");
    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    try {
      const res = await api.postChat(flightId, userMsg, selectedEvent || null);
      setMessages(res.chat.messages);
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Chat failed");
    } finally {
      setChatBusy(false);
    }
  }

  async function clearChat() {
    const chat = (await api.clearChat(flightId)) as {
      messages: Msg[];
    };
    setMessages(chat.messages || []);
  }

  return (
    <>
      <AppNav flightId={flightId} />
      <main>
        <section className="hero">
          <span className="kicker">Evidence memo</span>
          <h1>AI Diagnostics</h1>
          <p>
            GPT-4o mini interprets detector evidence only. It does not change scores,
            flags, or events. BARO is excluded from LSTM detection.
          </p>
        </section>

        {error && <div className="alert alert-error">{error}</div>}

        {!aiStatus?.configured && (
          <div className="alert alert-warn">
            OPENAI_API_KEY is not configured on the API server. Detector exports still work.
            Set the key in the project <span className="mono">.env</span> and restart uvicorn.
          </div>
        )}

        <div className="ai-split">
        <div>
        <section className="panel">
          <h2>Evidence pipeline trace</h2>
          {trace ? <pre className="trace">{trace}</pre> : <p className="muted">Analyze the flight first.</p>}
        </section>

        <section className="panel">
          <h2>Generate diagnostic</h2>
          <div className="row" style={{ marginBottom: "0.85rem" }}>
            <div className="file-field" style={{ minWidth: 220 }}>
              <label htmlFor="focus">Focus event (optional)</label>
              <select
                id="focus"
                value={selectedEvent}
                onChange={(e) => setSelectedEvent(e.target.value)}
              >
                <option value="">(all / flight overview)</option>
                {events.map((ev) => (
                  <option key={ev.event_id} value={ev.event_id}>
                    {ev.event_id}
                  </option>
                ))}
              </select>
            </div>
            <button
              className="btn btn-primary"
              onClick={runDiagnostic}
              disabled={busy || !aiStatus?.configured}
            >
              {busy ? "Calling model…" : "Generate AI diagnostic"}
            </button>
            <button className="btn" type="button" onClick={clearChat}>
              Clear chat
            </button>
          </div>
          <p className="muted">Model: {aiStatus?.model || "…"}</p>
        </section>

        {diagnostic && (
          <section className="panel">
            <h2>AI diagnostic report</h2>
            {diagnostic.insufficient_evidence && (
              <div className="alert alert-warn">Insufficient evidence for a specific physical fault.</div>
            )}
            <p>{diagnostic.summary}</p>
            {diagnostic.overall_limitations?.length ? (
              <div style={{ marginTop: "0.75rem" }}>
                <h3>Limitations</h3>
                <ul className="muted">
                  {diagnostic.overall_limitations.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {diagnostic.findings?.map((f) => (
              <div key={f.event_id} className="panel" style={{ background: "var(--surface-2)" }}>
                <h3 className="mono">{f.event_id}</h3>
                <p className="muted">Observed facts</p>
                <ul>
                  {(f.observed_facts || []).map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
                <p className="muted">Possible causes</p>
                <ul>
                  {(f.possible_causes || []).map((c) => (
                    <li key={c.hypothesis}>
                      <strong>{c.hypothesis}</strong> · {c.confidence}
                    </li>
                  ))}
                  {!f.possible_causes?.length && <li>(none)</li>}
                </ul>
                <p className="muted">Recommended checks</p>
                <ul>
                  {(f.recommended_checks || []).map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              </div>
            ))}
          </section>
        )}
        </div>

        <section className="panel chat-panel">
          <h2>Flight chat</h2>
          <div className="chat-log">
            {messages.map((m, i) => (
              <div key={`${m.role}-${i}`} className={`bubble ${m.role}`}>
                {m.content}
              </div>
            ))}
            {!messages.length && <p className="muted">No messages yet.</p>}
          </div>
          {error && <div className="alert alert-error">{error}</div>}
          <form className="row" onSubmit={onChat}>
            <input
              className="chat-input"
              style={{ flex: 1 }}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask a follow-up about this flight…"
              disabled={!diagnostic || !aiStatus?.configured || chatBusy}
            />
            <button
              className="btn btn-primary"
              type="submit"
              disabled={!diagnostic || !aiStatus?.configured || chatBusy}
            >
              Send
            </button>
          </form>
        </section>
        </div>

        <Link className="btn" href={`/flights/${flightId}`}>
          Back to analysis
        </Link>
      </main>
    </>
  );
}
