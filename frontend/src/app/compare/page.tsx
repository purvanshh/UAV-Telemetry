"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowDown, ArrowLeftRight, ArrowUp, CircleCheck, GitCompareArrows, Minus, OctagonAlert, Drone, Sparkle, TriangleAlert } from "lucide-react";
import { Alert, Button, ButtonLink, Card, Empty, Hero, PageHeader, Pill, Skeleton } from "@/components/ui";
import { api, ApiError, type AnalysisMeta, type CompareResult } from "@/lib/api";
import { fullDate } from "@/lib/format";
import { COMPARE_TONE, type Tone } from "@/lib/status";
import { metaTitle } from "@/lib/verdict";

const STATE: Record<string, { word: string; tone: Tone; icon: typeof CircleCheck }> = {
  fixed: { word: "Fixed", tone: "go", icon: CircleCheck },
  better: { word: "Better", tone: "go", icon: ArrowDown },
  still: { word: "Still there", tone: "caution", icon: TriangleAlert },
  worse: { word: "Worse", tone: "warn", icon: OctagonAlert },
  new: { word: "New", tone: "warn", icon: Sparkle },
  new_unchecked: { word: "New", tone: "caution", icon: Sparkle },
  unknown: { word: "Cannot tell", tone: "neutral", icon: Minus },
};

function withUnit(v: number, unit: string) {
  return unit === "°" || unit === "%" ? `${v}${unit}` : `${v} ${unit}`.trim();
}

export default function ComparePage() {
  return (
    <Suspense fallback={null}>
      <Compare />
    </Suspense>
  );
}

function Compare() {
  const params = useSearchParams();
  const router = useRouter();
  const [items, setItems] = useState<AnalysisMeta[] | null>(null);
  const [before, setBefore] = useState(params.get("before") ?? "");
  const [after, setAfter] = useState(params.get("after") ?? "");
  const [data, setData] = useState<Awaited<ReturnType<typeof api.compare>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.list().then(setItems).catch(() => setItems([]));
  }, []);

  useEffect(() => {
    const b = params.get("before");
    const a = params.get("after");
    setBefore(b ?? "");
    setAfter(a ?? "");
    if (!b || !a || b === a) return;
    setBusy(true);
    setError(null);
    api
      .compare(b, a)
      .then(setData)
      .catch((e) => setError(e instanceof ApiError || e instanceof Error ? e.message : "These flights could not be compared."))
      .finally(() => setBusy(false));
  }, [params]);

  const flights = useMemo(() => (items ?? []).filter((m) => !m.refused), [items]);
  const ready = !!before && !!after && before !== after;
  const go = (b: string, a: string) => router.push(`/compare?before=${encodeURIComponent(b)}&after=${encodeURIComponent(a)}`);
  const r: CompareResult | undefined = data?.result;

  return (
    <main className="page">
      <PageHeader
        title="Compare two flights"
        lede="Check whether a fix worked: pick the flight before the fix, then the flight after. Or compare any two flights."
      />

      {items === null ? (
        <Skeleton h={120} r={14} />
      ) : flights.length < 2 ? (
        <Empty icon={Drone} title="You need two drone flights to compare" action={<ButtonLink href="/upload" variant="primary">Upload a flight</ButtonLink>}>
          Upload at least two flights, for example one before a repair and one after it.
        </Empty>
      ) : (
        <Card>
          <form
            className="compare-picker no-print"
            onSubmit={(e) => {
              e.preventDefault();
              if (ready) go(before, after);
            }}
          >
            <label className="field">
              <span className="field-label">Earlier flight (before the fix)</span>
              <select className="select" value={before} onChange={(e) => setBefore(e.target.value)}>
                <option value="">Choose a flight</option>
                {flights.map((m) => (
                  <option key={m.analysis_id} value={m.analysis_id}>
                    {metaTitle(m)} · {fullDate(m.created_at)}
                  </option>
                ))}
              </select>
            </label>
            <Button
              variant="ghost"
              icon={ArrowLeftRight}
              aria-label="Swap the flights"
              title="Swap"
              onClick={() => {
                setBefore(after);
                setAfter(before);
              }}
            />
            <label className="field">
              <span className="field-label">Later flight (after the fix)</span>
              <select className="select" value={after} onChange={(e) => setAfter(e.target.value)}>
                <option value="">Choose a flight</option>
                {flights.map((m) => (
                  <option key={m.analysis_id} value={m.analysis_id}>
                    {metaTitle(m)} · {fullDate(m.created_at)}
                  </option>
                ))}
              </select>
            </label>
            <button type="submit" className="btn btn-primary" disabled={!ready || busy}>
              {busy ? <span className="spinner" aria-hidden="true" /> : <GitCompareArrows size={16} aria-hidden="true" />}
              Compare
            </button>
          </form>
        </Card>
      )}

      {error && (
        <div style={{ marginTop: "1rem" }}>
          <Alert tone="warn" title="Could not compare.">
            {error}
          </Alert>
        </div>
      )}

      {busy && !r && (
        <div className="stack" style={{ marginTop: "1.25rem" }}>
          <Skeleton h={90} r={14} />
          <Skeleton h={220} r={14} />
        </div>
      )}

      {r && data && (
        <div className="stack" style={{ ["--gap" as string]: "1.25rem", marginTop: "1.5rem" }} key={`${data.before.analysis_id}-${data.after.analysis_id}`}>
          <Hero
            tone={COMPARE_TONE[r.status] ?? "neutral"}
            title={r.status_text}
            detail={
              <>
                {r.status_detail}{" "}
                <span className="small muted">
                  <Link href={`/analysis/${data.before.analysis_id}`}>{data.before.flight_id ?? data.before.filename}</Link> →{" "}
                  <Link href={`/analysis/${data.after.analysis_id}`}>{data.after.flight_id ?? data.after.filename}</Link>
                </span>
              </>
            }
          />
          {r.notes.map((n) => (
            <Alert key={n} tone="info">
              {n}
            </Alert>
          ))}

          {r.problems.length > 0 && (
            <section>
              <h2 className="section-title" style={{ marginTop: 0 }}>
                Problems, flight to flight <span className="count">{r.problems.length}</span>
              </h2>
              <ul className="plain stack" style={{ ["--gap" as string]: "0.6rem" }}>
                {r.problems.map((p, i) => {
                  const st = STATE[p.state];
                  return (
                    <li key={p.code} className={`problem tone-${st.tone} rise`} style={{ ["--i" as string]: i }}>
                      <span className="problem-icon" aria-hidden="true">
                        <st.icon />
                      </span>
                      <div className="problem-top">
                        <Pill tone={st.tone}>{st.word}</Pill>
                        <span className="problem-name">{p.name}</span>
                      </div>
                      <p className="problem-line">{p.text}</p>
                      {p.evidence ? <p className="small mono text-2">{p.evidence}</p> : <span />}
                    </li>
                  );
                })}
              </ul>
            </section>
          )}

          {r.metrics.length > 0 && (
            <section>
              <h2 className="section-title">What changed</h2>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Measurement</th>
                      <th className="num">Before</th>
                      <th className="num">After</th>
                      <th>Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.metrics.map((m) => {
                      const tone: Tone = m.status === "better" ? "go" : m.status === "worse" ? "warn" : "neutral";
                      const Arrow = m.after > m.before ? ArrowUp : m.after < m.before ? ArrowDown : Minus;
                      return (
                        <tr key={m.key}>
                          <td>{m.label}</td>
                          <td className="num">{withUnit(m.before, m.unit)}</td>
                          <td className="num">{withUnit(m.after, m.unit)}</td>
                          <td>
                            <span className={`delta tone-${tone}`}>
                              <Arrow size={14} aria-hidden="true" />
                              {m.status === "same"
                                ? "About the same"
                                : `${m.status === "better" ? "Better" : "Worse"}${m.change_pct != null ? ` · ${Math.abs(m.change_pct)}%` : ""}`}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      )}
    </main>
  );
}
