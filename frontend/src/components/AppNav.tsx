"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getLastFlightId } from "@/lib/api";

export function AppNav({ flightId }: { flightId?: string }) {
  const pathname = usePathname();
  const [storedId, setStoredId] = useState<string | null>(null);

  useEffect(() => {
    setStoredId(getLastFlightId());
  }, [pathname, flightId]);

  const activeId = flightId || storedId || undefined;
  const analysisHref = activeId ? `/flights/${activeId}` : "";
  const aiHref = activeId ? `/flights/${activeId}/ai` : "";
  const reportsHref = activeId ? `/flights/${activeId}/reports` : "";

  const links = [
    { href: "/", label: "Overview", enabled: true, match: (p: string) => p === "/" },
    { href: "/upload", label: "New Analysis", enabled: true, match: (p: string) => p === "/upload" },
    {
      href: analysisHref,
      label: "Flight Analysis",
      enabled: Boolean(activeId),
      match: (p: string) => Boolean(activeId) && p === `/flights/${activeId}`,
    },
    {
      href: aiHref,
      label: "AI Diagnostics",
      enabled: Boolean(activeId),
      match: (p: string) => Boolean(activeId) && p.startsWith(`/flights/${activeId}/ai`),
    },
    {
      href: reportsHref,
      label: "Reports",
      enabled: Boolean(activeId),
      match: (p: string) => Boolean(activeId) && p.startsWith(`/flights/${activeId}/reports`),
    },
  ];

  return (
    <>
      <aside className="sidebar">
        <div>
          <Link href="/" className="brand">
            <span className="brand-dot" />
            <span className="brand-name">UAV Sentinel</span>
          </Link>
          <nav className="nav-links">
            {links.map((l) =>
              l.enabled ? (
                <Link
                  key={l.label}
                  href={l.href}
                  className={l.match(pathname) ? "nav-link active" : "nav-link"}
                >
                  {l.label}
                </Link>
              ) : (
                <span key={l.label} className="nav-link disabled" title="Open a saved flight first">
                  {l.label}
                </span>
              )
            )}
          </nav>
        </div>
        <div className="sidebar-foot">12-feature LSTM · BARO excluded</div>
      </aside>
      <header className="topstrip">
        <div>
          <div className="track-label">Active flight</div>
          <div className="track-id">{activeId || "None selected"}</div>
        </div>
        <Link className="btn btn-primary" href="/upload">
          New analysis
        </Link>
      </header>
    </>
  );
}
