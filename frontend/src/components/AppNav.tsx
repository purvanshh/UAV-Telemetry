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

  const links = [
    { href: "/", label: "Upload" },
    ...(activeId
      ? [
          { href: `/flights/${activeId}`, label: "Analysis" },
          { href: `/flights/${activeId}/ai`, label: "AI Diagnostics" },
          { href: `/flights/${activeId}/reports`, label: "Reports" },
        ]
      : []),
  ];

  return (
    <header className="topbar">
      <Link href="/" className="brand">
        <span className="brand-mark">UAV</span>
        <span className="brand-name">Sentinel</span>
      </Link>
      <nav className="nav-links">
        {links.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={pathname === l.href ? "nav-link active" : "nav-link"}
          >
            {l.label}
          </Link>
        ))}
      </nav>
      {activeId && (
        <span className="chip mono" title="Active flight">
          {activeId}
        </span>
      )}
    </header>
  );
}
