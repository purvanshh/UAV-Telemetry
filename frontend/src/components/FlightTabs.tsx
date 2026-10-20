"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function FlightTabs({ flightId }: { flightId: string }) {
  const pathname = usePathname();
  const base = `/flights/${encodeURIComponent(flightId)}`;
  const tabs = [
    { href: base, label: "Overview" },
    { href: `${base}/timeline`, label: "Timeline" },
    { href: `${base}/reports`, label: "Report" },
    { href: `${base}/advanced`, label: "Advanced" },
  ];
  return (
    <nav className="flight-tabs" aria-label="Flight sections">
      {tabs.map((t) => (
        <Link key={t.href} href={t.href} aria-current={pathname === t.href ? "page" : undefined}>
          {t.label}
        </Link>
      ))}
    </nav>
  );
}
