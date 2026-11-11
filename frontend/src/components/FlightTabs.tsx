"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** Secondary views of one Deep Analysis flight. The report is the default view. */
export function FlightTabs({ flightId }: { flightId: string }) {
  const pathname = usePathname();
  const base = `/flights/${encodeURIComponent(flightId)}`;
  const tabs = [
    { href: base, label: "Report" },
    { href: `${base}/timeline`, label: "Timeline" },
    { href: `${base}/advanced`, label: "Technical views" },
  ];
  return (
    <nav className="flight-tabs no-print" aria-label="Flight sections">
      {tabs.map((t) => (
        <Link key={t.href} href={t.href} aria-current={pathname === t.href ? "page" : undefined}>
          {t.label}
        </Link>
      ))}
    </nav>
  );
}
