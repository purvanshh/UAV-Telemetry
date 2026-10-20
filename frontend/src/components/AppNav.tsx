"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** Top bar. `flightId` is accepted for older pages; flight navigation lives in FlightTabs. */
export function AppNav(_props: { flightId?: string }) {
  void _props;
  const pathname = usePathname();
  const items = [
    { href: "/", label: "Flights", active: pathname === "/" },
    { href: "/upload", label: "New analysis", active: pathname === "/upload" },
  ];
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <Link href="/" className="brand">
          UAV Sentinel
        </Link>
        <nav className="topnav" aria-label="Main">
          {items.map((i) => (
            <Link key={i.href} href={i.href} aria-current={i.active ? "page" : undefined}>
              {i.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
