"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/", label: "Flights", match: (p: string) => p === "/" || p.startsWith("/flights") || p.startsWith("/direct") },
  { href: "/upload", label: "New analysis", match: (p: string) => p.startsWith("/upload") },
  { href: "/system", label: "System status", match: (p: string) => p.startsWith("/system") },
];

/** Top bar: three places only. `flightId` is accepted for older pages and ignored. */
export function AppNav(_props: { flightId?: string }) {
  void _props;
  const pathname = usePathname() || "/";
  return (
    <header className="topbar no-print">
      <div className="topbar-inner">
        <Link href="/" className="brand">
          UAV Sentinel
        </Link>
        <nav className="topnav" aria-label="Main">
          {ITEMS.map((i) => (
            <Link key={i.href} href={i.href} aria-current={i.match(pathname) ? "page" : undefined}>
              {i.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
