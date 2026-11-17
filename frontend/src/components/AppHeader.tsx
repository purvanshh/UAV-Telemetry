"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, GitCompareArrows, List, Plus } from "lucide-react";

const NAV = [
  { href: "/", label: "Flights", icon: List, match: (p: string) => p === "/" || p.startsWith("/analysis") },
  { href: "/compare", label: "Compare", icon: GitCompareArrows, match: (p: string) => p.startsWith("/compare") },
  { href: "/system", label: "System", icon: Activity, match: (p: string) => p.startsWith("/system") },
];

/** The arrowhead mark: ARJUNA is the archer. */
function Mark() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 2 21 21l-9-5-9 5L12 2Z" fill="currentColor" />
    </svg>
  );
}

export function AppHeader() {
  const path = usePathname() || "/";
  return (
    <header className="app-header no-print">
      <div className="app-header-inner">
        <Link href="/" className="brand" aria-label="ARJUNA home">
          <span className="brand-mark">
            <Mark />
          </span>
          <span className="brand-text">ARJUNA</span>
        </Link>
        <nav className="nav" aria-label="Main">
          {NAV.map(({ href, label, icon: Icon, match }) => (
            <Link key={href} href={href} aria-current={match(path) ? "page" : undefined}>
              <Icon size={16} aria-hidden="true" />
              <span>{label}</span>
            </Link>
          ))}
        </nav>
        <Link href="/upload" className="btn btn-primary btn-sm" aria-current={path.startsWith("/upload") ? "page" : undefined}>
          <Plus size={15} aria-hidden="true" />
          New analysis
        </Link>
      </div>
    </header>
  );
}
