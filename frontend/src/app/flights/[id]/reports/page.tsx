"use client";

import { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

/** The report is now the flight's main page; this route is kept so saved links still work. */
export default function ReportRedirect() {
  const id = String(useParams().id || "");
  const router = useRouter();
  useEffect(() => {
    router.replace(`/flights/${encodeURIComponent(id)}`);
  }, [id, router]);
  return null;
}
