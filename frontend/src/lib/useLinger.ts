import { useEffect, useRef, useState } from "react";

/** True while `on` is true, and for `ms` after it turns false: time for an exit animation. */
export function useLinger(on: boolean, ms: number): boolean {
  const [linger, setLinger] = useState(on);
  const prev = useRef(on);
  useEffect(() => {
    if (on) setLinger(true);
    else if (prev.current) {
      const t = window.setTimeout(() => setLinger(false), ms);
      prev.current = on;
      return () => window.clearTimeout(t);
    }
    prev.current = on;
  }, [on, ms]);
  return on || linger;
}
