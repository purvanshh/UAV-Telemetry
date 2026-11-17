"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { QuadTop } from "@/components/drone/Drone";

type Flight = { id: number; size: number; opacity: number; blur: number; keyframes: Keyframe[]; duration: number };

const MAX_AT_ONCE = 7;
const MAX_ON_PHONE = 3;
const FIRST_DELAY_MS = 1200;
const GAP_MS: [number, number] = [1400, 3800];
const SPEED_PX_S: [number, number] = [55, 125];
const SAMPLES = 36;

const rand = (a: number, b: number) => a + Math.random() * (b - a);
const pick = <T,>(xs: T[]) => xs[Math.floor(Math.random() * xs.length)];

/** One flight across the background: start and end off-screen, a gentle curve, nose along the path, banking into turns. */
function plan(w: number, h: number): Omit<Flight, "id"> {
  const m = 70;
  const routes: [number, number, number, number][] = [
    [-m, rand(0.12, 0.85) * h, w + m, rand(0.12, 0.85) * h], // left to right
    [w + m, rand(0.12, 0.85) * h, -m, rand(0.12, 0.85) * h], // right to left
    [rand(-0.05, 0.3) * w, -m, rand(0.7, 1.05) * w, h + m], // top-left to bottom-right
    [rand(-0.05, 0.3) * w, h + m, rand(0.7, 1.05) * w, -m], // bottom-left to top-right
    [rand(0.7, 1.05) * w, -m, rand(-0.05, 0.3) * w, h + m], // top-right to bottom-left
  ];
  const [x0, y0, x1, y1] = pick(routes);
  const dx = x1 - x0;
  const dy = y1 - y0;
  const dist = Math.hypot(dx, dy);
  // control point pushed sideways for a curved track
  const bow = rand(-0.22, 0.22) * dist;
  const cx = (x0 + x1) / 2 + (-dy / dist) * bow;
  const cy = (y0 + y1) / 2 + (dx / dist) * bow;
  const pt = (t: number) => [
    (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1,
    (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1,
  ];
  const heading = (t: number) => {
    const tx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx);
    const ty = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy);
    return (Math.atan2(ty, tx) * 180) / Math.PI;
  };
  const bank = Math.sign(bow) * Math.min(10, (Math.abs(bow) / dist) * 45); // lean into the turn
  const keyframes: Keyframe[] = [];
  for (let i = 0; i <= SAMPLES; i++) {
    const t = i / SAMPLES;
    const [x, y] = pt(t);
    const wobble = Math.sin(t * Math.PI * 3) * 1.5; // small attitude corrections
    keyframes.push({
      offset: t,
      transform: `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) rotate(${(heading(t) + 90 + bank * Math.sin(t * Math.PI) + wobble).toFixed(2)}deg)`,
    });
  }
  const depth = Math.random(); // 0 = far, 1 = near
  return {
    size: Math.round(13 + depth * 14),
    opacity: 0.08 + depth * 0.09,
    blur: depth < 0.3 ? 0.6 : 0,
    keyframes,
    duration: (dist / rand(...SPEED_PX_S)) * 1000,
  };
}

function Ambient({ flight, onDone }: { flight: Flight; onDone: (id: number) => void }) {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const anim = node.animate(flight.keyframes, { duration: flight.duration, easing: "linear", fill: "both" });
    anim.onfinish = () => onDone(flight.id);
    return () => anim.cancel();
  }, [flight, onDone]);
  return (
    <div
      ref={el}
      className="ambient-drone"
      style={{ width: flight.size, height: flight.size, opacity: flight.opacity, filter: flight.blur ? `blur(${flight.blur}px)` : undefined }}
    >
      <QuadTop size={flight.size} />
    </div>
  );
}

/** Small drones that now and then cross the background, behind every card, chart and control. */
export function AmbientDrones() {
  const [flights, setFlights] = useState<Flight[]>([]);
  const [enabled, setEnabled] = useState(false);
  const next = useRef(1);
  const layer = useRef<HTMLDivElement>(null);
  const done = useCallback((id: number) => setFlights((fs) => fs.filter((f) => f.id !== id)), []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setEnabled(!mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!enabled) {
      setFlights([]);
      return;
    }
    let timer = 0;
    const max = () => (window.innerWidth < 640 ? MAX_ON_PHONE : MAX_AT_ONCE);
    const schedule = (ms: number) => {
      timer = window.setTimeout(() => {
        if (!document.hidden) {
          setFlights((fs) =>
            fs.length >= max() ? fs : [...fs, { id: next.current++, ...plan(window.innerWidth, window.innerHeight) }]
          );
        }
        schedule(rand(...GAP_MS));
      }, ms);
    };
    schedule(FIRST_DELAY_MS);
    // pause everything while the tab is hidden; resume where they were
    const vis = () =>
      layer.current?.getAnimations({ subtree: true }).forEach((a) => (document.hidden ? a.pause() : a.play()));
    document.addEventListener("visibilitychange", vis);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", vis);
    };
  }, [enabled]);

  if (!enabled) return null;
  return (
    <div ref={layer} className="ambient" aria-hidden="true">
      {flights.map((f) => (
        <Ambient key={f.id} flight={f} onDone={done} />
      ))}
    </div>
  );
}
