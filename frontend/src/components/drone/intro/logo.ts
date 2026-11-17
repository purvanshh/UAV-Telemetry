/**
 * ARJUNA logo geometry, traced from the brand reference image (864 x 746 px) and kept in that pixel space.
 * Three solid shapes, mirror-symmetric about x = 407: a centre spike and two "claw" wedges.
 * The solid-white logo, the smoke mask and the drone flight routes are all derived from these numbers.
 */

export type Pt = [number, number];

export const LOGO_CX = 407;
export const LOGO_CY = 366;
/** Logo bounds (90..722 x 38..695) plus margin for smoke puffs that spill past the edge. */
export const BOX = { x0: 56, y0: 4, x1: 758, y1: 729 };
const MIRROR = 814;

const CENTRE: Pt[] = [
  [407, 38],
  [441, 410],
  [407, 695],
  [373, 410],
];

const LEFT: Pt[] = [
  [316, 146],
  [90, 553],
  [110, 612],
  [336, 507],
  [330, 462],
  [208, 517],
];

const mirror = (poly: Pt[]): Pt[] => poly.map(([x, y]) => [MIRROR - x, y]);

export const POLYS: Pt[][] = [CENTRE, LEFT, mirror(LEFT)];

/** Where each drone starts drawing, and the centre-line it flies ([x, y, smoke radius]). */
const CENTRE_ROUTE: number[][] = [
  [407, 38, 12],
  [407, 200, 22],
  [407, 410, 36],
  [407, 560, 24],
  [407, 695, 12],
];

const LEFT_ROUTE: number[][] = [
  [316, 146, 14],
  [270, 250, 26],
  [215, 380, 36],
  [160, 500, 40],
  [130, 555, 38],
  [200, 545, 34],
  [270, 515, 30],
  [328, 488, 26],
];

const mirrorRoute = (r: number[][]): number[][] => r.map(([x, y, rad]) => [MIRROR - x, y, rad]);

export interface RoutePoint {
  x: number;
  y: number;
  r: number;
}

/** Chaikin-smooth the control polyline (so drones carve corners) and resample it at ~1 unit spacing. */
function buildRoute(ctrl: number[][]): RoutePoint[] {
  let p = ctrl;
  for (let k = 0; k < 2; k++) {
    const next: number[][] = [p[0]];
    for (let i = 0; i < p.length - 1; i++) {
      const a = p[i];
      const b = p[i + 1];
      next.push(a.map((v, j) => v * 0.75 + b[j] * 0.25), a.map((v, j) => v * 0.25 + b[j] * 0.75));
    }
    next.push(p[p.length - 1]);
    p = next;
  }
  const out: RoutePoint[] = [];
  for (let i = 0; i < p.length - 1; i++) {
    const a = p[i];
    const b = p[i + 1];
    const steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])));
    for (let s = 0; s < steps; s++) {
      const f = s / steps;
      out.push({ x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, r: a[2] + (b[2] - a[2]) * f });
    }
  }
  const last = p[p.length - 1];
  out.push({ x: last[0], y: last[1], r: last[2] });
  return out;
}

export const ROUTES: RoutePoint[][] = [CENTRE_ROUTE, LEFT_ROUTE, mirrorRoute(LEFT_ROUTE)].map(buildRoute);

export function polysPath(): Path2D {
  const path = new Path2D();
  for (const poly of POLYS) {
    poly.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)));
    path.closePath();
  }
  return path;
}
