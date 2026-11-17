import { BOX, LOGO_CX, LOGO_CY, POLYS, ROUTES, polysPath } from "./logo";

/**
 * Cinematic intro: three UAVs descend through fog, write the logo in smoke, fly off, the smoke resolves into the
 * solid mark, and the camera pushes through it into the app. One 2D canvas, no dependencies.
 * Timeline is absolute milliseconds so every layer stays in sync whatever the frame rate.
 */

export interface IntroCallbacks {
  /** the app should start revealing underneath */
  onReveal: () => void;
  /** overlay can be removed */
  onDone: () => void;
}
export interface IntroHandle {
  skip: () => void;
  stop: () => void;
}

const FULL = { fog: 1100, enter: [100, 200, 300], arrive: 1250, drawEnd: [3250, 3330, 3410], hoverEnd: 3700, exitEnd: 4350, crossA: 3350, crossB: 4000, textA: 3800, textB: 4450, zoomA: 4750, zoomDur: 1700, reveal: 5550 };
const LITE = { crossA: 0, crossB: 700, textA: 600, textB: 1200, zoomA: 1500, zoomDur: 1500, reveal: 2150 };
const ZOOM_MAX = 26;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (a: number, b: number, v: number) => {
  const x = clamp01((v - a) / (b - a));
  return x * x * (3 - 2 * x);
};
const easeOutCubic = (x: number) => 1 - Math.pow(1 - x, 3);
const easeInOutSine = (x: number) => -(Math.cos(Math.PI * x) - 1) / 2;
const easeInOutQuad = (x: number) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);

function rngFrom(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCanvas(w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

/** Cloudy puff sprites: many soft blobs under a radial falloff, so a stamp reads as smoke rather than a dot. */
function makeSmokeSprites(rng: () => number, n: number) {
  const out: HTMLCanvasElement[] = [];
  for (let v = 0; v < n; v++) {
    const c = makeCanvas(128, 128);
    const g = c.getContext("2d")!;
    for (let j = 0; j < 24; j++) {
      const ang = rng() * Math.PI * 2;
      const d = rng() * 34;
      const x = 64 + Math.cos(ang) * d;
      const y = 64 + Math.sin(ang) * d;
      const r = 14 + rng() * 26;
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, "rgba(238,244,255,0.32)");
      grad.addColorStop(0.55, "rgba(238,244,255,0.13)");
      grad.addColorStop(1, "rgba(238,244,255,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 128, 128);
    }
    g.globalCompositeOperation = "destination-in";
    const fall = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    fall.addColorStop(0, "rgba(0,0,0,1)");
    fall.addColorStop(0.55, "rgba(0,0,0,0.85)");
    fall.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = fall;
    g.fillRect(0, 0, 128, 128);
    out.push(c);
  }
  return out;
}

/** Tileable-ish cloud mask; drawn with destination-out over the smoke so it billows instead of reading as flat white. */
function makeCloudNoise(rng: () => number) {
  const c = makeCanvas(256, 256);
  const g = c.getContext("2d")!;
  for (let j = 0; j < 70; j++) {
    const x = rng() * 256;
    const y = rng() * 256;
    const r = 12 + rng() * 38;
    for (const [ox, oy] of [[0, 0], [-256, 0], [256, 0], [0, -256], [0, 256]]) {
      const grad = g.createRadialGradient(x + ox, y + oy, 0, x + ox, y + oy, r);
      grad.addColorStop(0, "rgba(0,0,0,0.30)");
      grad.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 256, 256);
    }
  }
  return c;
}

function makeFogSprite() {
  const c = makeCanvas(256, 256);
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "rgba(96,138,208,1)");
  grad.addColorStop(0.5, "rgba(60,96,160,0.5)");
  grad.addColorStop(1, "rgba(40,70,130,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  return c;
}

interface Stamp {
  x: number;
  y: number;
  r: number;
  a: number;
  v: number;
}

interface DroneState {
  hx: number;
  hy: number;
  heading: number;
  lastS: number;
  px: number;
  py: number;
}

export function runIntro(canvas: HTMLCanvasElement, cb: IntroCallbacks, opts: { lite: boolean }): IntroHandle {
  const lite = opts.lite;
  const ctx = canvas.getContext("2d", { alpha: true })!;
  const overlay = canvas.parentElement as HTMLElement | null;
  const rng = rngFrom(7);
  const sprites = lite ? [] : makeSmokeSprites(rng, 4);
  const fogSprite = lite ? null : makeFogSprite();
  const cloudNoise = lite ? null : makeCloudNoise(rng);
  let cloudPattern: CanvasPattern | null = null;
  const logoPath = polysPath();

  let W = 0;
  let H = 0;
  let dpr = 1;
  let S = 1;
  let cx = 0;
  let cy = 0;
  let kl = 1;
  let layer: HTMLCanvasElement;
  let lctx: CanvasRenderingContext2D;
  let mask: HTMLCanvasElement;
  let tmp: HTMLCanvasElement;
  let tctx: CanvasRenderingContext2D;
  let bgGrad: CanvasGradient;
  let D = 40;
  const stamps: Stamp[] = [];
  let quality = 1;

  const drones: DroneState[] = [0, 1, 2].map(() => ({ hx: 0, hy: 0, heading: Math.PI / 2, lastS: 0, px: 0, py: -999 }));

  const stampToLayer = (s: Stamp) => {
    lctx.globalAlpha = s.a;
    const r = s.r * kl;
    lctx.drawImage(sprites[s.v], (s.x - BOX.x0) * kl - r, (s.y - BOX.y0) * kl - r, r * 2, r * 2);
  };

  const layout = () => {
    W = window.innerWidth;
    H = window.innerHeight;
    dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(3.4e6 / (W * H))));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    S = Math.min((W * (W < 700 ? 0.72 : 0.4)) / 632, (H * 0.56) / 657);
    cx = W / 2;
    cy = H * 0.45;
    D = Math.max(26, Math.min(54, Math.min(W, H) * 0.075));
    bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.hypot(W, H) * 0.6);
    bgGrad.addColorStop(0, "#112445");
    bgGrad.addColorStop(0.55, "#0a1428");
    bgGrad.addColorStop(1, "#050a15");

    kl = Math.min(S * dpr, 2.4);
    const lw = (BOX.x1 - BOX.x0) * kl;
    const lh = (BOX.y1 - BOX.y0) * kl;
    layer = makeCanvas(lw, lh);
    lctx = layer.getContext("2d")!;
    mask = makeCanvas(lw, lh);
    tmp = makeCanvas(lw, lh);
    tctx = tmp.getContext("2d")!;
    cloudPattern = cloudNoise ? tctx.createPattern(cloudNoise, "repeat") : null;
    const m = mask.getContext("2d")!;
    m.setTransform(kl, 0, 0, kl, -BOX.x0 * kl, -BOX.y0 * kl);
    m.fillStyle = "#fff";
    m.fill(logoPath);
    for (const s of stamps) stampToLayer(s);
  };
  layout();

  // ---------------------------------------------------------------- geometry helpers
  const toScreen = (x: number, y: number) => ({ x: cx + (x - LOGO_CX) * S, y: cy + (y - LOGO_CY) * S });
  const routeLen = ROUTES.map((r) => r.length - 1);
  const tipScreen = ROUTES.map((r) => toScreen(r[0].x, r[0].y));
  const endScreen = ROUTES.map((r) => toScreen(r[r.length - 1].x, r[r.length - 1].y));
  const EXIT_DIR = [
    { x: 0.15, y: 1 },
    { x: -1, y: -0.55 },
    { x: 1, y: -0.55 },
  ];
  const ENTER_START = [
    { fx: 0.04, fy: -0.14 },
    { fx: -0.32, fy: -0.1 },
    { fx: 0.34, fy: -0.1 },
  ];
  const ENTER_CTRL = [
    { fx: 0.22, fy: 0.12 },
    { fx: -0.1, fy: 0.3 },
    { fx: 0.3, fy: 0.02 },
  ];

  const dronePos = (i: number, t: number): { x: number; y: number; vis: boolean; drawing: boolean } => {
    const tip = tipScreen[i];
    const end = endScreen[i];
    const es = FULL.enter[i];
    const de = FULL.drawEnd[i];
    if (t < es) return { x: tip.x, y: -D * 3, vis: false, drawing: false };
    if (t < FULL.arrive) {
      const u = easeOutCubic((t - es) / (FULL.arrive - es));
      const p0 = { x: cx + ENTER_START[i].fx * W, y: ENTER_START[i].fy * H };
      const c = { x: tip.x + ENTER_CTRL[i].fx * W, y: tip.y * 0.35 + ENTER_CTRL[i].fy * H * 0.4 };
      const a = (1 - u) * (1 - u);
      const b = 2 * (1 - u) * u;
      const d = u * u;
      return { x: a * p0.x + b * c.x + d * tip.x, y: a * p0.y + b * c.y + d * tip.y, vis: true, drawing: false };
    }
    if (t < de) {
      const s = easeInOutSine((t - FULL.arrive) / (de - FULL.arrive)) * routeLen[i];
      const r = ROUTES[i][Math.min(routeLen[i], Math.floor(s))];
      const p = toScreen(r.x, r.y);
      return { x: p.x, y: p.y, vis: true, drawing: true };
    }
    if (t < FULL.hoverEnd) {
      const h = (t - de) / (FULL.hoverEnd - de);
      const w = Math.sin(Math.PI * h);
      const amp = D * 0.28;
      return { x: end.x + Math.sin(h * 6.3 + i) * amp * w, y: end.y + Math.cos(h * 5.1 + i * 2) * amp * 0.6 * w, vis: true, drawing: false };
    }
    const u = clamp01((t - FULL.hoverEnd) / (FULL.exitEnd - FULL.hoverEnd));
    const dist = Math.max(W, H) * 1.1;
    const e = Math.pow(u, 1.9);
    const dir = EXIT_DIR[i];
    const n = Math.hypot(dir.x, dir.y);
    return { x: end.x + (dir.x / n) * dist * e, y: end.y + (dir.y / n) * dist * e, vis: u < 1, drawing: false };
  };

  // ---------------------------------------------------------------- drawing pieces
  const drawFog = (t: number, a: number) => {
    if (!fogSprite || a <= 0) return;
    const sec = t / 1000;
    ctx.globalCompositeOperation = "source-over";
    const blobs = quality < 0.6 ? 3 : 5;
    for (let i = 0; i < blobs; i++) {
      const size = Math.max(W, H) * (0.75 + (i % 3) * 0.2);
      const x = W * (0.2 + 0.18 * i) + Math.sin(sec * 0.15 + i * 1.7) * W * 0.08;
      const y = H * (0.3 + 0.12 * ((i * 3) % 5)) + Math.cos(sec * 0.12 + i) * H * 0.05;
      ctx.globalAlpha = a * (0.11 + 0.03 * (i % 2));
      ctx.drawImage(fogSprite, x - size / 2, y - size / 2, size, size);
    }
    // volumetric light: soft shafts from above, fanning out behind the logo
    ctx.globalCompositeOperation = "lighter";
    for (const [ang, wid, al] of [[-0.2, 0.34, 0.07], [0.2, 0.34, 0.07], [0, 0.22, 0.05]]) {
      const sway = Math.sin(sec * 0.35 + ang * 9) * 0.03;
      ctx.save();
      ctx.translate(cx, -H * 0.12);
      ctx.rotate(ang + sway);
      const hw = (W * wid) / 2;
      const g = ctx.createLinearGradient(-hw, 0, hw, 0);
      g.addColorStop(0, "rgba(120,170,255,0)");
      g.addColorStop(0.5, `rgba(130,178,255,${al})`);
      g.addColorStop(1, "rgba(120,170,255,0)");
      const v = ctx.createLinearGradient(0, 0, 0, H * 1.1);
      v.addColorStop(0, "rgba(255,255,255,1)");
      v.addColorStop(1, "rgba(255,255,255,0)");
      ctx.globalAlpha = a;
      ctx.fillStyle = g;
      ctx.fillRect(-hw, 0, hw * 2, H * 1.2);
      ctx.restore();
    }
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.42);
    glow.addColorStop(0, "rgba(70,120,210,0.12)");
    glow.addColorStop(1, "rgba(70,120,210,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  };

  const drawDrone = (i: number, x: number, y: number, st: DroneState, speed: number, bank: number, t: number) => {
    const sec = t / 1000;
    const u = D / 2;
    ctx.save();
    ctx.translate(x, y);
    // soft ground shadow, offset away from the light above
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = "#000";
    ctx.beginPath();
    ctx.ellipse(u * 0.18, u * 0.32, u * 0.95, u * 0.55, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.rotate(st.heading);
    ctx.scale(1, 1 - 0.1 * Math.abs(bank));
    ctx.scale(u, u);

    // arms
    ctx.strokeStyle = "#1c273a";
    ctx.lineWidth = 0.13;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const [ax, ay] of [[0.7, 0.7], [0.7, -0.7], [-0.7, 0.7], [-0.7, -0.7]]) {
      ctx.moveTo(0, 0);
      ctx.lineTo(ax, ay);
    }
    ctx.stroke();
    ctx.strokeStyle = "rgba(140,170,215,0.5)";
    ctx.lineWidth = 0.03;
    ctx.stroke();

    // rotors: translucent disc, rim, and two spinning blades
    [[0.7, 0.7], [0.7, -0.7], [-0.7, 0.7], [-0.7, -0.7]].forEach(([rx, ry], k) => {
      const g = ctx.createRadialGradient(rx, ry, 0, rx, ry, 0.46);
      g.addColorStop(0, "rgba(190,215,255,0.04)");
      g.addColorStop(1, "rgba(190,215,255,0.16)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(rx, ry, 0.46, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(215,232,255,0.85)";
      ctx.lineWidth = 0.03;
      ctx.stroke();
      const ang = sec * 70 * (k % 2 ? 1 : -1) + k;
      ctx.strokeStyle = "rgba(235,243,255,0.7)";
      ctx.lineWidth = 0.035;
      ctx.beginPath();
      for (let b = 0; b < 2; b++) {
        const aa = ang + b * Math.PI;
        ctx.moveTo(rx, ry);
        ctx.lineTo(rx + Math.cos(aa) * 0.44, ry + Math.sin(aa) * 0.44);
      }
      ctx.stroke();
      ctx.fillStyle = "#0d1522";
      ctx.beginPath();
      ctx.arc(rx, ry, 0.08, 0, Math.PI * 2);
      ctx.fill();
    });

    // fuselage
    const body = ctx.createLinearGradient(0, -0.2, 0, 0.2);
    body.addColorStop(0, "#34465f");
    body.addColorStop(1, "#111a29");
    ctx.fillStyle = body;
    ctx.strokeStyle = "rgba(150,180,225,0.75)";
    ctx.lineWidth = 0.03;
    ctx.beginPath();
    ctx.roundRect(-0.38, -0.19, 0.78, 0.38, 0.17);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "rgba(170,200,245,0.35)";
    ctx.beginPath();
    ctx.ellipse(-0.04, -0.06, 0.22, 0.06, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#050a14";
    ctx.beginPath();
    ctx.arc(0.36, 0, 0.075, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#6fb4ff";
    ctx.beginPath();
    ctx.arc(0.375, -0.02, 0.02, 0, Math.PI * 2);
    ctx.fill();

    // navigation lights: red only, with a slow pulse on the tail
    ctx.globalCompositeOperation = "lighter";
    const strobe = (sec * 1.1 + i * 0.37) % 1 < 0.09;
    const lights: [number, number, string, number][] = [
      [-0.7, -0.7, "255,60,60", 1],
      [-0.7, 0.7, "255,60,60", 1],
      [-0.42, 0, "255,60,60", strobe ? 1 : 0.2],
    ];
    for (const [lx, ly, rgb, a] of lights) {
      const g = ctx.createRadialGradient(lx, ly, 0, lx, ly, 0.3);
      g.addColorStop(0, `rgba(${rgb},${0.9 * a})`);
      g.addColorStop(1, `rgba(${rgb},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(lx, ly, 0.3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
  };

  const drawSolidLogo = (a: number, z: number, glow: number) => {
    if (a <= 0) return;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(S * z, S * z);
    ctx.translate(-LOGO_CX, -LOGO_CY);
    ctx.globalAlpha = a;
    const g = ctx.createLinearGradient(90, 146, 722, 612);
    g.addColorStop(0, "#ffffff");
    g.addColorStop(1, "#e9eef6");
    ctx.shadowColor = "rgba(150,195,255,0.85)";
    ctx.shadowBlur = z > 1.4 ? 0 : glow * 0.45;
    ctx.fillStyle = g;
    ctx.fill(logoPath);
    ctx.restore();
  };

  // PINAKA wordmark under the logo: light, wide-set, slanted caps. Drawn per letter so it can pop in letter by letter.
  const WORD = "PINAKA";
  const WORD_W = 600; // logo units
  const WORD_TOP = 752;
  const drawWordmark = (a: number, z: number, glow: number, pop: number) => {
    if (a <= 0.003 || pop <= 0.003) return;
    const fs = 100;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(S * z, S * z);
    ctx.translate(-LOGO_CX, -LOGO_CY);
    ctx.font = `300 ${fs}px "Helvetica Neue", Helvetica, Arial, sans-serif`;
    ctx.textBaseline = "alphabetic";
    const widths = [...WORD].map((ch) => ctx.measureText(ch).width * 1.12);
    const gap = fs * 0.5;
    const total = widths.reduce((x, y) => x + y, 0) + gap * (WORD.length - 1);
    const k = WORD_W / total;
    const cap = fs * 0.72 * k;
    ctx.translate(LOGO_CX - WORD_W / 2, WORD_TOP + cap);
    ctx.scale(k, k);
    ctx.shadowColor = "rgba(150,195,255,0.8)";
    ctx.shadowBlur = z > 1.4 ? 0 : (glow * 0.45) / k;
    ctx.fillStyle = "#f2f5fa";
    let x = 0;
    [...WORD].forEach((ch, i) => {
      const lp = easeOutCubic(clamp01(pop * 1.7 - i * 0.12));
      if (lp > 0.003) {
        ctx.globalAlpha = a * lp;
        ctx.save();
        ctx.translate(x, (1 - lp) * 14);
        ctx.transform(1.12, 0, -0.2 * 1.12, 1, 0, 0);
        ctx.fillText(ch, 0, 0);
        ctx.restore();
      }
      x += widths[i] + gap;
    });
    ctx.restore();
  };

  // ---------------------------------------------------------------- main loop
  const T = lite ? LITE : FULL;
  const t0 = performance.now();
  let last = t0;
  let raf = 0;
  let revealed = false;
  let finished = false;
  let stopped = false;
  let frames = 0;
  let frameSum = 0;
  let bgCleared = false;

  const doReveal = () => {
    if (revealed) return;
    revealed = true;
    cb.onReveal();
  };
  const finish = () => {
    if (finished) return;
    finished = true;
    cb.onDone();
  };

  const frame = (now: number) => {
    if (stopped) return;
    const t = now - t0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;

    // drop to a lighter effect budget if the first frames are slow
    if (frames < 45 && frames > 5) {
      frameSum += dt;
      if (frames === 44 && frameSum / 39 > 0.034) quality = 0.5;
    }
    frames++;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, W, H);

    const zp = clamp01((t - T.zoomA) / T.zoomDur);
    const zoomE = easeInOutQuad(zp);
    const z = Math.exp(Math.log(ZOOM_MAX) * zoomE);
    const fade = smooth(0.5, 1, zp);
    const bgA = 1 - smooth(0.25, 0.85, zp);

    if (zp > 0 && !bgCleared) {
      bgCleared = true;
      if (overlay) overlay.style.background = "transparent";
    }
    ctx.globalAlpha = bgA;
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;

    if (t >= T.reveal) doReveal();

    if (lite) {
      const a = smooth(T.crossA, T.crossB, t) * (1 - fade);
      drawSolidLogo(a, z, 24 + 60 * zp);
      drawWordmark(1 - fade, z, 18 + 50 * zp, smooth(T.textA, T.textB, t));
    } else {
      const F = FULL;
      drawFog(t, smooth(0, F.fog, t) * (1 - fade));

      // drones: positions, heading, banking, smoke stamping and trails
      const pos = [0, 1, 2].map((i) => dronePos(i, t));
      pos.forEach((p, i) => {
        const st = drones[i];
        if (!p.vis) {
          st.px = p.x;
          st.py = p.y;
          return;
        }
        if (st.py === -999 || st.py < -D * 2.5) {
          st.px = p.x;
          st.py = p.y;
        }
        const vx = dt > 0 ? (p.x - st.px) / dt : 0;
        const vy = dt > 0 ? (p.y - st.py) / dt : 0;
        const speed = Math.hypot(vx, vy);
        if (speed > 25) {
          const target = Math.atan2(vy, vx);
          let d = target - st.heading;
          d = Math.atan2(Math.sin(d), Math.cos(d));
          st.heading += d * Math.min(1, dt * 7);
        }
        const bank = clamp01(Math.abs(Math.sin(st.heading - Math.atan2(vy, vx || 1e-6))) * 2) * (speed > 60 ? 1 : 0);
        st.px = p.x;
        st.py = p.y;
        (st as DroneState & { speed?: number; bank?: number }).speed = speed;
        (st as DroneState & { speed?: number; bank?: number }).bank = bank;

        // smoke written into the logo along the route (deterministic stamps, replayed on resize)
        if (p.drawing) {
          const u = (t - F.arrive) / (F.drawEnd[i] - F.arrive);
          const sNow = easeInOutSine(clamp01(u)) * routeLen[i];
          for (let s = st.lastS; s < sNow; s += 3.5) {
            const r = ROUTES[i][Math.min(routeLen[i], Math.floor(s))];
            for (let n = 0; n < 3; n++) {
              const ang = rng() * Math.PI * 2;
              const off = Math.sqrt(rng()) * r.r * 0.7;
              const stamp: Stamp = { x: r.x + Math.cos(ang) * off, y: r.y + Math.sin(ang) * off, r: r.r * (0.9 + rng() * 0.45), a: 0.09 + rng() * 0.07, v: (rng() * sprites.length) | 0 };
              stamps.push(stamp);
              stampToLayer(stamp);
            }
          }
          st.lastS = Math.max(st.lastS, sNow);
        }
      });

      // lingering atmospheric motion inside the finished smoke
      if (t > F.drawEnd[2] && t < F.crossB && quality > 0.6) {
        const poly = POLYS[(rng() * 3) | 0];
        const [px, py] = poly[(rng() * poly.length) | 0];
        const stamp: Stamp = { x: px + (LOGO_CX - px) * rng() * 0.3, y: py + (LOGO_CY - py) * rng() * 0.3, r: 26, a: 0.06, v: (rng() * sprites.length) | 0 };
        stamps.push(stamp);
        stampToLayer(stamp);
      }

      // the smoke logo, clipped to the true silhouette; drifts a touch once formed
      const solid = smooth(F.crossA, F.crossB, t);
      const smokeA = (1 - smooth(F.crossA + 150, F.crossB, t)) * (1 - fade);
      if (smokeA > 0 && t > F.arrive) {
        tctx.globalCompositeOperation = "source-over";
        tctx.globalAlpha = 1;
        tctx.clearRect(0, 0, tmp.width, tmp.height);
        tctx.drawImage(layer, 0, 0);
        tctx.globalCompositeOperation = "destination-in";
        tctx.drawImage(mask, 0, 0);
        if (cloudPattern) {
          // two drifting octaves carve soft gaps into the smoke; they fade out as it resolves to solid
          tctx.globalCompositeOperation = "destination-out";
          tctx.globalAlpha = 0.9 * (1 - smooth(F.crossA, F.crossB, t) * 0.6);
          for (const [sc2, vx, vy] of [[1.6 * kl, 14, 9], [0.8 * kl, -9, 12]]) {
            tctx.save();
            tctx.translate((t / 1000) * vx * kl, (t / 1000) * vy * kl);
            tctx.scale(sc2, sc2);
            tctx.fillStyle = cloudPattern;
            tctx.fillRect(-300, -300, tmp.width / sc2 + 600, tmp.height / sc2 + 600);
            tctx.restore();
          }
          tctx.globalAlpha = 1;
        }
        const formed = smooth(F.drawEnd[0], F.drawEnd[2] + 400, t);
        const wob = formed * Math.sin(t / 700) * 1.6;
        const sc = 1 + formed * Math.sin(t / 900) * 0.004;
        const w = (BOX.x1 - BOX.x0) * S * sc * z;
        const h = (BOX.y1 - BOX.y0) * S * sc * z;
        const left = cx + (BOX.x0 - LOGO_CX) * S * sc * z + wob;
        const top = cy + (BOX.y0 - LOGO_CY) * S * sc * z + wob * 0.6;
        ctx.globalAlpha = smokeA;
        ctx.drawImage(tmp, left, top, w, h);
        ctx.globalAlpha = 1;
      }
      drawSolidLogo(solid * (1 - fade), z, 22 + 70 * zp + 24 * Math.sin(Math.PI * solid) * (1 - zp));

      drawWordmark(1 - fade, z, 14 + 50 * zp, smooth(F.textA, F.textB, t));

      pos.forEach((p, i) => {
        if (!p.vis) return;
        const st = drones[i] as DroneState & { speed?: number; bank?: number };
        drawDrone(i, p.x, p.y, st, st.speed ?? 0, st.bank ?? 0, t);
      });
    }

    // soft bloom while the camera pushes through the mark
    if (zp > 0 && zp < 1) {
      const bloom = Math.sin(Math.PI * zp) * 0.35;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.7);
      g.addColorStop(0, `rgba(190,220,255,${bloom})`);
      g.addColorStop(1, "rgba(190,220,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
    }

    if (t >= T.zoomA + T.zoomDur) {
      finish();
      return;
    }
    raf = requestAnimationFrame(frame);
  };

  const onResize = () => layout();
  window.addEventListener("resize", onResize);
  raf = requestAnimationFrame(frame);

  return {
    skip() {
      if (stopped || finished) return;
      stopped = true;
      cancelAnimationFrame(raf);
      doReveal();
      if (overlay) {
        overlay.style.transition = "opacity 320ms ease";
        overlay.style.opacity = "0";
      }
      window.setTimeout(finish, 340);
      window.removeEventListener("resize", onResize);
    },
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    },
  };
}
