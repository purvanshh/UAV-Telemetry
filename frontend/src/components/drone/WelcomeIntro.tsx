"use client";

import { useEffect, useRef, useState } from "react";
import { runIntro, type IntroHandle } from "@/components/drone/intro/engine";

export const INTRO_KEY = "arjuna:welcomed";
const STAGGER_MS = 1300;

/**
 * Runs before first paint (in <head>): play the intro once per browser session, never with reduced motion.
 * It only sets an attribute; CSS hides the app under the overlay and staggers it in. React does not own this attribute.
 */
export const introScript = `try{if(!sessionStorage.getItem("${INTRO_KEY}")&&!matchMedia("(prefers-reduced-motion: reduce)").matches){document.documentElement.setAttribute("data-intro","playing")}}catch(e){}`;

/** Weak devices get a short, effect-free version: the mark fades in and the camera pushes through it. */
function isLowPower() {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return (nav.hardwareConcurrency ?? 8) <= 2 || (nav.deviceMemory ?? 8) <= 2;
}

function markWelcomed() {
  try {
    sessionStorage.setItem(INTRO_KEY, "1");
  } catch {
    /* storage unavailable: it may play again next load */
  }
}

/** Three UAVs write the ARJUNA mark in smoke, then the camera zooms through it into the app. Click or any key skips. */
export function WelcomeIntro() {
  const [done, setDone] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handle = useRef<IntroHandle | null>(null);

  useEffect(() => {
    const html = document.documentElement;
    const canvas = canvasRef.current;
    if (html.getAttribute("data-intro") !== "playing" || !canvas) {
      setDone(true);
      return;
    }
    handle.current = runIntro(
      canvas,
      {
        onReveal: () => {
          html.setAttribute("data-intro", "revealing");
          markWelcomed();
          // not tied to this effect: the stagger must finish even after the overlay unmounts
          window.setTimeout(() => html.removeAttribute("data-intro"), STAGGER_MS);
        },
        onDone: () => setDone(true),
      },
      { lite: isLowPower() },
    );
    const skip = () => handle.current?.skip();
    window.addEventListener("keydown", skip, { once: true });
    return () => {
      handle.current?.stop();
      window.removeEventListener("keydown", skip);
    };
  }, []);

  if (done) return null;
  return (
    <div className="intro" aria-hidden="true" onClick={() => handle.current?.skip()}>
      <canvas ref={canvasRef} className="intro-canvas" />
    </div>
  );
}
