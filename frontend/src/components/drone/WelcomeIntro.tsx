"use client";

import { useEffect, useState } from "react";
import { QuadSide } from "@/components/drone/Drone";

export const INTRO_KEY = "arjuna:welcomed";
const INTRO_MS = 2900;

/**
 * Runs before first paint (in <head>): play the welcome once per browser session, never with reduced motion.
 * It only sets an attribute; CSS shows the overlay and staggers the app in. React does not own this attribute.
 */
export const introScript = `try{if(!sessionStorage.getItem("${INTRO_KEY}")&&!matchMedia("(prefers-reduced-motion: reduce)").matches){document.documentElement.setAttribute("data-intro","playing")}}catch(e){}`;

/** A drone flies in, says hello, climbs away; the app rises in behind it. Click or any key skips it. */
export function WelcomeIntro() {
  const [done, setDone] = useState(false);

  useEffect(() => {
    const html = document.documentElement;
    if (html.getAttribute("data-intro") !== "playing") {
      setDone(true);
      return;
    }
    const finish = () => {
      html.removeAttribute("data-intro");
      try {
        sessionStorage.setItem(INTRO_KEY, "1");
      } catch {
        /* storage unavailable: it may play again next load */
      }
      setDone(true);
    };
    const t = window.setTimeout(finish, INTRO_MS);
    const skip = () => {
      window.clearTimeout(t);
      finish();
    };
    window.addEventListener("keydown", skip, { once: true });
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("keydown", skip);
    };
  }, []);

  if (done) return null;
  return (
    <div
      className="intro"
      aria-hidden="true"
      onClick={() => {
        document.documentElement.removeAttribute("data-intro");
        try {
          sessionStorage.setItem(INTRO_KEY, "1");
        } catch {
          /* ignore */
        }
        setDone(true);
      }}
    >
      <div className="intro-stage">
        <div className="intro-drone">
          <div className="intro-tilt">
            <QuadSide width={120} />
          </div>
        </div>
        <div className="intro-text">
          <div className="intro-title">Welcome to ARJUNA</div>
          <div className="intro-sub">UAV flight analysis, from log to verdict.</div>
        </div>
      </div>
    </div>
  );
}
