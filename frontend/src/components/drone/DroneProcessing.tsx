import type { ReactNode } from "react";
import { QuadSide } from "@/components/drone/Drone";

/**
 * Work in progress, shown as a drone flying a route with waypoints. It keeps making passes while the work runs;
 * with ``leaving`` it climbs out of the frame so the result can take its place.
 */
export function DroneProcessing({ title, detail, leaving, compact }: {
  title: string;
  detail?: ReactNode;
  leaving?: boolean;
  compact?: boolean;
}) {
  return (
    <div className={`dproc${compact ? " is-compact" : ""}${leaving ? " is-leaving" : ""}`} role="status" aria-live="polite">
      <div className="dproc-track" aria-hidden="true">
        <span className="dproc-route" />
        <span className="dproc-wp" style={{ left: "12%" }} />
        <span className="dproc-wp" style={{ left: "50%" }} />
        <span className="dproc-wp" style={{ left: "88%" }} />
        <span className="dproc-fly">
          <span className="dproc-tilt">
            <QuadSide width={compact ? 46 : 70} />
          </span>
        </span>
      </div>
      <div className="dproc-text">
        <span className="dproc-title" key={title}>
          {title}
        </span>
        {detail && <span className="dproc-detail">{detail}</span>}
      </div>
    </div>
  );
}
