import { POLYS } from "@/components/drone/intro/logo";

/** The ARJUNA mark: centre spike and two claws, from the same geometry the intro draws. Colour comes from `currentColor`. */
export function LogoMark({ height = 16, className }: { height?: number; className?: string }) {
  return (
    <svg className={className} height={height} width={(height * 632) / 657} viewBox="90 38 632 657" aria-hidden="true">
      {POLYS.map((poly, i) => (
        <polygon key={i} points={poly.map(([x, y]) => `${x},${y}`).join(" ")} fill="currentColor" />
      ))}
    </svg>
  );
}
