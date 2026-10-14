/** A tiny inline trend line: plain SVG, scales to its box, draws nothing until
 * there are two points. Nulls are skipped; a series that never moved is drawn as a faint dotted
 * baseline so it reads as "steady" rather than as a divider. Decorative unless given a label. */
import { cn } from "@/lib/utils";

export function Sparkline({
  values,
  className,
  area = false,
  label,
}: {
  values: readonly (number | null)[];
  className?: string;
  area?: boolean;
  label?: string;
}) {
  const points = values.flatMap((v, i) => (v === null ? [] : [{ i, v }]));
  if (points.length < 2) return <div className={className} aria-hidden />;
  const min = Math.min(...points.map((p) => p.v));
  const max = Math.max(...points.map((p) => p.v));
  const span = max - min || 1;
  const flat = max === min;
  const last = Math.max(values.length - 1, 1);
  const coords = points.map(({ i, v }) => {
    const x = (i / last) * 100;
    // Flat series sit mid-height rather than on the floor.
    const y = flat ? 50 : 92 - ((v - min) / span) * 84;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const line = coords.join(" ");
  const first = coords[0]!.split(",")[0];
  const end = coords[coords.length - 1]!.split(",")[0];
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className={cn("text-accent overflow-visible", className)}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {area && !flat && (
        <polygon points={`${first},100 ${line} ${end},100`} className="fill-current opacity-10" />
      )}
      <polyline
        points={line}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeDasharray={flat ? "1 4" : undefined}
        opacity={flat ? 0.45 : 1}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
