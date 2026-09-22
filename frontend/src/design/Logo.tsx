/**
 * The RaceIQ mark and wordmark.
 *
 * The mark is three forward-leaning bars rising left to right — a
 * strategy chart and the slant of F1's own speed lines in one shape. The
 * tallest bar is F1 red; the others are carbon (or white on a dark
 * surface). The wordmark is heavy italic Titillium, "RACE" in the
 * surface's ink and "IQ" in red — red used as F1 uses it, as the one hot
 * note, not a fill.
 */

import { C, F } from "./tokens";
import { DEFAULT_ACCENT } from "./theme";

export function LogoMark({ size = 28, onDark = false }: { size?: number; onDark?: boolean }) {
  const ink = onDark ? "#FFFFFF" : C.carbon;
  // Parallelograms on a 32x32 grid, every bar leaning the same 16 degrees
  // (its lean scales with its height), 6 units wide.
  const lean = Math.tan((16 * Math.PI) / 180);
  const bar = (x: number, h: number, fill: string) => {
    const dx = h * lean;
    return <path d={`M${x + dx} ${32 - h} L${x + dx + 6} ${32 - h} L${x + 6} 32 L${x} 32 Z`} fill={fill} />;
  };
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      {bar(0.5, 14, ink)}
      {bar(8.5, 22, ink)}
      {bar(16.5, 32, DEFAULT_ACCENT)}
    </svg>
  );
}

export function Logo({ height = 28, onDark = false }: { height?: number; onDark?: boolean }) {
  const ink = onDark ? "#FFFFFF" : C.carbon;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: height * 0.3 }} aria-label="RaceIQ">
      <LogoMark size={height} onDark={onDark} />
      <span
        style={{
          fontFamily: F.display,
          fontWeight: 900,
          fontStyle: "italic",
          fontSize: height * 0.95,
          letterSpacing: "-0.01em",
          lineHeight: 1,
          color: ink,
        }}
      >
        RACE<span style={{ color: DEFAULT_ACCENT }}>IQ</span>
      </span>
    </span>
  );
}
