/**
 * Dynamic theming via CSS custom properties — design system template §2.
 *
 * The whole app can re-skin to a specific colour (a constructor's livery,
 * a compound, a status) by setting four CSS variables on a wrapping
 * element: no prop drilling, no context provider, no re-render of the
 * subtree's own styles. Every component that wants "the current accent"
 * reads `var(--rq-accent)` (i.e. `C.gold`) rather than a passed-in prop.
 */

import type { CSSProperties } from "react";

/** PRD Section 13.1's named accent — RaceIQ's baseline brand colour. */
export const DEFAULT_ACCENT = "#C9A84C";

export function accentVars(accent: string): CSSProperties {
  return {
    // 8-digit hex: the last byte IS the alpha channel, so one base colour
    // yields a whole tint set with no colour-math library.
    "--rq-accent": accent,
    "--rq-edge": `${accent}1A`, // ~10% — hairline borders
    "--rq-edge-strong": `${accent}3D`, // ~24% — emphasised borders
    "--rq-hover": `${accent}14`, // ~8%  — hover fills
    "--rq-on-accent": readableOn(accent), // legible text COLOUR on top of the accent
  } as CSSProperties;
}

/**
 * Not optional when theming against a palette you don't control. Real F1
 * constructor colours span Haas white and Red Bull navy — fixed
 * white-on-accent fails the light ones, fixed black-on-accent fails the
 * dark ones. Compute it.
 */
export function readableOn(hex: string): string {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return L > 0.45 ? "#000" : "#fff"; // WCAG relative luminance
}

/**
 * Real 2026-grid constructor colours, used to theme any view scoped to a
 * single team (see §2's "a page with a specific entity in play overrides
 * the default wash with that entity's own colour"). Keys are this
 * project's own `team_id` values from silver.constructors.
 */
export const TEAM_COLORS: Record<string, string> = {
  mclaren: "#FF8000",
  ferrari: "#E8002D",
  red_bull: "#3671C6",
  mercedes: "#27F4D2",
  aston_martin: "#229971",
  alpine: "#FF87BC",
  williams: "#64C4FF",
  rb: "#6692FF",
  sauber: "#52E252",
  haas: "#B6BABD",
  audi: "#BB0A30",
  cadillac: "#C9A84C",
};

export function teamAccent(teamId: string | null | undefined): string {
  if (!teamId) return DEFAULT_ACCENT;
  return TEAM_COLORS[teamId] ?? DEFAULT_ACCENT;
}
