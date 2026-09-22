/**
 * Dynamic theming via CSS custom properties — design system template §2.
 *
 * The whole app can re-skin to a specific colour (a constructor's livery,
 * a compound, a status) by setting four CSS variables on a wrapping
 * element: no prop drilling, no context provider, no re-render of the
 * subtree's own styles. Every component that wants "the current accent"
 * reads `var(--rq-accent)` (i.e. `C.accent`) rather than a passed-in prop.
 */

import type { CSSProperties } from "react";

/** F1 red — the baseline brand colour, as in the official F1 app. */
export const DEFAULT_ACCENT = "#E10600";

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
  return onLight(TEAM_COLORS[teamId] ?? DEFAULT_ACCENT);
}

function luminance(hex: string): number {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/**
 * A team colour made usable as a mark on a white card: darkened in small
 * steps until it clears 3:1 contrast (the minimum for chart marks and
 * large text). Mercedes teal, Haas grey and Sauber green are all far too
 * pale on white as published; Ferrari red and Red Bull navy pass as-is
 * and come back unchanged. Hue is kept, so the team still reads as itself.
 */
export function onLight(hex: string, minContrast = 3): string {
  let [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.replace("#", "").slice(i, i + 2), 16));
  const contrast = () => 1.05 / (luminance(toHex(r, g, b)) + 0.05);
  for (let i = 0; i < 40 && contrast() < minContrast; i++) {
    [r, g, b] = [r, g, b].map((c) => Math.round(c * 0.94));
  }
  return toHex(r, g, b);
}

function toHex(r: number, g: number, b: number): string {
  return `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`.toUpperCase();
}
