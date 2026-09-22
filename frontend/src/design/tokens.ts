/**
 * Design tokens — one object per concern, imported everywhere instead of
 * re-typed. A page that reaches for `C.bg` and `F.mono` instead of a hex
 * string and "monospace" cannot drift from the system by typo.
 *
 * The palette is themed after the official F1 app: light surfaces, carbon
 * text and chrome, F1 red as the brand accent, constructor colours for
 * anything team-scoped (see theme.ts). It replaced a near-black, gold
 * palette that read as too dark and moody for a sport this loud. The
 * design system template's PATTERNS are unchanged — only its values are.
 *
 * Every translucent colour is a tint of carbon (#15151E), not of black or
 * white, so rules and fills sit in the same colour family as the text.
 */

import type { CSSProperties } from "react";

export const C = {
  bg: "#F2F2F4", // page background — the F1 app's light grey, not white, so white cards lift off it
  surface: "#FFFFFF", // card background
  raised: "#F6F6F8", // an input or nested panel inside a card
  carbon: "#15151E", // F1's carbon black — primary text and dark chrome
  rule: "rgba(21,21,30,0.07)", // hairline dividers
  edge: "rgba(21,21,30,0.12)", // card and control borders
  line: "rgba(21,21,30,0.24)", // zero lines, reference lines, crosshairs
  fill: "rgba(21,21,30,0.06)", // neutral fills — bar tracks, ranges, row hover
  inactive: "#B8B8C2", // marks that aren't the point of the chart
  text: "#15151E", // primary text
  dim: "rgba(21,21,30,0.74)", // secondary text
  faint: "rgba(21,21,30,0.52)", // tertiary text, axis ticks
  muted: "#63636E", // captions, disabled states
  accent: "var(--rq-accent)", // the ACTIVE accent — F1 red by default, a team's colour on team pages
  hover: "var(--rq-hover)", // accent tint for hover / selected fills
  red: "#C8102E", // danger / not classified — always shipped with a symbol and a word
  amber: "#D98E04", // warning — the yellow flag
  green: "#0B8A4B", // good / positions gained — dark enough to read on white
} as const;

export const F = {
  display: "'Titillium Web', 'Arial Narrow', sans-serif", // F1's longtime brand face; see DISPLAY
  body: "'Titillium Web', sans-serif",
  mono: "'Titillium Web', sans-serif", // labels and eyebrows: the F1 app sets these in its sans, in caps
  num: "'Roboto Mono', monospace", // numbers that line up in columns
} as const;

/**
 * Headlines and big numbers. Titillium has real weights, unlike the
 * single-weight display face it replaced, so the weight and caps have to
 * be asked for — every display use spreads this rather than setting
 * `fontFamily: F.display` alone.
 */
export const DISPLAY: CSSProperties = {
  fontFamily: F.display,
  fontWeight: 900,
  textTransform: "uppercase",
  letterSpacing: "0.005em",
};

/**
 * Numerals in a mono face still aren't guaranteed to line up in columns
 * unless you ask for it explicitly.
 */
export const NUM: CSSProperties = {
  fontFamily: F.num,
  fontVariantNumeric: "tabular-nums",
};

/** Compound colours — the one place F1 domain meaning overrides the palette. */
// Pirelli's red / yellow / white, adjusted to read on a white card: the
// medium is a deeper yellow and the hard a mid grey (white on white is
// invisible). Validated with the dataviz palette script — CVD and
// normal-vision separation pass; the medium sits just under 3:1 contrast,
// so charts using these direct-label their series.
export const COMPOUND_COLORS: Record<string, string> = {
  SOFT: "#D6231E",
  MEDIUM: "#C28F00",
  HARD: "#7C7C86",
  INTERMEDIATE: "#1E9E4A",
  WET: "#1F5FD1",
};

export function compoundColor(compound: string | null | undefined): string {
  if (!compound) return C.muted;
  return COMPOUND_COLORS[compound.toUpperCase()] ?? C.muted;
}
