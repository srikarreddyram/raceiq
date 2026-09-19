/**
 * Design tokens — one object per concern, imported everywhere instead of
 * re-typed. A page that reaches for `C.bg` and `F.mono` instead of
 * "#0a0a0f" and "monospace" cannot drift from the system by typo.
 *
 * Palette values come from PRD Section 13.1, which names the same
 * background/accent/secondary/danger colours the design system template
 * was built around.
 */

export const C = {
  bg: "#0a0a0f", // page background
  surface: "#0e0e16", // card background, one step up from bg
  raised: "#16161f", // a card floating above another card
  rule: "rgba(255,255,255,0.04)", // hairline dividers — barely-there on purpose
  edge: "rgba(255,255,255,0.08)", // card borders
  text: "#F0F0F0", // primary text (never pure #fff — too harsh on this bg)
  dim: "rgba(240,240,240,0.62)", // secondary text
  faint: "rgba(240,240,240,0.38)", // tertiary / placeholder text
  muted: "#9aa7bd", // captions, disabled states (PRD 13.1's secondary text)
  gold: "var(--rq-accent)", // the ACTIVE accent — dynamic, see theme.ts
  red: "#DC2626",
  green: "#16A34A",
} as const;

export const F = {
  display: "'Bebas Neue', sans-serif", // headlines, big numbers — condensed, loud
  mono: "'JetBrains Mono', monospace", // ALL data, labels, captions, eyebrows
  body: "'Inter', sans-serif", // paragraphs, longer descriptive text
} as const;

/**
 * Numerals in a mono face still aren't guaranteed to line up in columns
 * unless you ask for it explicitly.
 */
export const NUM: React.CSSProperties = {
  fontFamily: F.mono,
  fontVariantNumeric: "tabular-nums",
};

/** Compound colours — the one place F1 domain meaning overrides the palette. */
export const COMPOUND_COLORS: Record<string, string> = {
  SOFT: "#DC2626",
  MEDIUM: "#D4B23C",
  HARD: "#E8E8E8",
  INTERMEDIATE: "#16A34A",
  WET: "#2563EB",
};

export function compoundColor(compound: string | null | undefined): string {
  if (!compound) return C.muted;
  return COMPOUND_COLORS[compound.toUpperCase()] ?? C.muted;
}
