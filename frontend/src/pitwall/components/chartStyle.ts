/**
 * Recharts styling shared by every Pit Wall chart, so axes, grids and
 * tooltips stay recessive and identical across views.
 *
 * ACCENT is the CSS variable, not a hex: SVG presentation attributes
 * resolve `var(--rq-accent)`, so a chart inside a team-themed subtree
 * (theme.ts's accentVars) draws in that team's colour with no prop drilling.
 */

import { C, F } from "../../design/tokens";

export const AXIS_TICK = { fill: C.faint, fontSize: 11, fontFamily: F.mono, fontWeight: 600 };
export const AXIS_LINE = { stroke: C.edge };
export const GRID_STROKE = C.rule;
export const ACCENT = "var(--rq-accent)";
export const TOOLTIP_STYLE = {
  background: C.surface,
  border: `1px solid ${C.edge}`,
  borderRadius: 6,
  boxShadow: "0 4px 16px rgba(21,21,30,0.12)",
  fontFamily: F.body,
  fontSize: 12,
  color: C.text,
};
