/**
 * Recharts styling shared by every Pit Wall chart, so axes, grids and
 * tooltips stay recessive and identical across views. Recharts takes raw
 * colour strings in SVG attributes, so these are the literal token values
 * rather than CSS variables.
 */

export const AXIS_TICK = { fill: "rgba(240,240,240,0.38)", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" };
export const AXIS_LINE = { stroke: "rgba(255,255,255,0.08)" };
export const GRID_STROKE = "rgba(255,255,255,0.04)";
export const ACCENT = "#C9A84C";
export const TOOLTIP_STYLE = {
  background: "#16161f",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 4,
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
  color: "#F0F0F0",
};
