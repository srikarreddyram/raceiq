/**
 * Brand chrome — design system template §4. A small set of "always-on"
 * decorative elements, added once, that make every page read as one
 * coherent product instead of a pile of independently-styled screens.
 */

import { C } from "./tokens";

// Edges in carbon, not red: the always-on frame should be quiet, with F1
// red saved for the mark and active states.
const BRAND_PRIMARY = C.carbon;
const BRAND_SECONDARY = C.carbon;

/**
 * A persistent colour frame down both edges of the viewport — fixed
 * position (survives scrolling), pointer-events none (never steals a
 * click). The one piece of brand identity every page shares, themed
 * pages included.
 */
export function EdgeBezel() {
  return (
    <>
      <div
        aria-hidden
        style={{
          position: "fixed",
          left: 0,
          top: 0,
          bottom: 0,
          width: 4,
          background: BRAND_PRIMARY,
          opacity: 0.9,
          zIndex: 40,
          pointerEvents: "none",
        }}
      />
      <div
        aria-hidden
        style={{
          position: "fixed",
          right: 0,
          top: 0,
          bottom: 0,
          width: 4,
          background: BRAND_SECONDARY,
          opacity: 0.9,
          zIndex: 40,
          pointerEvents: "none",
        }}
      />
    </>
  );
}

/**
 * Default ambient page wash for anywhere with no specific entity to theme
 * against — corner-anchored radial glows, never a full-bleed gradient
 * (which would fight the content for attention).
 */
export function DefaultWash() {
  return (
    <div aria-hidden style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0 }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(65% 55% at 8% 0%, ${BRAND_PRIMARY}14 0%, transparent 62%)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(65% 55% at 92% 0%, ${BRAND_SECONDARY}0D 0%, transparent 62%)`,
        }}
      />
    </div>
  );
}

/** A three-band brand stripe under a page header — cheap, instantly recognisable. */
export function HeaderStripe({ bottom = 0 }: { bottom?: number | string }) {
  return (
    <div
      aria-hidden
      style={{ position: "absolute", left: 0, right: 0, bottom, height: 4, display: "flex" }}
    >
      <div style={{ flex: 3, background: BRAND_SECONDARY }} />
      <div style={{ flex: 1, background: C.surface }} />
      <div style={{ flex: 1, background: BRAND_PRIMARY }} />
    </div>
  );
}
