/**
 * The animation catalog — design system template §6, kept as ONE shared
 * style block rather than letting individual components redeclare
 * `@keyframes` under slightly different names (the template calls out its
 * own `goldPulse` existing in two places for exactly that avoidable
 * reason). Mounted once, at the app root.
 *
 * Rules of thumb these follow:
 * - Loading pulses use `ease`, never `linear` — a linear pulse reads as
 *   mechanical rather than alive.
 * - A "pop into place" uses overshoot easing (cubic-bezier(0.34,1.56,0.64,1))
 *   so it feels like it has weight; a plain fade-in never does.
 * - Colour/background transitions default to 200-300ms; layout-affecting
 *   transitions (width, height) get 700ms.
 */

import { C, F } from "./tokens";

export const SPRING = "cubic-bezier(0.34,1.56,0.64,1)";

export function Animations() {
  return (
    <style>{`
      /* Word-by-word drop-in for a hero headline — staggered per word via
         animation-delay, not a single fade on the whole line. Reads as the
         headline being TYPESET in front of you rather than just appearing. */
      @keyframes wordDrop {
        from { opacity: 0; transform: translateY(-36px); filter: blur(5px); }
        to   { opacity: 1; transform: translateY(0);      filter: blur(0);   }
      }

      /* The generic "this block just entered" animation. */
      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0);     }
      }

      /* A stat/number "landing" with a slight overshoot. */
      @keyframes statPop {
        from { opacity: 0; transform: scale(0.85) translateY(12px); }
        to   { opacity: 1; transform: scale(1)    translateY(0);     }
      }

      /* Ambient pulsing glow — a "still working" heartbeat, or a CTA that
         should breathe without being obnoxious. */
      @keyframes accentPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(225,6,0,0.28); }
        50%      { box-shadow: 0 0 0 10px rgba(225,6,0,0); }
      }

      /* A tiny vertical nudge + opacity pulse for a "scroll for more" affordance. */
      @keyframes scrollNudge {
        0%, 100% { opacity: 0.3; transform: scaleY(0.8); }
        50%      { opacity: 1;   transform: scaleY(1.1); }
      }

      /* A progress/percentile bar growing in from zero width on first render. */
      @keyframes barFill { from { width: 0; } }

      /* A slow ambient opacity breathe for a background decoration. */
      @keyframes ambientPulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.9; } }

      /* A location marker "radar ping" — for "here's the thing I want your
         eye on" on a track map, chart, or canvas overlay. */
      @keyframes pulseRing {
        0%, 100%  { opacity: 1; transform: translate(-50%,-50%) scale(1);   box-shadow: 0 0 8px currentColor; }
        50%       { opacity: 0; transform: translate(-50%,-50%) scale(1.6); box-shadow: 0 0 24px currentColor; }
      }

      /* Base resets kept here so there is exactly one global style source. */
      * { box-sizing: border-box; }
      html, body, #root { margin: 0; padding: 0; min-height: 100%; }
      body {
        background: ${C.bg};
        color: ${C.text};
        font-family: ${F.body};
        -webkit-font-smoothing: antialiased;
      }
      select, option, button { font-family: ${F.body}; }
      /* Scrollbar, themed to the light page rather than left as the OS default. */
      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-track { background: ${C.bg}; }
      ::-webkit-scrollbar-thumb { background: #CACAD2; border-radius: 5px; }
      ::-webkit-scrollbar-thumb:hover { background: #A9A9B3; }

      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
        }
      }
    `}</style>
  );
}
