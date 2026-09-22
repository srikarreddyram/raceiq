/**
 * Core primitives — design system template §3. Small, composable, and
 * every one of them reads its colours from tokens.ts, never a literal.
 *
 * Styled after the F1 app: white cards with a soft lift off a light grey
 * page, a coloured stripe down the left edge where a card belongs to
 * something (a team, a compound, a status), and bold caps for labels.
 */

import type { CSSProperties, ReactNode } from "react";
import { C, DISPLAY, F, NUM } from "./tokens";

export function Card({
  children,
  accent,
  style,
}: {
  children: ReactNode;
  accent?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        background: C.surface,
        border: `1px solid ${C.edge}`,
        // An accent is a left-border stripe, not a full-card fill — the
        // F1 app's own device for team-coloured cards, and it keeps
        // saturated colour for the one thing that should draw the eye.
        borderLeft: accent ? `4px solid ${accent}` : `1px solid ${C.edge}`,
        borderRadius: 8,
        boxShadow: "0 1px 2px rgba(21,21,30,0.05), 0 2px 8px rgba(21,21,30,0.04)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/**
 * The eyebrow: small bold uppercase carbon type behind a short accent tab.
 * The tab carries the colour (F1 red, or a team's on team pages) so the
 * accent marks every block without painting every label red — which is
 * what made the first light theme read as a Ferrari site.
 */
export function SectionLabel({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontFamily: F.mono,
        fontWeight: 700,
        fontSize: 11.5,
        color: C.text,
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        ...style,
      }}
    >
      <span aria-hidden style={{ width: 4, height: 13, background: C.accent, borderRadius: 1, flex: "none" }} />
      <span>{children}</span>
    </div>
  );
}

export type SegmentedOption = { value: string; label: string };

/**
 * The two/three-way toggle. The active option is a solid accent pill with
 * bolder type — "which one is active" should be readable from across the
 * room, not just from a slightly different grey.
 */
export function Segmented({
  options,
  value,
  onChange,
}: {
  options: SegmentedOption[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div
      style={{
        display: "inline-flex",
        background: C.raised,
        border: `1px solid ${C.edge}`,
        borderRadius: 999,
        padding: 3,
        gap: 2,
      }}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            aria-pressed={active}
            style={{
              background: active ? C.carbon : "transparent",
              color: active ? "#fff" : C.dim,
              border: "none",
              borderRadius: 999,
              padding: "6px 16px",
              cursor: "pointer",
              fontFamily: F.mono,
              fontWeight: active ? 700 : 600,
              fontSize: 11.5,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              transition: "all 200ms",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

const TIERS = [
  { min: 0.9, color: C.green, label: "Elite" },
  { min: 0.7, color: "#5E9E1B", label: "Strong" },
  { min: 0.4, color: C.amber, label: "Average" },
  { min: 0.2, color: "#D9620B", label: "Below average" },
  { min: 0, color: C.red, label: "Poor" },
];

export function tierColor(pct: number | null | undefined): string {
  if (pct == null || Number.isNaN(pct)) return C.muted;
  return (TIERS.find((t) => pct >= t.min) ?? TIERS[TIERS.length - 1]).color;
}

export function tierLabel(pct: number | null | undefined): string {
  if (pct == null || Number.isNaN(pct)) return "No data";
  return (TIERS.find((t) => pct >= t.min) ?? TIERS[TIERS.length - 1]).label;
}

/** A progress/percentile bar's colour IS the data (§9), not a fixed brand colour. */
export function Bar({
  pct,
  color,
  height = 3,
}: {
  pct: number;
  color?: string;
  height?: number;
}) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(pct) ? pct : 0));
  return (
    <div
      style={{
        flex: 1,
        height,
        background: C.fill,
        borderRadius: height / 2,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${Math.max(2, clamped * 100)}%`,
          height: "100%",
          background: color ?? tierColor(clamped),
          borderRadius: height / 2,
          transition: "width 300ms",
        }}
      />
    </div>
  );
}

/**
 * Shape-matched skeleton, not a generic shimmer bar: the row shape mirrors
 * the REAL row it stands in for, and rows fade toward the bottom, which
 * reads as "more content below" rather than a hard cutoff.
 */
export function TableSkeleton({ rows = 10, columns = 4 }: { rows?: number; columns?: number }) {
  return (
    <Card style={{ overflow: "hidden" }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            gap: 16,
            padding: "11px 14px",
            borderBottom: `1px solid ${C.rule}`,
            alignItems: "center",
            opacity: 1 - i * (0.6 / rows),
          }}
        >
          <div style={{ width: 22, height: 9, background: C.fill, borderRadius: 2 }} />
          <div style={{ width: 150, height: 9, background: C.fill, borderRadius: 2 }} />
          <div style={{ flex: 1 }} />
          {Array.from({ length: columns }).map((_, j) => (
            <div key={j} style={{ width: 54, height: 9, background: C.fill, borderRadius: 2 }} />
          ))}
        </div>
      ))}
    </Card>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        padding: "44px 24px",
        textAlign: "center",
        color: C.muted,
        fontFamily: F.mono,
        fontWeight: 600,
        fontSize: 12,
        letterSpacing: "0.08em",
      }}
    >
      {children}
    </div>
  );
}

/** An error state names what a person can actually go check. */
export function ErrorState({ message }: { message: string }) {
  return (
    <Card accent={C.red} style={{ padding: "16px 20px", marginBottom: 16 }}>
      <div style={{ fontFamily: F.body, fontWeight: 700, fontSize: 15, color: C.text }}>
        The engine didn't answer
      </div>
      <div style={{ fontFamily: F.body, fontSize: 13, color: C.muted, marginTop: 5, lineHeight: 1.6 }}>
        {message} — check that the RaceIQ API is running (
        <span style={{ ...NUM, fontSize: 12, color: C.accent }}>uv run uvicorn serving.api.main:app</span>).
      </div>
    </Card>
  );
}

/** A labelled number, the dashboard's most-repeated atom. */
export function Stat({
  label,
  value,
  unit,
  color,
  size = 30,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  color?: string;
  size?: number;
}) {
  return (
    <div>
      <div
        style={{
          fontFamily: F.mono,
          fontWeight: 600,
          fontSize: 10.5,
          color: C.faint,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <div style={{ ...DISPLAY, fontWeight: 700, fontSize: size, color: color ?? C.text, lineHeight: 1 }}>
          {value}
        </div>
        {unit && <div style={{ fontFamily: F.mono, fontWeight: 600, fontSize: 11, color: C.faint }}>{unit}</div>}
      </div>
    </div>
  );
}
