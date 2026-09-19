/**
 * Core primitives — design system template §3. Small, composable, and
 * every one of them reads its colours from tokens.ts, never a literal.
 */

import type { CSSProperties, ReactNode } from "react";
import { C, F, NUM } from "./tokens";

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
        // An accent is a left-border stripe, not a full-card fill —
        // reserves saturated colour for something that actually needs to
        // draw the eye, keeps the base case quiet.
        borderLeft: accent ? `3px solid ${accent}` : `1px solid ${C.edge}`,
        borderRadius: 6,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/**
 * The eyebrow. Wide letter-spacing + mono + small size + accent colour is
 * the one recurring "this labels a block below it" signature used
 * everywhere — reuse this exact recipe rather than inventing a new label
 * style per screen.
 */
export function SectionLabel({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        fontFamily: F.mono,
        fontSize: 10,
        color: C.gold,
        letterSpacing: "0.4em",
        textTransform: "uppercase",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export type SegmentedOption = { value: string; label: string };

/**
 * The two/three-way toggle. The active option switches TYPEFACE (mono →
 * display) as well as colour — "which one is active" should be readable
 * from 20 feet away, not just from a slightly-different grey.
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
        background: C.surface,
        border: `1px solid ${C.edge}`,
        borderRadius: 4,
        padding: 3,
        gap: 3,
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
              background: active ? C.gold : "transparent",
              color: active ? "var(--rq-on-accent, #000)" : C.dim,
              border: "none",
              borderRadius: 2,
              padding: "7px 18px",
              cursor: "pointer",
              fontFamily: active ? F.display : F.mono,
              fontSize: active ? 15 : 10,
              letterSpacing: active ? "0.06em" : "0.22em",
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
  { min: 0.9, color: "#16A34A", label: "Elite" },
  { min: 0.7, color: "#84CC16", label: "Strong" },
  // The brand accent sits at "average" ON PURPOSE: "merely average" should
  // read as neutral information, not as the app's own celebratory colour
  // being spent on a mediocre result.
  { min: 0.4, color: "#C9A84C", label: "Average" },
  { min: 0.2, color: "#EA580C", label: "Below average" },
  { min: 0, color: "#DC2626", label: "Poor" },
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
        background: "rgba(255,255,255,0.05)",
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
          <div style={{ width: 22, height: 9, background: "rgba(255,255,255,0.05)", borderRadius: 2 }} />
          <div style={{ width: 150, height: 9, background: "rgba(255,255,255,0.07)", borderRadius: 2 }} />
          <div style={{ flex: 1 }} />
          {Array.from({ length: columns }).map((_, j) => (
            <div key={j} style={{ width: 54, height: 9, background: "rgba(255,255,255,0.05)", borderRadius: 2 }} />
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
        fontSize: 11,
        letterSpacing: "0.18em",
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
      <div style={{ fontFamily: F.body, fontWeight: 600, fontSize: 14, color: C.text }}>
        The engine didn't answer
      </div>
      <div style={{ fontFamily: F.mono, fontSize: 11.5, color: C.muted, marginTop: 5, lineHeight: 1.6 }}>
        {message} — check that the RaceIQ API is running (
        <span style={{ color: C.gold }}>uv run uvicorn serving.api.main:app</span>).
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
          fontSize: 9,
          color: C.faint,
          letterSpacing: "0.22em",
          textTransform: "uppercase",
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <div style={{ ...NUM, fontFamily: F.display, fontSize: size, color: color ?? C.text, lineHeight: 1 }}>
          {value}
        </div>
        {unit && <div style={{ fontFamily: F.mono, fontSize: 10, color: C.faint }}>{unit}</div>}
      </div>
    </div>
  );
}
