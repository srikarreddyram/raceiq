/**
 * Car — one page per team and season: where the car is fast (straight-line
 * speed against overall pace, and why), what its inferred characteristics
 * are, and how it treats its tyres. It merges what used to be separate
 * Car Profile and Tyre views, under one team picker.
 *
 * The speed section's map places every team on two axes that together say
 * where a car makes its lap time: speed-trap km/h against the field (power
 * against drag) and race pace against the field. The selected team is the
 * only coloured mark; the rest are grey and labelled, so identity never
 * rests on eleven livery colours.
 */

import { useMemo, useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../../api/client";
import type { SpeedProfile, TeamSpeed } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, SectionLabel, Stat, TableSkeleton } from "../../design/primitives";
import { accentVars, teamAccent } from "../../design/theme";
import { C, DISPLAY, F } from "../../design/tokens";
import { AXIS_LINE, AXIS_TICK, GRID_STROKE, TOOLTIP_STYLE } from "../components/chartStyle";
import { TeamPicker, useTeamSelection } from "../components/TeamPicker";
import { CarProfileSection } from "./CarProfileView";
import { TyreSection } from "./TyreView";

const kph = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;
const secs = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;

function CharacterMap({ data, teamId }: { data: SpeedProfile; teamId: string }) {
  const W = 620;
  const H = 400;
  const pad = { l: 56, r: 20, t: 20, b: 44 };
  const xs = data.teams.map((t) => t.top_speed_delta_kph);
  const ys = data.teams.map((t) => t.pace_delta_s);
  const xMax = Math.max(4, ...xs.map(Math.abs)) * 1.15;
  const yMax = Math.max(0.3, ...ys.map(Math.abs)) * 1.15;
  const x = (v: number) => pad.l + ((v + xMax) / (2 * xMax)) * (W - pad.l - pad.r);
  // Faster (negative pace delta) plots UP.
  const y = (v: number) => pad.t + ((v + yMax) / (2 * yMax)) * (H - pad.t - pad.b);
  const sorted = [...data.teams].sort((a, b) => (a.team_id === teamId ? 1 : 0) - (b.team_id === teamId ? 1 : 0));

  // Keep labels from landing on each other: walk the points top to bottom
  // and push a label down whenever it would overlap one already placed.
  const labelY = new Map<string, number>();
  const placed: { x: number; y: number }[] = [];
  for (const t of [...data.teams].sort((a, b) => y(a.pace_delta_s) - y(b.pace_delta_s))) {
    const px = x(t.top_speed_delta_kph);
    let py = y(t.pace_delta_s) + 4;
    while (placed.some((q) => Math.abs(q.x - px) < 90 && Math.abs(q.y - py) < 15)) py += 15;
    placed.push({ x: px, y: py });
    labelY.set(t.team_id, py);
  }

  const quadrant = (label: string, qx: number, qy: number, anchor: "start" | "end") => (
    <text x={qx} y={qy} textAnchor={anchor} fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={11} letterSpacing="0.08em">
      {label}
    </text>
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} role="img" aria-label="Straight-line speed against race pace, every team">
      <line x1={x(0)} x2={x(0)} y1={pad.t} y2={H - pad.b} stroke={C.line} strokeDasharray="4 4" />
      <line x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} stroke={C.line} strokeDasharray="4 4" />
      {quadrant("QUICK EVERYWHERE", W - pad.r - 4, pad.t + 12, "end")}
      {quadrant("WINS THE CORNERS", pad.l + 4, pad.t + 12, "start")}
      {quadrant("STRAIGHT-LINE CAR", W - pad.r - 4, H - pad.b - 8, "end")}
      {quadrant("DOWN ON BOTH", pad.l + 4, H - pad.b - 8, "start")}
      {sorted.map((t) => {
        const on = t.team_id === teamId;
        return (
          <g key={t.team_id}>
            <circle
              cx={x(t.top_speed_delta_kph)}
              cy={y(t.pace_delta_s)}
              r={on ? 9 : 6}
              fill={on ? "var(--rq-accent)" : C.inactive}
              stroke={C.surface}
              strokeWidth={2}
            />
            <text
              x={x(t.top_speed_delta_kph) + (on ? 13 : 10)}
              y={labelY.get(t.team_id)}
              fill={on ? C.text : C.muted}
              fontFamily={F.body}
              fontWeight={on ? 700 : 600}
              fontSize={on ? 14 : 12}
            >
              {t.name.replace(/ F1 Team$/, "")}
            </text>
          </g>
        );
      })}
      <text x={(pad.l + W - pad.r) / 2} y={H - 8} textAnchor="middle" fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={11}>
        SPEED TRAP VS FIELD (KM/H) →  FASTER ON THE STRAIGHTS
      </text>
      <text transform={`translate(16 ${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle" fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={11}>
        RACE PACE VS FIELD →  FASTER OVER A LAP
      </text>
      {[-xMax * 0.66, 0, xMax * 0.66].map((v) => (
        <text key={v} x={x(v)} y={H - pad.b + 16} textAnchor="middle" fill={C.faint} fontFamily={F.num} fontSize={11}>
          {kph(v)}
        </text>
      ))}
      {[-yMax * 0.66, yMax * 0.66].map((v) => (
        <text key={v} x={pad.l - 8} y={y(v) + 4} textAnchor="end" fill={C.faint} fontFamily={F.num} fontSize={11}>
          {secs(v)}s
        </text>
      ))}
    </svg>
  );
}

function SpeedSection({ season, teamId, teamName }: { season: number; teamId: string; teamName: string }) {
  const speed = useAsync(() => api.speed(season), [season], season >= 2023);
  const team: TeamSpeed | undefined = speed.data?.teams.find((t) => t.team_id === teamId);
  const bars = useMemo(
    () => (team?.by_race ?? []).map((r) => ({ race: `R${r.race_id.split("_")[1]}`, delta: r.trap_delta_kph, circuit: r.circuit_name, kph: r.trap_kph })),
    [team],
  );

  if (season < 2023) {
    return (
      <Card style={{ padding: "18px 20px" }}>
        <EmptyState>SPEED-TRAP DATA STARTS IN 2023 — PICK A SEASON FROM 2023 ON</EmptyState>
      </Card>
    );
  }
  if (speed.loading) return <TableSkeleton rows={6} columns={3} />;
  if (speed.error || !speed.data) return <EmptyState>{(speed.error ?? "No speed data").toUpperCase()}</EmptyState>;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)", gap: 18, alignItems: "start" }}>
      <Card style={{ padding: "18px 20px" }}>
        <SectionLabel style={{ marginBottom: 4 }}>Where every car makes its lap time</SectionLabel>
        <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, lineHeight: 1.55, marginBottom: 8 }}>
          {speed.data.races} races of {season}, green-flag laps only, each race compared within itself so circuits cancel out.
          Top speed is power against drag — nobody publishes the power itself.
        </div>
        <CharacterMap data={speed.data} teamId={teamId} />
      </Card>

      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <Card accent="var(--rq-accent)" style={{ padding: "18px 20px" }}>
          <SectionLabel style={{ marginBottom: 12 }}>{teamName} — the reading</SectionLabel>
          {team ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
                <Stat label="Speed trap vs field" value={kph(team.top_speed_delta_kph)} unit="km/h" size={26} />
                <Stat label="Typical trap speed" value={team.median_trap_kph.toFixed(0)} unit="km/h" size={26} />
                <Stat label="Race pace vs field" value={secs(team.pace_delta_s)} unit="s/lap" size={26} />
              </div>
              <div style={{ fontFamily: F.body, fontSize: 14, color: C.text, lineHeight: 1.6, marginTop: 14 }}>{team.reading}</div>
              {team.top_speed_ci95 && (
                <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginTop: 8 }}>
                  95% interval on the speed-trap gap: {kph(team.top_speed_ci95[0])} to {kph(team.top_speed_ci95[1])} km/h over {team.races} races.
                </div>
              )}
            </>
          ) : (
            <EmptyState>NO SPEED-TRAP LAPS FOR THIS TEAM IN {season}</EmptyState>
          )}
        </Card>
        {team && (
          <Card style={{ padding: "18px 20px" }}>
            <SectionLabel style={{ marginBottom: 8 }}>Straight-line gap, race by race</SectionLabel>
            <ResponsiveContainer width="100%" height={190}>
              <BarChart data={bars} margin={{ top: 6, right: 6, bottom: 0, left: -10 }}>
                <XAxis dataKey="race" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={40} />
                <ReferenceLine y={0} stroke={C.line} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: C.fill }}
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.circuit ?? ""}
                  formatter={(v, _n, item) => [`${kph(Number(v))} km/h (${item?.payload?.kph} km/h)`, "vs field"]}
                />
                <Bar dataKey="delta" radius={[3, 3, 3, 3]} isAnimationActive={false}>
                  {bars.map((b) => (
                    <Cell key={b.race} fill={b.delta >= 0 ? "var(--rq-accent)" : C.inactive} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginTop: 4, lineHeight: 1.5 }}>
              Low-drag circuits (Monza, Las Vegas, Baku) show the biggest straight-line gaps; tight ones barely separate the cars.
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

const SECTIONS = [
  { id: "speed", label: "Speed" },
  { id: "characteristics", label: "Characteristics" },
  { id: "tyres", label: "Tyres" },
];

export function CarView() {
  const team = useTeamSelection();
  const [, setJump] = useState(0);

  return (
    <div style={accentVars(teamAccent(team.teamId || null))}>
      <SectionLabel>Car</SectionLabel>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <h1 style={{ ...DISPLAY, fontSize: 38, margin: "8px 0 18px" }}>
          {team.teamName || "Car"} <span style={{ color: C.faint, fontWeight: 700 }}>{team.season}</span>
        </h1>
        <div style={{ display: "flex", gap: 6 }}>
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              onClick={() => setJump((n) => n + 1)}
              style={{
                fontFamily: F.mono,
                fontWeight: 700,
                fontSize: 12,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: C.dim,
                textDecoration: "none",
                border: `1px solid ${C.edge}`,
                borderRadius: 999,
                padding: "6px 14px",
                background: C.surface,
              }}
            >
              {s.label}
            </a>
          ))}
        </div>
      </div>

      <TeamPicker state={team} />

      {team.ready && (
        <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
          <section id="speed" style={{ scrollMarginTop: 90 }}>
            <SpeedSection season={team.season} teamId={team.teamId} teamName={team.teamName} />
          </section>
          <section id="characteristics" style={{ scrollMarginTop: 90 }}>
            <SectionLabel style={{ marginBottom: 12 }}>Inferred characteristics</SectionLabel>
            <CarProfileSection team={team} />
          </section>
          <section id="tyres" style={{ scrollMarginTop: 90 }}>
            <SectionLabel style={{ marginBottom: 12 }}>Tyres</SectionLabel>
            <TyreSection team={team} />
          </section>
        </div>
      )}
    </div>
  );
}
