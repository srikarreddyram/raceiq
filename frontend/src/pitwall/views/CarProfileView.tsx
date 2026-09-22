/**
 * Car Profile View — PRD Section 13.2: the team's inferred car
 * characteristics with confidence intervals, how they settled over the
 * season, and the tyre-temperature picture.
 *
 * Each characteristic is one row of a forest plot, on its own scale
 * (units differ row to row): the grey band is the spread of the whole grid,
 * the tick is the grid mean, and the team-coloured dot and whisker are this car and
 * its 95% interval. A whisker that spans most of the band means the data
 * can't yet tell this car apart from the field — early in a season, that
 * is most of them, and the view says so instead of ranking noise.
 *
 * Manually seeded priors (PRD 8.1 Layer 2) aren't ingested; the card for
 * them states that rather than showing invented values.
 */

import { useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../api/client";
import { accentVars, teamAccent } from "../../design/theme";
import type { CarCharacteristic, CarProfile } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, ErrorState, SectionLabel, Stat, TableSkeleton } from "../../design/primitives";
import { C, DISPLAY, F, NUM } from "../../design/tokens";
import { ACCENT, AXIS_LINE, AXIS_TICK, GRID_STROKE, TOOLTIP_STYLE } from "../components/chartStyle";
import { TeamPicker, useTeamSelection } from "../components/TeamPicker";

function fmt(value: number | null | undefined, unit: string): string {
  if (value == null) return "—";
  const digits = unit === "r" ? 2 : unit === "s/lap" ? 3 : 2;
  return `${value > 0 && unit !== "places" ? "+" : ""}${value.toFixed(digits)}`;
}

/** Can the data tell this car apart from the grid mean yet? */
function isDistinct(c: CarCharacteristic): boolean {
  return c.ci95 != null && c.field_mean != null && (c.ci95[0] > c.field_mean || c.ci95[1] < c.field_mean);
}

function ForestRow({
  c,
  active,
  onSelect,
}: {
  c: CarCharacteristic;
  active: boolean;
  onSelect: () => void;
}) {
  const [hover, setHover] = useState(false);
  const values = [c.field_min, c.field_max, c.value, ...(c.ci95 ?? [])].filter((v): v is number => v != null);
  const lo = Math.min(...values, 0);
  const hi = Math.max(...values, 0);
  const pad = (hi - lo) * 0.08 || 1;
  const W = 360;
  const x = (v: number) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * W;
  const distinct = isDistinct(c);

  return (
    <button
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "grid",
        gridTemplateColumns: "150px 1fr 128px",
        alignItems: "center",
        gap: 16,
        width: "100%",
        padding: "12px 14px",
        background: active ? C.hover : hover ? C.fill : "transparent",
        border: "none",
        borderLeft: `2px solid ${active ? C.accent : "transparent"}`,
        borderBottom: `1px solid ${C.rule}`,
        cursor: "pointer",
        textAlign: "left",
        position: "relative",
      }}
    >
      <div>
        <div style={{ fontFamily: F.body, fontSize: 13, color: C.text }}>{c.label}</div>
        <div style={{ fontFamily: F.mono, fontSize: 9, color: C.faint, letterSpacing: "0.12em", marginTop: 3 }}>
          {c.unit.toUpperCase()} · N={c.n}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} 28`} style={{ width: "100%", height: 28 }} aria-hidden>
        {c.field_min != null && c.field_max != null && (
          <rect
            x={x(c.field_min)}
            y={10}
            width={Math.max(x(c.field_max) - x(c.field_min), 2)}
            height={8}
            rx={4}
            fill={C.fill}
          />
        )}
        <line x1={x(0)} x2={x(0)} y1={4} y2={24} stroke={C.line} strokeDasharray="2 3" />
        {c.field_mean != null && (
          <line x1={x(c.field_mean)} x2={x(c.field_mean)} y1={6} y2={22} stroke={C.muted} strokeWidth={2} />
        )}
        {c.ci95 && (
          <line x1={x(c.ci95[0])} x2={x(c.ci95[1])} y1={14} y2={14} stroke={ACCENT} strokeWidth={2} strokeOpacity={0.7} />
        )}
        {c.value != null && <circle cx={x(c.value)} cy={14} r={5} fill={ACCENT} stroke={C.surface} strokeWidth={2} />}
      </svg>

      <div style={{ textAlign: "right" }}>
        <div style={{ ...NUM, fontSize: 14, color: C.text }}>{fmt(c.value, c.unit)}</div>
        <div style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.1em", color: distinct ? C.dim : C.faint, marginTop: 3 }}>
          {c.value == null
            ? "NOT ENOUGH DATA"
            : !distinct
              ? "WITHIN FIELD NOISE"
              : c.rank != null
                ? `P${c.rank} OF ${c.teams_ranked}`
                : c.value > (c.field_mean ?? 0)
                  ? "ABOVE FIELD"
                  : "BELOW FIELD"}
        </div>
      </div>

      {hover && (
        <div
          style={{
            position: "absolute",
            left: 164,
            top: "100%",
            zIndex: 5,
            maxWidth: 420,
            background: C.raised,
            border: `1px solid ${C.edge}`,
            borderRadius: 4,
            padding: "10px 12px",
            fontFamily: F.body,
            fontSize: 12,
            color: C.dim,
            lineHeight: 1.55,
            pointerEvents: "none",
          }}
        >
          {c.description}
          <div style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, marginTop: 8 }}>
            THIS CAR {fmt(c.value, c.unit)}
            {c.ci95 && ` [${fmt(c.ci95[0], c.unit)}, ${fmt(c.ci95[1], c.unit)}]`} · GRID MEAN {fmt(c.field_mean, c.unit)}
          </div>
        </div>
      )}
    </button>
  );
}

function Trajectory({ c, minRaces }: { c: CarCharacteristic; minRaces: number }) {
  // Scale to the estimates from `minRaces` on. A round-2 interval can be
  // ten times wider than the season's settled spread and would flatten
  // everything after it; those early bands are clipped, and said to be.
  const settled = c.trajectory.filter((p) => p.n >= minRaces);
  const extent = (settled.length ? settled : c.trajectory).flatMap((p) =>
    [p.value, ...(p.ci95 ?? [])].filter((v): v is number => v != null),
  );
  if (c.field_mean != null) extent.push(c.field_mean);
  const span = extent.length ? Math.max(...extent) - Math.min(...extent) : 1;
  const domain: [number, number] = extent.length
    ? [Math.min(...extent) - span * 0.1, Math.max(...extent) + span * 0.1]
    : [-1, 1];
  const clipped = c.trajectory.some((p) => p.ci95 && (p.ci95[0] < domain[0] || p.ci95[1] > domain[1]));
  const data = c.trajectory.map((p, i) => ({
    race: `R${i + 1}`,
    race_id: p.race_id,
    value: p.value,
    band: p.ci95 ?? undefined,
  }));
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 4 }}>{c.label} — over the season</SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14 }}>
        The estimate after each race, with its 95% interval. The band narrows as races accumulate; where it still
        straddles the grid mean (dashed), the car isn't distinguishable from the field on this measure.
        {clipped && ` Intervals from before race ${minRaces} run off the chart — too few races to be worth scaling to.`}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
          <CartesianGrid stroke={GRID_STROKE} vertical={false} />
          <XAxis dataKey="race" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
          <YAxis
            tick={AXIS_TICK}
            axisLine={false}
            tickLine={false}
            width={52}
            domain={domain}
            allowDataOverflow
            tickFormatter={(v: number) => v.toFixed(c.unit === "s/lap" ? 2 : 1)}
          />
          {c.field_mean != null && <ReferenceLine y={c.field_mean} stroke={C.muted} strokeDasharray="4 4" />}
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(value, name) =>
              name === "band" && Array.isArray(value)
                ? [`${fmt(value[0] as number, c.unit)} to ${fmt(value[1] as number, c.unit)}`, "95% interval"]
                : [fmt(value as number, c.unit), "Estimate"]
            }
            labelFormatter={(label, payload) => `${label} · ${payload?.[0]?.payload?.race_id ?? ""}`}
          />
          <Area dataKey="band" stroke="none" fill={ACCENT} fillOpacity={0.14} isAnimationActive={false} connectNulls />
          <Line dataKey="value" stroke={ACCENT} strokeWidth={2} dot={{ r: 3, fill: ACCENT }} isAnimationActive={false} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  );
}

function TyreWindow({ profile }: { profile: CarProfile }) {
  const points = profile.tyre_window.filter((p) => p.track_temp != null && p.degradation_vs_field != null);
  const heat = profile.characteristics.find((c) => c.key === "tyre_temp_sensitivity");
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 4 }}>Tyre operating window</SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        Each race this season: track temperature against how much faster than the field this car wore its tyres
        (below zero = kinder than the field). A rising cloud means the car falls out of its window in the heat.
        {heat?.value != null &&
          ` Correlation so far: r = ${heat.value.toFixed(2)}${heat.ci95 ? ` (95% ${heat.ci95[0].toFixed(2)} to ${heat.ci95[1].toFixed(2)})` : ""}.`}
      </div>
      {points.length === 0 ? (
        <EmptyState>NO RACES WITH BOTH A TRACK TEMPERATURE AND A DEGRADATION READING</EmptyState>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <ScatterChart margin={{ top: 6, right: 8, bottom: 8, left: -12 }}>
            <CartesianGrid stroke={GRID_STROKE} />
            <XAxis
              type="number"
              dataKey="track_temp"
              name="Track temp"
              unit="°C"
              domain={["dataMin - 2", "dataMax + 2"]}
              tick={AXIS_TICK}
              axisLine={AXIS_LINE}
              tickLine={false}
              tickFormatter={(v: number) => v.toFixed(0)}
            />
            <YAxis
              type="number"
              dataKey="degradation_vs_field"
              name="Degradation vs field"
              tick={AXIS_TICK}
              axisLine={false}
              tickLine={false}
              width={52}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <ReferenceLine y={0} stroke={C.line} strokeDasharray="4 4" />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              cursor={{ stroke: C.line }}
              content={({ payload }) => {
                const p = payload?.[0]?.payload;
                if (!p) return null;
                return (
                  <div style={{ ...TOOLTIP_STYLE, padding: "8px 10px" }}>
                    <div style={{ color: C.text }}>{p.race_name}</div>
                    <div style={{ color: C.dim }}>
                      TRACK {p.track_temp.toFixed(1)}°C
                      {p.circuit_baseline_track_temp != null && ` · USUAL ${p.circuit_baseline_track_temp.toFixed(1)}°C`}
                    </div>
                    <div style={{ color: C.dim }}>VS FIELD {fmt(p.degradation_vs_field, "s/lap")} S/LAP</div>
                  </div>
                );
              }}
            />
            <Scatter data={points} fill={ACCENT} stroke={C.surface} strokeWidth={2} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

export function CarProfileView() {
  const team = useTeamSelection();
  const profile = useAsync(
    () => api.carProfile(team.teamId, team.season),
    [team.teamId, team.season, team.ready],
    team.ready,
  );
  const [selectedKey, setSelectedKey] = useState("degradation_vs_field");

  const data = profile.data;
  const selected = data?.characteristics.find((c) => c.key === selectedKey) ?? data?.characteristics[0];
  const distinctCount = data?.characteristics.filter(isDistinct).length ?? 0;

  // Team-scoped, so the whole view wears the team's colour — the F1 app's team pages do the same.
  return (
    <div style={accentVars(teamAccent(team.teamId))}>
      <SectionLabel>Car profile</SectionLabel>
      <h1 style={{ ...DISPLAY, fontSize: 34, margin: "8px 0 20px", letterSpacing: "0.02em" }}>
        What the data says about the car
      </h1>

      <TeamPicker state={team} />

      {(team.teams.error || profile.error) && <ErrorState message={(team.teams.error || profile.error)!} />}
      {profile.loading && <TableSkeleton rows={8} columns={3} />}

      {data && !profile.loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <Card accent={C.accent} style={{ padding: "22px 26px" }}>
            <SectionLabel>{data.season} season</SectionLabel>
            <h2 style={{ ...DISPLAY, fontSize: 40, margin: "10px 0 18px", letterSpacing: "0.02em" }}>
              {team.teamName}
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 24 }}>
              <Stat label="Races observed" value={data.races_observed} size={26} />
              <Stat label="Through" value={data.last_race_id.replace("_", " R")} size={22} />
              <Stat
                label="Distinct from field"
                value={`${distinctCount} / ${data.characteristics.length}`}
                color={C.accent}
                size={26}
              />
            </div>
            {data.races_observed < data.min_races_for_confidence && (
              <div style={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.14em", color: C.red, marginTop: 16 }}>
                FEWER THAN {data.min_races_for_confidence} RACES — TREAT EVERY VALUE BELOW AS A GUESS
              </div>
            )}
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)", gap: 18, alignItems: "start" }}>
            <Card style={{ overflow: "visible" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "14px 14px 10px",
                  borderBottom: `1px solid ${C.edge}`,
                  gap: 12,
                  flexWrap: "wrap",
                }}
              >
                <SectionLabel>Inferred characteristics</SectionLabel>
                <div style={{ display: "flex", gap: 14, fontFamily: F.mono, fontSize: 9, letterSpacing: "0.12em", color: C.faint }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: ACCENT }} /> THIS CAR ± 95%
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 2, height: 10, background: C.muted }} /> GRID MEAN
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 14, height: 6, borderRadius: 3, background: C.line }} /> GRID RANGE
                  </span>
                </div>
              </div>
              {data.characteristics.map((c) => (
                <ForestRow key={c.key} c={c} active={c.key === selected?.key} onSelect={() => setSelectedKey(c.key)} />
              ))}
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
              {selected && <Trajectory c={selected} minRaces={data.min_races_for_confidence} />}
              <Card style={{ padding: "18px 22px", borderStyle: "dashed" }}>
                <SectionLabel style={{ marginBottom: 8 }}>Seeded engineering priors — absent</SectionLabel>
                <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.65 }}>
                  {data.seeded_priors_note}
                </div>
              </Card>
            </div>
          </div>

          <TyreWindow profile={data} />
        </div>
      )}
    </div>
  );
}
