/**
 * Tyre View — PRD Section 13.2: team-specific degradation curves per
 * compound, predicted remaining life, and the operating window against
 * track temperature.
 *
 * Every curve is fuel-corrected (car_profiles/degradation_curves.py): raw
 * lap times fall through a stint as fuel burns off, so an uncorrected curve
 * would show tyres getting faster with age. The correction is measured
 * from the data per season and shown in the header, so it can be checked
 * rather than trusted.
 *
 * Compound colours are F1's own (soft red, medium yellow, hard white) —
 * the one place domain convention overrides the palette — and each series
 * is also direct-labelled, so identity never rests on colour alone.
 */

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
import type { CompoundReport, TyreReport } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, ErrorState, SectionLabel, Segmented, Stat, TableSkeleton } from "../../design/primitives";
import { C, DISPLAY, F, NUM, compoundColor } from "../../design/tokens";
import { AXIS_LINE, AXIS_TICK, GRID_STROKE, TOOLTIP_STYLE } from "../components/chartStyle";
import { TeamPicker, useTeamSelection } from "../components/TeamPicker";

const perLap = (v: number | null) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(3)}`);

function CompoundCard({ c }: { c: CompoundReport }) {
  const colour = compoundColor(c.compound);
  const delta = c.wear_s_per_lap != null && c.field_wear_s_per_lap != null ? c.wear_s_per_lap - c.field_wear_s_per_lap : null;
  const distinct =
    c.wear_ci95 != null && c.field_wear_s_per_lap != null &&
    (c.wear_ci95[0] > c.field_wear_s_per_lap || c.wear_ci95[1] < c.field_wear_s_per_lap);
  return (
    <Card accent={colour} style={{ padding: "18px 20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div style={{ ...DISPLAY, fontSize: 24, letterSpacing: "0.04em", color: C.text }}>{c.compound}</div>
        <div style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.14em", color: C.faint }}>{c.stints} STINTS</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        <Stat label="Wear" value={perLap(c.wear_s_per_lap)} unit="s/lap" size={24} />
        <Stat label="Field" value={perLap(c.field_wear_s_per_lap)} unit="s/lap" size={24} color={C.dim} />
      </div>
      <div style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, marginTop: 12, letterSpacing: "0.08em", lineHeight: 1.6 }}>
        {c.wear_ci95 ? `95% ${perLap(c.wear_ci95[0])} TO ${perLap(c.wear_ci95[1])}` : "TOO FEW LAPS FOR AN INTERVAL"}
        <br />
        {delta == null ? (
          " "
        ) : distinct ? (
          <span style={{ color: delta < 0 ? C.green : C.red }}>
            {delta < 0 ? "KINDER" : "HARDER"} ON THIS COMPOUND THAN THE FIELD
          </span>
        ) : (
          "NOT DISTINGUISHABLE FROM THE FIELD"
        )}
      </div>
    </Card>
  );
}

function DegradationCurves({ report }: { report: TyreReport }) {
  const [showField, setShowField] = useState(true);
  // One row per tyre age, a column per compound (team and field), which is
  // the shape a multi-series LineChart wants.
  const rows = useMemo(() => {
    const byAge = new Map<number, Record<string, number>>();
    for (const c of report.compounds) {
      for (const p of c.curve) {
        byAge.set(p.tyre_age, { ...(byAge.get(p.tyre_age) ?? { tyre_age: p.tyre_age }), [c.compound]: p.median_delta_s, [`${c.compound}_laps`]: p.laps });
      }
      for (const p of c.field_curve) {
        byAge.set(p.tyre_age, { ...(byAge.get(p.tyre_age) ?? { tyre_age: p.tyre_age }), [`${c.compound}_field`]: p.median_delta_s });
      }
    }
    return [...byAge.values()].sort((a, b) => a.tyre_age - b.tyre_age);
  }, [report]);

  const present = report.compounds.filter((c) => c.curve.length > 1);

  return (
    <Card style={{ padding: "20px 22px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 6 }}>
        <SectionLabel>Degradation curves</SectionLabel>
        <Segmented
          value={showField ? "field" : "team"}
          onChange={(v) => setShowField(v === "field")}
          options={[
            { value: "field", label: "Team + field" },
            { value: "team", label: "Team only" },
          ]}
        />
      </div>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        Fuel-corrected lap time against tyre age, relative to each stint's laps 2–4 — the median across stints. Solid
        is this team, dashed is the whole grid on the same compound. A point needs at least fifteen laps behind it:
        only unusually long stints reach high tyre ages, so a curve's tail describes a few atypical stints.
      </div>
      {present.length === 0 ? (
        <EmptyState>NOT ENOUGH DRY GREEN-FLAG STINTS THIS SEASON TO DRAW A CURVE</EmptyState>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rows} margin={{ top: 6, right: 16, bottom: 4, left: -8 }}>
            <CartesianGrid stroke={GRID_STROKE} vertical={false} />
            <XAxis
              dataKey="tyre_age"
              type="number"
              domain={[2, "dataMax"]}
              tick={AXIS_TICK}
              axisLine={AXIS_LINE}
              tickLine={false}
              label={{ value: "TYRE AGE (LAPS)", position: "insideBottomRight", offset: -2, fill: C.faint, fontSize: 9, fontFamily: F.mono }}
            />
            <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={48} tickFormatter={(v: number) => v.toFixed(1)} />
            <ReferenceLine y={0} stroke={C.line} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelFormatter={(age) => `Tyre age ${age}`}
              formatter={(value, name) => [`${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)} s`, String(name)]}
            />
            <Legend wrapperStyle={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.1em" }} />
            {present.map((c) => (
              <Line
                key={c.compound}
                dataKey={c.compound}
                name={c.compound}
                stroke={compoundColor(c.compound)}
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
                label={(props: { x?: number; y?: number; index?: number }) => {
                  // Direct label at the series' last point.
                  const last = rows.map((r) => r[c.compound] != null).lastIndexOf(true);
                  if (props.index !== last || props.x == null || props.y == null) return <g />;
                  return (
                    <text x={props.x + 6} y={props.y + 3} fill={C.dim} fontSize={9} fontFamily={F.mono}>
                      {c.compound[0]}
                    </text>
                  );
                }}
              />
            ))}
            {showField &&
              present.map((c) => (
                <Line
                  key={`${c.compound}_field`}
                  dataKey={`${c.compound}_field`}
                  name={`${c.compound} (field)`}
                  stroke={compoundColor(c.compound)}
                  strokeOpacity={0.45}
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                  legendType="plainline"
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

function OperatingWindow({ report }: { report: TyreReport }) {
  const present = report.compounds.filter((c) => c.by_race.length > 0);
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 4 }}>Operating window vs track temperature</SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        This team's fuel-corrected wear rate in each race, against that race's track temperature. Per-race points
        rather than a fitted window: a season is too few races to claim an optimum.
      </div>
      {present.length === 0 ? (
        <EmptyState>NO PER-RACE WEAR READINGS THIS SEASON</EmptyState>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 6, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid stroke={GRID_STROKE} />
            <XAxis
              type="number"
              dataKey="track_temp"
              name="Track temp"
              domain={["dataMin - 2", "dataMax + 2"]}
              tick={AXIS_TICK}
              axisLine={AXIS_LINE}
              tickLine={false}
              tickFormatter={(v: number) => `${v.toFixed(0)}°`}
            />
            <YAxis
              type="number"
              dataKey="wear_s_per_lap"
              name="Wear"
              tick={AXIS_TICK}
              axisLine={false}
              tickLine={false}
              width={48}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <ReferenceLine y={0} stroke={C.line} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              cursor={{ stroke: C.line }}
              content={({ payload }) => {
                const p = payload?.[0]?.payload;
                if (!p) return null;
                return (
                  <div style={{ ...TOOLTIP_STYLE, padding: "8px 10px" }}>
                    <div style={{ color: C.text }}>
                      {p.race_id.replace("_", " R")} · {p.compound}
                    </div>
                    <div style={{ color: C.dim }}>
                      TRACK {p.track_temp.toFixed(1)}°C · WEAR {perLap(p.wear_s_per_lap)} S/LAP · {p.laps} LAPS
                    </div>
                  </div>
                );
              }}
            />
            <Legend wrapperStyle={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.1em" }} />
            {present.map((c) => (
              <Scatter
                key={c.compound}
                name={c.compound}
                data={c.by_race.map((r) => ({ ...r, compound: c.compound }))}
                fill={compoundColor(c.compound)}
                stroke={C.surface}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

function RemainingLife({ report }: { report: TyreReport }) {
  const trace = report.remaining_life;
  // Most laps first, so the default isn't a driver who retired on lap 6.
  const drivers = useMemo(() => {
    const counts = new Map<string, number>();
    for (const l of trace?.laps ?? []) counts.set(l.driver_id, (counts.get(l.driver_id) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([d]) => d);
  }, [trace]);
  const [driver, setDriver] = useState<string | null>(null);
  const active = driver && drivers.includes(driver) ? driver : drivers[0];

  if (!trace) {
    return (
      <Card style={{ padding: "20px 22px" }}>
        <SectionLabel style={{ marginBottom: 10 }}>Predicted remaining life</SectionLabel>
        <EmptyState>NO LAPS FOR THIS TEAM IN THE TYRE MODEL'S DATASET</EmptyState>
      </Card>
    );
  }
  const laps = trace.laps.filter((l) => l.driver_id === active);
  const mae = laps.length ? laps.reduce((s, l) => s + Math.abs(l.predicted_remaining - l.actual_remaining), 0) / laps.length : null;

  return (
    <Card style={{ padding: "20px 22px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 6 }}>
        <SectionLabel>Predicted remaining life — {trace.race_id.replace("_", " R")}</SectionLabel>
        {drivers.length > 1 && (
          <Segmented value={active} onChange={setDriver} options={drivers.map((d) => ({ value: d, label: d.replace(/_/g, " ") }))} />
        )}
      </div>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        The promoted Tyre Degradation model's estimate of laps left on the current set, lap by lap (team colour), against
        how many laps that stint actually ran on (grey). For a final stint "actual" is laps to the flag, and for a
        retirement laps to the retirement — neither is laps the tyre could still have given.
        {mae != null && (
          <span style={{ ...NUM, color: C.text }}> Mean absolute error this race: {mae.toFixed(1)} laps.</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={laps} margin={{ top: 6, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke={GRID_STROKE} vertical={false} />
          <XAxis dataKey="lap_number" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
          <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={40} />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(lap, payload) => {
              const p = payload?.[0]?.payload;
              return p ? `Lap ${lap} · ${p.compound} · age ${p.tyre_age}` : `Lap ${lap}`;
            }}
            formatter={(value, name) => [`${Number(value).toFixed(1)} laps`, String(name)]}
          />
          <Legend wrapperStyle={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.1em" }} />
          <Line dataKey="actual_remaining" name="Actual" stroke={C.muted} strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="predicted_remaining" name="Predicted" stroke={C.accent} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}

export function TyreView() {
  const team = useTeamSelection();
  const report = useAsync(() => api.tyres(team.teamId, team.season), [team.teamId, team.season, team.ready], team.ready);
  const data = report.data;

  // Team-scoped: the whole view takes the team's colour (compound colours stay Pirelli's).
  return (
    <div style={accentVars(teamAccent(team.teamId))}>
      <SectionLabel>Tyre view</SectionLabel>
      <h1 style={{ ...DISPLAY, fontSize: 34, margin: "8px 0 20px", letterSpacing: "0.02em" }}>
        Degradation, compound by compound
      </h1>

      <TeamPicker state={team} />

      {(team.teams.error || report.error) && <ErrorState message={(team.teams.error || report.error)!} />}
      {report.loading && <TableSkeleton rows={6} columns={3} />}

      {data && !report.loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.14em", color: C.faint }}>
            {team.teamName.toUpperCase()} · {data.season} · FUEL + TRACK EFFECT MEASURED AT{" "}
            <span style={{ ...NUM, color: C.dim }}>{data.fuel_track_seconds_per_lap.toFixed(3)}</span> S/LAP AND REMOVED
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 18 }}>
            {data.compounds.map((c) => (
              <CompoundCard key={c.compound} c={c} />
            ))}
          </div>
          <DegradationCurves report={data} />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 18 }}>
            <OperatingWindow report={data} />
            <RemainingLife report={data} />
          </div>
        </div>
      )}
    </div>
  );
}
