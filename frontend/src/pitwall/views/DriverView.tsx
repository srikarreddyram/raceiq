/**
 * Driver View — PRD Section 13.2: per-driver performance profile, circuit
 * history, wet/dry splits.
 *
 * The headline pace number is the gap to the teammate on shared clean
 * laps — the only comparison that holds the car constant, and so the only
 * one comparable across the three regulation eras this data spans.
 * Field-relative pace is shown only where labelled as car-and-driver.
 *
 * Finishing averages count classified finishes only; Ergast's position is
 * finishing ORDER and puts a lap-3 retirement at P22, which would turn a
 * reliability problem into an apparent driving one.
 */

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../api/client";
import { accentVars, teamAccent } from "../../design/theme";
import type { DriverProfile, DriverRaceResult } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, ErrorState, SectionLabel, Segmented, Stat, TableSkeleton } from "../../design/primitives";
import { C, DISPLAY, F, NUM } from "../../design/tokens";
import { ACCENT, AXIS_LINE, AXIS_TICK, GRID_STROKE, TOOLTIP_STYLE } from "../components/chartStyle";

const SEASONS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018];
const SLOWER = C.inactive; // neutral pole of the teammate-gap diverging pair; the team colour is "faster"

const gap = (v: number | null | undefined) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(3)}s`);
const pos = (v: number | null | undefined) => (v == null ? "—" : `P${v}`);
const pretty = (id: string | null | undefined) => (id ?? "—").replace(/_/g, " ");

const controlStyle = {
  background: C.raised,
  color: C.text,
  border: `1px solid ${C.edge}`,
  borderRadius: 4,
  padding: "8px 10px",
  fontFamily: F.mono,
  fontSize: 12,
  outline: "none",
};

/** Grid → finish for every race: a line from where they started to where they ended. */
function RaceStrip({ races }: { races: DriverRaceResult[] }) {
  const [hover, setHover] = useState<DriverRaceResult | null>(null);
  // Drawn near 1:1 with its card so marks and labels keep their size.
  const W = 600;
  const H = 280;
  const padX = 30;
  const top = 16;
  const bottom = 30;
  const maxPos = Math.max(20, ...races.map((r) => Math.max(r.grid ?? 0, r.position ?? 0)));
  const x = (i: number) => padX + (i + 0.5) * ((W - 2 * padX) / races.length);
  const y = (p: number) => top + ((p - 1) / (maxPos - 1)) * (H - top - bottom);

  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} role="img" aria-label="Grid and finishing position by race">
        {[1, 5, 10, 15, 20].filter((p) => p <= maxPos).map((p) => (
          <g key={p}>
            <line x1={padX} x2={W - padX} y1={y(p)} y2={y(p)} stroke={GRID_STROKE} />
            <text x={padX - 8} y={y(p) + 3} textAnchor="end" fill={C.faint} fontSize={10} fontFamily={F.mono}>
              P{p}
            </text>
          </g>
        ))}
        {races.map((r, i) => {
          const cx = x(i);
          const finishY = r.position != null ? y(r.position) : null;
          const gridY = r.grid != null ? y(r.grid) : null;
          const gained = r.grid != null && r.position != null && r.classified && r.position < r.grid;
          return (
            <g key={r.race_id} onMouseEnter={() => setHover(r)} onMouseLeave={() => setHover(null)}>
              <rect x={cx - 16} y={0} width={32} height={H} fill="transparent" />
              {gridY != null && finishY != null && r.classified && (
                <line x1={cx} x2={cx} y1={gridY} y2={finishY} stroke={gained ? ACCENT : SLOWER} strokeWidth={2} />
              )}
              {gridY != null && <circle cx={cx} cy={gridY} r={4.5} fill={C.bg} stroke={C.muted} strokeWidth={1.5} />}
              {r.classified && finishY != null ? (
                <circle cx={cx} cy={finishY} r={5.5} fill={r.position! <= 3 ? ACCENT : C.text} stroke={C.surface} strokeWidth={2} />
              ) : (
                <text x={cx} y={H - bottom + 2} textAnchor="middle" fill={C.red} fontSize={12} fontFamily={F.mono}>
                  ×
                </text>
              )}
              <text x={cx} y={H - 6} textAnchor="middle" fill={C.faint} fontSize={9} fontFamily={F.mono}>
                R{r.race_id.split("_")[1]}
              </text>
            </g>
          );
        })}
      </svg>
      {hover && (
        <div style={{ ...TOOLTIP_STYLE, position: "absolute", top: 4, right: 4, padding: "8px 12px", lineHeight: 1.7, pointerEvents: "none" }}>
          <div>{hover.race_name}</div>
          <div style={{ color: C.dim }}>
            GRID {pos(hover.grid)} → {hover.classified ? pos(hover.position) : `${hover.status?.toUpperCase()} (NOT CLASSIFIED)`}
          </div>
          <div style={{ color: C.dim }}>
            {pretty(hover.teammate_id).toUpperCase()} {hover.teammate_position != null ? pos(hover.teammate_position) : ""} · GAP{" "}
            {gap(hover.teammate_gap_s)}
            {hover.teammate_gap_laps != null && ` OVER ${hover.teammate_gap_laps} LAPS`}
          </div>
        </div>
      )}
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 8, fontFamily: F.mono, fontSize: 9, letterSpacing: "0.14em", color: C.faint }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", border: `1.5px solid ${C.muted}` }} /> GRID
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.text }} /> FINISH
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: ACCENT }} /> PODIUM
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 14, height: 2, background: ACCENT }} /> PLACES GAINED
        </span>
        <span style={{ color: C.red }}>× NOT CLASSIFIED</span>
      </div>
    </div>
  );
}

function TeammateGaps({ races }: { races: DriverRaceResult[] }) {
  const data = races
    .filter((r) => r.teammate_gap_s != null)
    .map((r) => ({ race: `R${r.race_id.split("_")[1]}`, gap: r.teammate_gap_s!, laps: r.teammate_gap_laps, teammate: r.teammate_id, name: r.race_name }));
  if (!data.length) return <EmptyState>NO SHARED CLEAN LAPS WITH A TEAMMATE THIS SEASON</EmptyState>;
  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -6 }}>
        <CartesianGrid stroke={GRID_STROKE} vertical={false} />
        <XAxis dataKey="race" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
        <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={46} tickFormatter={(v: number) => v.toFixed(1)} />
        <ReferenceLine y={0} stroke={C.line} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          cursor={{ fill: C.fill }}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.name ?? ""}
          formatter={(value, _name, item) => [
            `${gap(Number(value))} vs ${pretty(item?.payload?.teammate)} (${item?.payload?.laps} laps)`,
            "Median gap",
          ]}
        />
        <Bar dataKey="gap" radius={[2, 2, 2, 2]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.race} fill={d.gap < 0 ? ACCENT : SLOWER} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: F.mono, fontSize: 11.5 }}>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th
                key={h}
                style={{
                  textAlign: i === 0 ? "left" : "right",
                  padding: "8px 10px",
                  fontSize: 9,
                  letterSpacing: "0.16em",
                  color: C.faint,
                  fontWeight: 400,
                  borderBottom: `1px solid ${C.edge}`,
                  whiteSpace: "nowrap",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, i) => (
                <td
                  key={i}
                  style={{
                    ...NUM,
                    textAlign: i === 0 ? "left" : "right",
                    padding: "8px 10px",
                    color: i === 0 ? C.text : C.dim,
                    borderBottom: `1px solid ${C.rule}`,
                    whiteSpace: "nowrap",
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WetDry({ profile }: { profile: DriverProfile }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 14 }}>
      {profile.wet_dry.map((s) => {
        const delta = s.wet_teammate_gap_s != null && s.dry_teammate_gap_s != null ? s.wet_teammate_gap_s - s.dry_teammate_gap_s : null;
        return (
          <div key={s.era} style={{ background: C.raised, border: `1px solid ${C.edge}`, borderRadius: 6, padding: "14px 16px" }}>
            <div style={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.2em", color: C.accent }}>{s.era} REGS</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
              <Stat label="Dry vs teammate" value={gap(s.dry_teammate_gap_s)} size={20} />
              <Stat label="Wet vs teammate" value={gap(s.wet_teammate_gap_s)} size={20} color={s.wet_teammate_gap_s == null ? C.faint : undefined} />
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, marginTop: 10, lineHeight: 1.7, letterSpacing: "0.06em" }}>
              {s.dry_laps} DRY LAPS · {s.wet_laps} WET LAPS OVER {s.wet_races} RACE{s.wet_races === 1 ? "" : "S"}
              <br />
              {s.wet_teammate_gap_s == null ? (
                `FEWER THAN ${profile.min_wet_laps} WET LAPS — NO SPLIT`
              ) : (
                <span style={{ color: C.dim }}>
                  {delta! < 0 ? "RELATIVELY STRONGER" : "RELATIVELY WEAKER"} IN THE WET BY {Math.abs(delta!).toFixed(3)}S
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function DriverView() {
  const [season, setSeason] = useState(2026);
  const [driverId, setDriverId] = useState("leclerc");
  const drivers = useAsync(() => api.drivers(season), [season]);
  const ready = Boolean(drivers.data?.some((d) => d.driver_id === driverId));

  useEffect(() => {
    if (drivers.data?.length && !drivers.data.some((d) => d.driver_id === driverId)) {
      setDriverId(drivers.data[0].driver_id);
    }
  }, [drivers.data, driverId]);

  const profile = useAsync(() => api.driverProfile(driverId, season), [driverId, season, ready], ready);
  const [showAllCircuits, setShowAllCircuits] = useState(false);
  const data = profile.data;

  const circuits = useMemo(() => (showAllCircuits ? data?.circuits : data?.circuits.slice(0, 10)) ?? [], [data, showAllCircuits]);

  // Themed in the colour of the team the driver raced for most recently in the selected season.
  return (
    <div style={accentVars(teamAccent(profile.data?.races[profile.data.races.length - 1]?.team_id ?? null))}>
      <SectionLabel>Driver view</SectionLabel>
      <h1 style={{ ...DISPLAY, fontSize: 34, margin: "8px 0 20px", letterSpacing: "0.02em" }}>
        Driver against the only fair benchmark
      </h1>

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          background: C.surface,
          border: `1px solid ${C.edge}`,
          borderRadius: 6,
          padding: "12px 16px",
          marginBottom: 20,
          flexWrap: "wrap",
        }}
      >
        <select value={season} onChange={(e) => setSeason(Number(e.target.value))} style={controlStyle} aria-label="Season">
          {SEASONS.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select value={driverId} onChange={(e) => setDriverId(e.target.value)} style={{ ...controlStyle, minWidth: 220 }} aria-label="Driver">
          {[...(drivers.data ?? [])]
            .sort((a, b) => (a.family_name ?? "").localeCompare(b.family_name ?? ""))
            .map((d) => (
              <option key={d.driver_id} value={d.driver_id}>
                {d.given_name} {d.family_name} — {pretty(d.constructor_id)}
              </option>
            ))}
        </select>
      </div>

      {(drivers.error || profile.error) && <ErrorState message={(drivers.error || profile.error)!} />}
      {profile.loading && <TableSkeleton rows={8} columns={4} />}

      {data && !profile.loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <Card accent={C.accent} style={{ padding: "22px 26px" }}>
            <SectionLabel>
              {data.nationality} · {data.season} · {pretty(data.races[0]?.team_id)}
            </SectionLabel>
            <h2 style={{ ...DISPLAY, fontSize: 40, margin: "10px 0 18px", letterSpacing: "0.02em" }}>
              {data.given_name} {data.family_name}
              {data.code && <span style={{ color: C.faint, fontSize: 24, marginLeft: 14 }}>{data.code}</span>}
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 22 }}>
              <Stat label="Race points" value={data.summary.points ?? "—"} size={26} />
              <Stat label="Wins" value={data.summary.wins} size={26} />
              <Stat label="Podiums" value={data.summary.podiums} size={26} />
              <Stat label="Avg grid" value={data.summary.avg_grid ?? "—"} size={26} />
              <Stat label="Avg finish" value={data.summary.avg_finish ?? "—"} size={26} />
              <Stat label="Not classified" value={data.summary.not_classified} size={26} color={data.summary.not_classified ? C.red : undefined} />
              <Stat
                label="Ahead of teammate"
                value={`${data.summary.ahead_of_teammate}/${data.summary.head_to_head_races}`}
                size={26}
                color={C.accent}
              />
              <Stat
                label="Pace vs teammate"
                value={gap(data.summary.median_teammate_gap_s)}
                size={22}
                color={(data.summary.median_teammate_gap_s ?? 0) < 0 ? C.accent : undefined}
              />
            </div>
            <div style={{ fontFamily: F.body, fontSize: 11.5, color: C.faint, marginTop: 14 }}>
              Race points only, sprints excluded. Average finish counts classified finishes. Pace vs teammate is the
              median of per-race median gaps on shared clean laps; negative is faster.
            </div>
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(460px, 1fr))", gap: 18 }}>
            <Card style={{ padding: "20px 22px" }}>
              <SectionLabel style={{ marginBottom: 12 }}>Race by race — grid to finish</SectionLabel>
              <RaceStrip races={data.races} />
            </Card>
            <Card style={{ padding: "20px 22px" }}>
              <SectionLabel style={{ marginBottom: 4 }}>Gap to teammate, per race</SectionLabel>
              <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 12 }}>
                Median lap-time gap on laps both drivers raced clean. Team colour is faster; below zero is faster.
              </div>
              <TeammateGaps races={data.races} />
            </Card>
          </div>

          <Card style={{ padding: "20px 22px" }}>
            <SectionLabel style={{ marginBottom: 4 }}>Wet / dry, by regulation era</SectionLabel>
            <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
              Wet means laps on intermediates or wets — the track, not the weather station. Compared against the
              teammate in the same car, so the car cancels; split by era because a 2019 teammate in a 2019 car isn't
              the same benchmark as a 2026 one.
            </div>
            <WetDry profile={data} />
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(460px, 1fr))", gap: 18, alignItems: "start" }}>
            <Card style={{ padding: "20px 22px" }}>
              <SectionLabel style={{ marginBottom: 12 }}>Career</SectionLabel>
              <Table
                head={["Season", "Era", "Team", "Races", "Pts", "Wins", "Pod", "Avg fin", "Not cl.", "vs TM"]}
                rows={data.career.map((c) => [
                  c.season,
                  c.era,
                  c.teams.map(pretty).join(", "),
                  c.races,
                  c.points ?? "—",
                  c.wins,
                  c.podiums,
                  c.avg_finish ?? "—",
                  c.not_classified,
                  gap(c.median_teammate_gap_s),
                ])}
              />
            </Card>
            <Card style={{ padding: "20px 22px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, gap: 12 }}>
                <SectionLabel>Circuit history</SectionLabel>
                {data.circuits.length > 10 && (
                  <Segmented
                    value={showAllCircuits ? "all" : "top"}
                    onChange={(v) => setShowAllCircuits(v === "all")}
                    options={[
                      { value: "top", label: "Most raced" },
                      { value: "all", label: `All ${data.circuits.length}` },
                    ]}
                  />
                )}
              </div>
              <Table
                head={["Circuit", "Races", "Best", "Avg fin", "Wins", "Last", "vs TM"]}
                rows={circuits.map((c) => [
                  c.circuit_name ?? c.circuit_id,
                  c.races,
                  pos(c.best_finish),
                  c.avg_finish ?? "—",
                  c.wins,
                  c.last_position != null ? pos(c.last_position) : (c.last_status ?? "—"),
                  gap(c.median_teammate_gap_s),
                ])}
              />
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
