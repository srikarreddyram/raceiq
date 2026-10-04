/**
 * Practice long runs — race_plan/long_runs.py — on the Race Weekend page.
 *
 * Friday's race simulations, per driver and per team: push-lap pace and
 * spread, how lap time grows over a run, and degradation per compound.
 * One driver is highlighted (the weekend's driver unless another is
 * clicked); everyone else is grey, and compound is the only other colour,
 * so the charts never ask anyone to tell sixteen liveries apart.
 *
 * The section ends with what the plan actually takes from practice — the
 * field's degradation, blended with history — and says plainly what it
 * doesn't, because pace and team-by-team degradation from practice were
 * measured and don't predict the race.
 */

import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { LongRunDriver, LongRuns } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, SectionLabel, TableSkeleton } from "../../design/primitives";
import { C, F, NUM, compoundColor } from "../../design/tokens";

const signed = (v: number, digits = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
const deg = (v: number | null) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`);

function SessionTabs({ available, value, onChange }: { available: string[]; value: string; onChange: (s: string) => void }) {
  const options = [{ id: "", label: "All practice" }, ...available.map((s) => ({ id: s, label: s }))];
  return (
    <div style={{ display: "flex", gap: 6 }}>
      {options.map((o) => {
        const on = o.id === value;
        return (
          <button
            key={o.id || "all"}
            onClick={() => onChange(o.id)}
            style={{
              background: on ? C.carbon : C.raised,
              color: on ? "#fff" : C.dim,
              border: `1px solid ${on ? C.carbon : C.edge}`,
              borderRadius: 6,
              padding: "6px 12px",
              fontFamily: F.mono,
              fontWeight: 700,
              fontSize: 11.5,
              letterSpacing: "0.06em",
              cursor: "pointer",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Every driver's push laps on one row, quickest at the top. */
function PaceStrip({ data, highlight, onPick }: { data: LongRuns; highlight: string; onPick: (code: string) => void }) {
  const drivers = data.drivers;
  const rowH = 24;
  const pad = { l: 46, r: 178, t: 26, b: 30 };
  const W = 640;
  const H = pad.t + rowH * drivers.length + pad.b;
  const times = data.laps.map((l) => l.lap_time);
  const lo = Math.min(...times) - 0.2;
  const hi = Math.max(...times) + 0.2;
  const x = (t: number) => pad.l + ((t - lo) / (hi - lo)) * (W - pad.l - pad.r);
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let t = Math.ceil(lo); t <= hi; t += 1) out.push(t);
    return out;
  }, [lo, hi]);
  const lapsBy = useMemo(() => {
    const m = new Map<string, typeof data.laps>();
    for (const l of data.laps) m.set(l.code, [...(m.get(l.code) ?? []), l]);
    return m;
  }, [data]);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} role="img" aria-label="Push-lap times per driver">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={pad.t - 6} y2={H - pad.b} stroke={C.rule} />
          <text x={x(t)} y={H - pad.b + 16} textAnchor="middle" fill={C.faint} fontFamily={F.num} fontSize={11}>
            {t}
          </text>
        </g>
      ))}
      {[
        ["GAP", W - pad.r + 44],
        ["LAPS", W - pad.r + 104],
        ["SD", W - pad.r + 156],
      ].map(([label, lx]) => (
        <text key={label} x={lx} y={pad.t - 10} textAnchor="end" fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={10.5} letterSpacing="0.08em">
          {label}
        </text>
      ))}
      {drivers.map((d, i) => {
        const y = pad.t + i * rowH + rowH / 2;
        const on = d.code === highlight;
        return (
          <g key={d.code} onClick={() => onPick(d.code)} style={{ cursor: "pointer" }}>
            <rect x={0} y={y - rowH / 2} width={W} height={rowH} fill={on ? C.hover : i % 2 ? "transparent" : C.fill} opacity={on ? 1 : 0.5} />
            <text x={pad.l - 8} y={y + 4} textAnchor="end" fill={on ? C.text : C.dim} fontFamily={F.mono} fontWeight={on ? 800 : 600} fontSize={12}>
              {d.code}
            </text>
            {(lapsBy.get(d.code) ?? []).map((l, k) => (
              <circle key={k} cx={x(l.lap_time)} cy={y} r={on ? 4.5 : 3.6} fill={compoundColor(l.compound)} stroke={C.carbon} strokeOpacity={0.55} strokeWidth={0.8} opacity={on ? 1 : 0.8} />
            ))}
            <line x1={x(d.mean_lap)} x2={x(d.mean_lap)} y1={y - 8} y2={y + 8} stroke={on ? "var(--rq-accent)" : C.carbon} strokeWidth={on ? 3 : 2} />
            <text x={W - pad.r + 44} y={y + 4} textAnchor="end" fill={C.text} fontFamily={F.num} fontSize={12} fontWeight={on ? 700 : 500}>
              {i === 0 ? "—" : signed(d.gap)}
            </text>
            <text x={W - pad.r + 104} y={y + 4} textAnchor="end" fill={C.dim} fontFamily={F.num} fontSize={12}>
              {d.push_laps}
            </text>
            <text x={W - pad.r + 156} y={y + 4} textAnchor="end" fill={C.dim} fontFamily={F.num} fontSize={12}>
              {d.sd == null ? "—" : d.sd.toFixed(2)}
            </text>
          </g>
        );
      })}
      <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={10.5}>
        PUSH-LAP TIME (S) — BAR IS THE AVERAGE
      </text>
    </svg>
  );
}

/** Lap time against tyre age, one line per run. */
function RunLines({ data, highlight }: { data: LongRuns; highlight: string }) {
  const W = 640;
  const H = 330;
  const pad = { l: 50, r: 16, t: 14, b: 40 };
  const laps = data.laps.filter((l) => l.tyre_age != null);
  const teamOf = data.drivers.find((d) => d.code === highlight)?.team_id;
  const ages = laps.map((l) => l.tyre_age as number);
  const times = laps.map((l) => l.lap_time);
  if (!laps.length) return <EmptyState>NO PUSH LAPS WITH A TYRE AGE</EmptyState>;
  const aLo = Math.min(...ages) - 0.5;
  const aHi = Math.max(...ages) + 0.5;
  const tLo = Math.min(...times) - 0.15;
  const tHi = Math.max(...times) + 0.15;
  const x = (a: number) => pad.l + ((a - aLo) / (aHi - aLo)) * (W - pad.l - pad.r);
  const y = (t: number) => pad.t + ((tHi - t) / (tHi - tLo)) * (H - pad.t - pad.b);

  const runs = new Map<string, typeof laps>();
  for (const l of laps) {
    const key = `${l.code}|${l.session}|${l.run}`;
    runs.set(key, [...(runs.get(key) ?? []), l]);
  }
  const ordered = [...runs.entries()].sort(([a], [b]) => {
    const rank = (k: string) => (k.startsWith(`${highlight}|`) ? 2 : data.drivers.find((d) => d.code === k.split("|")[0])?.team_id === teamOf ? 1 : 0);
    return rank(a) - rank(b);
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} role="img" aria-label="Lap time against tyre age for every long run">
      {Array.from({ length: Math.floor(tHi) - Math.ceil(tLo) + 1 }, (_, i) => Math.ceil(tLo) + i).map((t) => {
        return (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke={C.rule} />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" fill={C.faint} fontFamily={F.num} fontSize={11}>
              {t}
            </text>
          </g>
        );
      })}
      {Array.from({ length: Math.floor(aHi) - Math.ceil(aLo) + 1 }, (_, i) => Math.ceil(aLo) + i)
        .filter((a) => a % 2 === 0)
        .map((a) => (
          <text key={a} x={x(a)} y={H - pad.b + 16} textAnchor="middle" fill={C.faint} fontFamily={F.num} fontSize={11}>
            {a}
          </text>
        ))}
      {ordered.map(([key, run]) => {
        const code = key.split("|")[0];
        const mine = code === highlight;
        const mate = !mine && data.drivers.find((d) => d.code === code)?.team_id === teamOf;
        const pts = [...run].sort((a, b) => (a.tyre_age as number) - (b.tyre_age as number));
        const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.tyre_age as number)},${y(p.lap_time)}`).join(" ");
        return (
          <g key={key}>
            <path
              d={path}
              fill="none"
              stroke={mine || mate ? "var(--rq-accent)" : C.inactive}
              strokeWidth={mine ? 2.6 : mate ? 1.8 : 1}
              strokeDasharray={mate ? "5 4" : undefined}
              opacity={mine || mate ? 1 : 0.55}
            />
            {(mine || mate) &&
              pts.map((p, i) => (
                <circle key={i} cx={x(p.tyre_age as number)} cy={y(p.lap_time)} r={mine ? 4 : 3} fill={compoundColor(p.compound)} stroke={C.carbon} strokeOpacity={0.6} strokeWidth={0.8} />
              ))}
          </g>
        );
      })}
      <text x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle" fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={10.5}>
        TYRE AGE (LAPS)
      </text>
      <text transform={`translate(13 ${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle" fill={C.faint} fontFamily={F.mono} fontWeight={700} fontSize={10.5}>
        LAP TIME (S) ↑ SLOWER
      </text>
    </svg>
  );
}

function CompoundKey() {
  return (
    <div style={{ display: "flex", gap: 14, alignItems: "center", fontFamily: F.mono, fontSize: 11, fontWeight: 700, color: C.dim }}>
      {["SOFT", "MEDIUM", "HARD"].map((c) => (
        <span key={c} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 10, height: 10, borderRadius: 5, background: compoundColor(c), border: `1px solid ${C.line}` }} />
          {c}
        </span>
      ))}
    </div>
  );
}

const th = { fontFamily: F.mono, fontWeight: 700, fontSize: 10.5, letterSpacing: "0.08em", color: C.faint, textAlign: "right" as const, padding: "6px 8px" };
const td = { ...NUM, fontSize: 12.5, color: C.text, textAlign: "right" as const, padding: "6px 8px", borderTop: `1px solid ${C.rule}` };

function TeamTable({ data, highlightTeam }: { data: LongRuns; highlightTeam: string | undefined }) {
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr>
          <th style={{ ...th, textAlign: "left" }}>TEAM</th>
          <th style={th}>GAP</th>
          <th style={th}>LAPS</th>
          <th style={th}>SOFT</th>
          <th style={th}>MEDIUM</th>
          <th style={th}>HARD</th>
        </tr>
      </thead>
      <tbody>
        {data.teams.map((t, i) => {
          const on = t.team_id === highlightTeam;
          return (
            <tr key={t.team_id} style={{ background: on ? C.hover : undefined }}>
              <td style={{ ...td, textAlign: "left", fontFamily: F.body, fontWeight: on ? 800 : 600 }}>
                {t.team_name}{" "}
                <span style={{ color: C.faint, fontWeight: 500, fontSize: 11.5 }}>{t.drivers.join(" · ")}</span>
              </td>
              <td style={td}>{i === 0 ? "—" : signed(t.gap)}</td>
              <td style={{ ...td, color: C.dim }}>{t.push_laps}</td>
              <td style={td}>{deg(t.deg_soft)}</td>
              <td style={td}>{deg(t.deg_medium)}</td>
              <td style={td}>{deg(t.deg_hard)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function WhatThePlanUses({ data }: { data: LongRuns }) {
  return (
    <Card accent="var(--rq-accent)" style={{ padding: "18px 20px" }}>
      <SectionLabel style={{ marginBottom: 10 }}>What the plan takes from practice</SectionLabel>
      <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: 10 }}>
        <thead>
          <tr>
            <th style={{ ...th, textAlign: "left" }}>TYRE WEAR (S/LAP PER LAP)</th>
            <th style={th}>PRACTICE</th>
            <th style={th}>HISTORY</th>
            <th style={th}>PLAN USES</th>
          </tr>
        </thead>
        <tbody>
          {data.wear.map((w) => (
            <tr key={w.compound}>
              <td style={{ ...td, textAlign: "left", fontFamily: F.mono, fontWeight: 700 }}>
                <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 5, background: compoundColor(w.compound), border: `1px solid ${C.line}`, marginRight: 7 }} />
                {w.compound}
              </td>
              <td style={td}>{w.practice == null ? "—" : w.practice.toFixed(3)}</td>
              <td style={{ ...td, color: C.dim }}>{w.history.toFixed(3)}</td>
              <td style={{ ...td, fontWeight: 800 }}>{w.used.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, lineHeight: 1.6 }}>
        {data.practice_in_plan ? "" : "No practice long runs yet, so the plan uses history alone. "}
        {data.evidence}
      </div>
    </Card>
  );
}

export function LongRunsSection({ raceId, driverId }: { raceId: string; driverId: string }) {
  // null until the race's own default arrives: the session with the most
  // long-run laps. Pooling sessions mixes fuel loads and track conditions.
  const [session, setSession] = useState<string | null>(null);
  const first = useAsync(() => api.longRuns(raceId), [raceId], Boolean(raceId));
  useEffect(() => setSession(null), [raceId]);
  const chosen = session ?? first.data?.default_session ?? "";
  const runs = useAsync(
    () => (chosen === "" && first.data ? Promise.resolve(first.data) : api.longRuns(raceId, chosen || undefined)),
    [raceId, chosen, first.data],
    Boolean(raceId) && !first.loading,
  );
  const weekendDriver: LongRunDriver | undefined = runs.data?.drivers.find((d) => d.driver_id === driverId);
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => setPicked(null), [raceId, driverId]);
  const highlight = picked ?? weekendDriver?.code ?? runs.data?.drivers[0]?.code ?? "";
  const highlighted = runs.data?.drivers.find((d) => d.code === highlight);

  if (first.loading || runs.loading) return <TableSkeleton rows={8} columns={4} />;
  if (first.error || runs.error || !runs.data) {
    return (
      <Card style={{ padding: "18px 20px" }}>
        <SectionLabel style={{ marginBottom: 8 }}>Practice long runs</SectionLabel>
        <EmptyState>
          {String(first.error ?? runs.error ?? "No practice laps").toUpperCase().includes("NO PRACTICE")
            ? "NO PRACTICE LAPS YET FOR THIS WEEKEND"
            : String(first.error ?? runs.error).toUpperCase()}
        </EmptyState>
      </Card>
    );
  }
  const data = runs.data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <Card style={{ padding: "18px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 14, flexWrap: "wrap", marginBottom: 6 }}>
          <SectionLabel>Practice long runs</SectionLabel>
          <SessionTabs available={data.sessions_available} value={chosen} onChange={setSession} />
        </div>
        <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, lineHeight: 1.55, marginBottom: 12 }}>
          Push laps from runs of five or more laps on one set of tyres, under green flags, within 1.5% of the run's
          average — traffic and cool-down laps dropped. Fuel loads aren't published, so pace gaps here are indicative.
          Click a driver to highlight them.
        </div>
        {data.drivers.length === 0 ? (
          <EmptyState>NO LONG RUNS IN THIS SESSION</EmptyState>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(440px, 1fr))", gap: 22, alignItems: "start" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <div style={{ fontFamily: F.mono, fontWeight: 700, fontSize: 11, color: C.dim, letterSpacing: "0.06em" }}>LONG-RUN PACE</div>
                <CompoundKey />
              </div>
              <PaceStrip data={data} highlight={highlight} onPick={setPicked} />
            </div>
            <div>
              <div style={{ fontFamily: F.mono, fontWeight: 700, fontSize: 11, color: C.dim, letterSpacing: "0.06em", marginBottom: 6 }}>
                OVER THE RUN — {highlight}
                {highlighted ? ` (SOLID) AND TEAMMATE (DASHED)` : ""}
              </div>
              <RunLines data={data} highlight={highlight} />
              {highlighted && (
                <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.text, lineHeight: 1.55, marginTop: 6 }}>
                  {highlighted.code}: {highlighted.push_laps} push laps, average {highlighted.mean_lap.toFixed(3)} s
                  {highlighted.gap > 0 ? ` (${signed(highlighted.gap)} s to the quickest)` : " (quickest)"}. Degradation, fuel effect removed:{" "}
                  {[
                    ["soft", highlighted.deg_soft],
                    ["medium", highlighted.deg_medium],
                    ["hard", highlighted.deg_hard],
                  ]
                    .filter(([, v]) => v != null)
                    .map(([c, v]) => `${c} ${deg(v as number)} s/lap per lap`)
                    .join(", ") || "too few laps on one compound to measure"}
                  .
                </div>
              )}
            </div>
          </div>
        )}
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 18, alignItems: "start" }}>
        <Card style={{ padding: "18px 20px" }}>
          <SectionLabel style={{ marginBottom: 4 }}>Teams — pace and tyre degradation</SectionLabel>
          <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, lineHeight: 1.55, marginBottom: 10 }}>
            Both drivers' push laps together. Degradation is seconds per lap lost per lap of tyre age, fuel effect removed.
            Read it as Friday's story, not a forecast: which team degrades more than the others in practice hasn't
            predicted which does in the race.
          </div>
          <TeamTable data={data} highlightTeam={highlighted?.team_id} />
        </Card>
        <WhatThePlanUses data={data} />
      </div>
    </div>
  );
}
