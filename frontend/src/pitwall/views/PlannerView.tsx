/**
 * Race Weekend Planner — the project's centrepiece (race_plan/).
 *
 * For one driver at one race, from the conditions and the grid slot:
 * which tyre sets to spend in practice and which to keep for qualifying
 * and the race; which compound to start on and what to switch to; the
 * window for each stop; and, for each stop, how long it's worth holding
 * on for a safety car or VSC before pulling the plug and pitting anyway.
 *
 * Honest by construction: a starting-compound "choice" that's within
 * simulation noise is shown as a tie, a new venue with no history says so,
 * and the grid slot can be changed to ask "what if we start P8?" before
 * qualifying is known.
 */

import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { PlannedStop, RacePlan } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, ErrorState, SectionLabel, Stat } from "../../design/primitives";
import { accentVars, teamAccent } from "../../design/theme";
import { C, DISPLAY, F, NUM, compoundColor } from "../../design/tokens";
import { RaceSelector, controlStyle, labelStyle, useRaceSelection } from "../components/RaceSelector";

const pct = (v: number) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

function daysUntil(date: string): string {
  const days = Math.round((new Date(date).setHours(0, 0, 0, 0) - new Date().setHours(0, 0, 0, 0)) / 86_400_000);
  return days <= 0 ? "RACE DAY" : days === 1 ? "TOMORROW" : `IN ${days} DAYS`;
}

/**
 * The headline: where this car finishes on a typical strategy here — the
 * number that's checked against real results (race_plan/grid_sensitivity.py).
 */
function Outcome({ plan: p }: { plan: RacePlan }) {
  const typical = p.typical_expected_finish != null;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, auto)", gap: 26, alignItems: "end" }}>
      <Stat label="Expected finish" value={`P${(typical ? p.typical_expected_finish! : p.expected_finish).toFixed(1)}`} size={30} />
      <Stat label="Win" value={pct(typical ? p.typical_win_probability! : p.win_probability)} size={30} />
      <Stat label="Points" value={pct(typical ? p.typical_points_probability! : p.points_probability)} size={30} />
      <Stat label="Exp. points" value={(typical ? p.typical_expected_points! : p.expected_points).toFixed(1)} size={30} />
    </div>
  );
}

/**
 * What the simulation credits this plan with over a typical strategy, next
 * to what real races say that credit is worth. The plan is still the call
 * to make; the extra places are not a forecast.
 */
function PlanEdge({ plan: p }: { plan: RacePlan }) {
  if (p.typical_expected_finish == null) return null;
  const edge = p.typical_expected_finish - p.expected_finish;
  return (
    <div
      style={{
        marginTop: 16,
        paddingTop: 14,
        borderTop: `1px solid ${C.edge}`,
        display: "flex",
        gap: 18,
        alignItems: "baseline",
        flexWrap: "wrap",
      }}
    >
      <div style={{ fontFamily: F.mono, fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: C.faint }}>
        On this plan
      </div>
      <div style={{ ...NUM, fontSize: 15, color: C.text }}>
        P{p.expected_finish.toFixed(1)}{" "}
        <span style={{ color: C.muted, fontSize: 13 }}>
          in the simulation ({edge >= 0 ? "+" : "−"}
          {Math.abs(edge).toFixed(1)} places)
        </span>
      </div>
      <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, lineHeight: 1.55, flex: "1 1 320px" }}>{p.strategy_edge_note}</div>
    </div>
  );
}

/** The F1 graphic for a tyre: a ring in the compound colour, its initial inside. */
export function TyreIcon({ compound, size = 26 }: { compound: string; size?: number }) {
  const colour = compoundColor(compound);
  return (
    <span
      title={compound}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: size,
        height: size,
        borderRadius: "50%",
        border: `${Math.max(3, size / 7)}px solid ${colour}`,
        background: C.surface,
        fontFamily: F.display,
        fontWeight: 900,
        fontSize: size * 0.42,
        color: C.carbon,
        flex: "none",
      }}
    >
      {compound[0]}
    </span>
  );
}

function StintTimeline({ plan }: { plan: RacePlan }) {
  const W = 1000;
  const H = 150;
  const padX = 20;
  const x = (lap: number) => padX + ((lap - 0) / plan.total_laps) * (W - 2 * padX);
  const stints = plan.compound_sequence.map((compound, i) => ({
    compound,
    from: i === 0 ? 0 : plan.stops[i - 1].nominal_lap,
    to: i < plan.stops.length ? plan.stops[i].nominal_lap : plan.total_laps,
  }));
  // Every 10 laps plus the flag — dropping a 10-lap tick that would sit on
  // top of the flag's label.
  const ticks = Array.from({ length: Math.floor(plan.total_laps / 10) + 1 }, (_, i) => i * 10)
    .filter((t) => t === 0 || plan.total_laps - t >= 6)
    .concat(plan.total_laps);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} role="img" aria-label="Race plan stint timeline">
        {/* Pit windows and caution-hold windows sit behind the stints. */}
        {plan.stops.map((s) => (
          <g key={`w${s.stop_number}`}>
            <rect x={x(s.window_open)} y={20} width={Math.max(x(s.window_close) - x(s.window_open), 3)} height={62} fill={C.fill} rx={4} />
            {s.wait?.has_window && (
              <rect
                x={x(s.wait.open_lap)}
                y={86}
                width={Math.max(x(s.wait.pull_the_plug_lap) - x(s.wait.open_lap), 3)}
                height={10}
                rx={5}
                fill="var(--rq-accent)"
                opacity={0.85}
              />
            )}
          </g>
        ))}
        {stints.map((st, i) => (
          <g key={i}>
            <rect x={x(st.from) + 2} y={36} width={Math.max(x(st.to) - x(st.from) - 4, 4)} height={30} rx={6} fill={compoundColor(st.compound)} />
            <text x={(x(st.from) + x(st.to)) / 2} y={56} textAnchor="middle" fill="#fff" fontFamily={F.mono} fontWeight={700} fontSize={14}>
              {st.compound} · {st.to - st.from} LAPS
            </text>
          </g>
        ))}
        {plan.stops.map((s) => (
          <g key={`n${s.stop_number}`}>
            <line x1={x(s.nominal_lap)} x2={x(s.nominal_lap)} y1={16} y2={86} stroke={C.carbon} strokeWidth={2} />
            <text x={x(s.nominal_lap)} y={12} textAnchor="middle" fill={C.carbon} fontFamily={F.mono} fontWeight={700} fontSize={13}>
              STOP {s.stop_number} · L{s.nominal_lap}
            </text>
          </g>
        ))}
        <line x1={padX} x2={W - padX} y1={112} y2={112} stroke={C.edge} />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={108} y2={116} stroke={C.line} />
            <text x={x(t)} y={134} textAnchor={t === 0 ? "start" : t === plan.total_laps ? "end" : "middle"} fill={C.faint} fontFamily={F.mono} fontWeight={600} fontSize={12}>
              {t === 0 ? "START" : t === plan.total_laps ? `L${t} FLAG` : `L${t}`}
            </text>
          </g>
        ))}
      </svg>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", fontFamily: F.mono, fontWeight: 600, fontSize: 11, letterSpacing: "0.06em", color: C.faint, marginTop: 6 }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 18, height: 10, background: C.fill, borderRadius: 2, border: `1px solid ${C.edge}` }} /> PIT WINDOW
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 18, height: 6, background: "var(--rq-accent)", borderRadius: 3 }} /> HOLD FOR A CAUTION
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 2, height: 12, background: C.carbon }} /> BEST LAP TO STOP
        </span>
      </div>
    </div>
  );
}

function StopCard({ stop }: { stop: PlannedStop }) {
  const w = stop.wait;
  return (
    <div style={{ background: C.raised, border: `1px solid ${C.edge}`, borderRadius: 8, padding: "14px 16px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ ...DISPLAY, fontSize: 18 }}>Stop {stop.stop_number}</div>
        <TyreIcon compound={stop.compound} size={24} />
        <div style={{ fontFamily: F.mono, fontWeight: 700, fontSize: 12, color: C.dim }}>{stop.compound}</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
        <Stat label="Pit window" value={`L${stop.window_open}–${stop.window_close}`} size={20} />
        <Stat label="Best lap" value={`L${stop.nominal_lap}`} size={20} />
      </div>
      <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.rule}` }}>
        <div style={{ ...labelStyle, marginBottom: 4 }}>Safety car / VSC call</div>
        {w?.has_window ? (
          <div style={{ fontFamily: F.body, fontSize: 13.5, color: C.text, lineHeight: 1.55 }}>
            Hold out for a caution from <b>L{w.open_lap}</b> to <b>L{w.pull_the_plug_lap}</b>, then pull the plug and pit.
            <span style={{ color: C.muted }}>
              {" "}
              A caution saves ~{w.caution_saving_seconds.toFixed(0)}s; chance of one on any lap here ≈ {pct(w.per_lap_caution_probability)}.
            </span>
          </div>
        ) : (
          <div style={{ fontFamily: F.body, fontSize: 13.5, color: C.text, lineHeight: 1.55 }}>
            Don't wait — pit in the window.{" "}
            <span style={{ color: C.muted }}>{w?.reason ?? "No caution window worth holding for."}</span>
          </div>
        )}
      </div>
    </div>
  );
}

const SOURCE_LABEL: Record<RacePlan["conditions"]["source"], string> = {
  measured: "Measured on the day",
  forecast: "Forecast · Open-Meteo",
  typical: "Typical for this circuit",
  override: "Set by hand",
};

function Conditions({ plan }: { plan: RacePlan }) {
  const c = plan.conditions;
  const tempDelta = c.circuit_baseline_track_temp != null ? c.track_temp - c.circuit_baseline_track_temp : null;
  return (
    <Card style={{ padding: "18px 20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, marginBottom: 14 }}>
        <SectionLabel>Conditions</SectionLabel>
        <span style={{ fontFamily: F.mono, fontWeight: 700, fontSize: 10.5, letterSpacing: "0.08em", color: C.dim, border: `1px solid ${C.edge}`, borderRadius: 999, padding: "3px 10px" }}>
          {SOURCE_LABEL[c.source].toUpperCase()}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 18 }}>
        <Stat label="Track temp" value={c.track_temp.toFixed(0)} unit={tempDelta == null ? "°C" : `°C · ${tempDelta >= 0 ? "+" : ""}${tempDelta.toFixed(0)} vs usual`} size={24} />
        <Stat label="Air temp" value={c.air_temp.toFixed(0)} unit="°C" size={24} />
        <Stat
          label="Weather"
          value={c.rain_expected ? "Rain" : "Dry"}
          unit={c.rain_probability != null ? `${Math.round(c.rain_probability * 100)}% chance` : undefined}
          size={24}
          color={c.rain_expected ? "#1F5FD1" : undefined}
        />
        <Stat label="Safety car history" value={c.historical_sc_rate == null ? "—" : pct(c.historical_sc_rate)} unit={c.historical_sc_rate == null ? "" : "of races"} size={24} />
        <Stat
          label="Overtaking"
          value={c.historical_overtaking_rate == null ? "—" : c.historical_overtaking_rate < 0.12 ? "Hard" : c.historical_overtaking_rate > 0.25 ? "Easy" : "Average"}
          size={24}
        />
        <Stat label="Pit lane loss" value={c.pit_loss_seconds.toFixed(1)} unit="s" size={24} />
      </div>
      <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, marginTop: 14, lineHeight: 1.55 }}>{c.note}</div>
      {c.cold_start_circuit && (
        <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, marginTop: 8, lineHeight: 1.55 }}>
          <b style={{ color: C.text }}>New venue.</b> No previous race here in this data, so safety-car likelihood, overtaking
          difficulty and the usual track temperature have no history behind them — grid-wide averages stand in.
          {!plan.laps_known && ` The lap count (${plan.total_laps}) is the calendar's typical race length, not this circuit's.`}
        </div>
      )}
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.faint, marginTop: 8, lineHeight: 1.55 }}>
        Track temperature changes the safety-car picture, not the tyre plan: across every era in this data, hotter races show no
        measurable change in tyre wear once Pirelli has picked the compounds for the circuit (harder ones for the hot, abrasive tracks).
      </div>
    </Card>
  );
}

function StartingOptions({ plan }: { plan: RacePlan }) {
  const best = plan.starting_options[0];
  return (
    <Card style={{ padding: "18px 20px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, marginBottom: 12 }}>
        <SectionLabel>Starting tyre</SectionLabel>
        {!plan.compound_choice_is_decisive && (
          <span style={{ fontFamily: F.mono, fontWeight: 700, fontSize: 10.5, letterSpacing: "0.08em", color: C.amber }}>
            ▲ TOO CLOSE TO CALL
          </span>
        )}
      </div>
      {plan.starting_options.map((o) => (
        <div
          key={o.starting_compound}
          style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto", gap: 14, alignItems: "center", padding: "10px 0", borderTop: `1px solid ${C.rule}` }}
        >
          <TyreIcon compound={o.starting_compound} size={26} />
          <div>
            <div style={{ fontFamily: F.body, fontWeight: 700, fontSize: 13.5, color: C.text }}>Start on {o.starting_compound.toLowerCase()}</div>
            <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted }}>{o.plan}</div>
          </div>
          <div style={{ ...NUM, fontSize: 13, color: C.text, textAlign: "right" }}>P{o.expected_finish.toFixed(1)}</div>
          <div style={{ ...NUM, fontSize: 13, color: o === best ? C.text : C.muted, textAlign: "right", minWidth: 64 }}>
            {o.expected_points.toFixed(1)} pts
          </div>
        </div>
      ))}
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginTop: 8, lineHeight: 1.55 }}>
        {plan.compound_choice_is_decisive
          ? "The best start beats the next by more than simulation noise."
          : "These are within simulation noise of each other — the start tyre isn't what decides this race; the pit windows are."}
      </div>
    </Card>
  );
}

function TyreSets({ plan }: { plan: RacePlan }) {
  const t = plan.tyres;
  const group = (title: string, counts: Record<string, number>, note: string) => (
    <div style={{ padding: "12px 0", borderTop: `1px solid ${C.rule}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div style={{ fontFamily: F.body, fontWeight: 700, fontSize: 13.5 }}>{title}</div>
        <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted }}>{note}</div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
        {Object.entries(counts).flatMap(([compound, n]) => Array.from({ length: n }, (_, i) => <TyreIcon key={`${compound}${i}`} compound={compound} size={28} />))}
        {Object.values(counts).every((n) => n === 0) && <span style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted }}>None</span>}
      </div>
    </div>
  );
  return (
    <Card style={{ padding: "18px 20px" }}>
      <SectionLabel style={{ marginBottom: 6 }}>Tyre sets for the weekend</SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, marginBottom: 6, lineHeight: 1.55 }}>
        {Object.values(t.allocation).reduce((a, b) => a + b, 0)} dry sets. The race plan and qualifying are reserved first; what's left is
        the practice budget.
      </div>
      {group("Keep for the race", t.race_reserved, "follows the plan above")}
      {group("Keep for qualifying", t.quali_reserved, "fresh softs per segment")}
      {group("Spend in practice", t.practice_budget, "everything else")}
      {t.used_in_practice.length > 0 && (
        <div style={{ paddingTop: 12, borderTop: `1px solid ${C.rule}` }}>
          <div style={{ fontFamily: F.body, fontWeight: 700, fontSize: 13.5, marginBottom: 8 }}>What was actually used before the race</div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {t.used_in_practice.map((s, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: F.mono, fontWeight: 600, fontSize: 11, color: C.dim }}>
                <TyreIcon compound={s.compound} size={22} /> {s.laps_run}L · {s.sessions.join("+")}
              </div>
            ))}
          </div>
        </div>
      )}
      {t.warnings.map((w) => (
        <div key={w} style={{ fontFamily: F.body, fontSize: 12.5, color: C.amber, marginTop: 10 }}>
          ▲ {w}
        </div>
      ))}
    </Card>
  );
}

export function PlannerView() {
  const state = useRaceSelection();
  const { selection, race } = state;
  const [grid, setGrid] = useState<number | null>(null);
  const [trackTemp, setTrackTemp] = useState<number | null>(null);
  const [rain, setRain] = useState<boolean | null>(null);
  // What-ifs belong to one race and driver.
  useEffect(() => {
    setGrid(null);
    setTrackTemp(null);
    setRain(null);
  }, [selection.raceId, selection.driverId]);

  const plan = useAsync(
    () => api.racePlan(selection.raceId, selection.driverId, { grid, trackTemp, rain }),
    [selection.raceId, selection.driverId, grid, trackTemp, rain],
    Boolean(selection.raceId && selection.driverId),
  );
  const p = plan.data;
  const driver = state.drivers.data?.find((d) => d.driver_id === selection.driverId);

  return (
    <div style={accentVars(teamAccent(p?.team_id ?? driver?.constructor_id ?? null))}>
      <SectionLabel>Race weekend planner</SectionLabel>
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", margin: "8px 0 4px" }}>
        <h1 style={{ ...DISPLAY, fontSize: 38, margin: 0 }}>{race?.name ?? "Race weekend"}</h1>
        {race && (
          <span
            style={{
              fontFamily: F.mono,
              fontWeight: 700,
              fontSize: 11,
              letterSpacing: "0.1em",
              padding: "5px 12px",
              borderRadius: 999,
              background: race.has_results ? C.fill : C.carbon,
              color: race.has_results ? C.dim : "#fff",
            }}
          >
            {race.has_results ? "COMPLETED" : `UPCOMING · ${daysUntil(race.date)}`}
          </span>
        )}
      </div>
      <div style={{ fontFamily: F.body, fontSize: 14, color: C.muted, marginBottom: 18 }}>
        {race ? `${race.circuit_name} · ${race.country} · ${new Date(race.date).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "long" })}` : ""}
        {p ? ` · ${p.total_laps} laps${p.laps_known ? "" : " (assumed)"}` : ""}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 14, alignItems: "start" }}>
        <RaceSelector state={state} showLap={false} />
        <div style={{ display: "flex", gap: 14 }}>
          <div style={{ background: C.surface, border: `1px solid ${C.edge}`, borderRadius: 8, padding: "14px 18px" }}>
            <label style={labelStyle}>Grid slot</label>
            <select value={grid ?? ""} onChange={(e) => setGrid(e.target.value ? Number(e.target.value) : null)} style={{ ...controlStyle, width: 190 }}>
              <option value="">
                {race?.has_results
                  ? `Real grid${p?.actual_grid_position ? ` — P${p.actual_grid_position}` : ""}`
                  : `Expected${p && grid == null ? ` P${p.grid_position}` : ""}`}
              </option>
              {Array.from({ length: 22 }, (_, i) => i + 1).map((g) => (
                <option key={g} value={g}>
                  What if P{g}?
                </option>
              ))}
            </select>
          </div>
          <div style={{ background: C.surface, border: `1px solid ${C.edge}`, borderRadius: 8, padding: "14px 18px" }}>
            <label style={labelStyle}>What-if conditions</label>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                type="number"
                min={10}
                max={70}
                placeholder={p ? `${p.conditions.track_temp.toFixed(0)}°C` : "°C"}
                value={trackTemp ?? ""}
                onChange={(e) => setTrackTemp(e.target.value === "" ? null : Number(e.target.value))}
                aria-label="Track temperature"
                style={{ ...controlStyle, ...NUM, width: 86 }}
              />
              <select
                value={rain == null ? "" : rain ? "rain" : "dry"}
                onChange={(e) => setRain(e.target.value === "" ? null : e.target.value === "rain")}
                aria-label="Weather"
                style={{ ...controlStyle, width: 124 }}
              >
                <option value="">{p ? `${p.conditions.rain_expected ? "Rain" : "Dry"} · as is` : "Weather"}</option>
                <option value="dry">Dry</option>
                <option value="rain">Rain</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {plan.error && <ErrorState message={plan.error} />}
      {plan.loading && (
        <Card style={{ padding: "40px 24px", textAlign: "center" }}>
          <div style={{ ...DISPLAY, fontSize: 22 }}>Planning the race…</div>
          <div style={{ fontFamily: F.body, fontSize: 13, color: C.muted, marginTop: 8 }}>
            Simulating every legal strategy from each starting tyre — about 150,000 races, a few seconds.
          </div>
        </Card>
      )}

      {p && !plan.loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {p.conditions.rain_expected && (
            <Card accent="#1F5FD1" style={{ padding: "14px 20px" }}>
              <div style={{ fontFamily: F.body, fontSize: 13.5, color: C.text, lineHeight: 1.55 }}>
                <b>Rain {p.conditions.rain_probability != null ? `likely (${Math.round(p.conditions.rain_probability * 100)}%)` : "expected"}.</b>{" "}
                This is the dry baseline plan — intermediates and wets are a call made on the day, when the track actually gets wet.
                Rain raises the chance of a safety car, which the hold windows below already account for.
              </div>
            </Card>
          )}
          <Card accent="var(--rq-accent)" style={{ padding: "22px 26px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
              <div>
                <div style={{ ...labelStyle, marginBottom: 6 }}>
                  {driver ? `${driver.given_name} ${driver.family_name}` : p.driver_id} · starts P{p.grid_position}
                  {grid != null ? " (what-if)" : p.grid_is_expected ? " (expected — qualifying not run)" : ""}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                  {p.compound_sequence.map((c, i) => (
                    <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
                      {i > 0 && <span style={{ fontFamily: F.mono, fontWeight: 700, color: C.faint }}>→ L{p.stops[i - 1].nominal_lap} →</span>}
                      <TyreIcon compound={c} size={34} />
                    </span>
                  ))}
                </div>
                <div style={{ ...DISPLAY, fontSize: 24, marginTop: 12 }}>
                  {p.stops.length === 0 ? "No stop" : `${p.stops.length}-stop`} · start on {p.starting_compound.toLowerCase()}
                </div>
              </div>
              <Outcome plan={p} />
            </div>
            <PlanEdge plan={p} />
          </Card>

          <Card style={{ padding: "20px 22px" }}>
            <SectionLabel style={{ marginBottom: 12 }}>The plan, lap by lap</SectionLabel>
            <StintTimeline plan={p} />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14, marginTop: 18 }}>
              {p.stops.map((s) => (
                <StopCard key={s.stop_number} stop={s} />
              ))}
            </div>
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 18, alignItems: "start" }}>
            <Conditions plan={p} />
            <StartingOptions plan={p} />
          </div>
          <TyreSets plan={p} />

          <div style={{ fontFamily: F.body, fontSize: 12, color: C.faint, lineHeight: 1.6 }}>
            {p.n_simulations.toLocaleString()} simulated races per finalist strategy, lap by lap with traffic and track position. Every car's
            pace is its season form{p.qualifying_in_pace ? ", sharpened by this weekend's qualifying" : " (qualifying not run yet)"}; rivals
            pit like real cars here ({p.rival_stops_source === "rule" ? "no race history at this circuit, so when their tyres run out" : p.rival_stops_source}).
            Pace differences between strategies come from measured tyre wear for this season.
            {p.is_future ? " Planned before the race: the entry list, form and car profiles are as they stand after the latest round." : ""}
          </div>
        </div>
      )}
    </div>
  );
}
