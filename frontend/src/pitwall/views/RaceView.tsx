/**
 * Race View — PRD Section 13.2's first named view: "live or historical
 * race state ... current strategy recommendation with simulation outcome
 * distribution".
 *
 * Layout follows §5's rhythm: one wide hero card establishing identity and
 * the headline numbers, then detail sections below it — never
 * re-litigating who this is in every panel.
 *
 * The page is themed to the driver's constructor colour via §2's
 * accentVars: the whole subtree re-skins with no prop drilling, so a
 * Ferrari strategy call reads Ferrari red and a McLaren one papaya,
 * without a single conditional inside the components.
 */

import { useState } from "react";
import { api } from "../../api/client";
import type { ScoredStrategy } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, ErrorState, SectionLabel, Stat, TableSkeleton, tierColor } from "../../design/primitives";
import { accentVars, teamAccent } from "../../design/theme";
import { C, DISPLAY, F, NUM, compoundColor } from "../../design/tokens";
import { SPRING } from "../../design/Animations";
import { FinishDistribution } from "../components/FinishDistribution";
import { RaceSelector, useRaceSelection } from "../components/RaceSelector";
import { StrategyLab } from "./SimulationView";

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** A pit plan rendered as compound chips rather than the raw label string. */
function PitPlanChips({ plan }: { plan: [number, string][] }) {
  if (plan.length === 0) {
    return (
      <span style={{ fontFamily: F.mono, fontSize: 11, color: C.muted, letterSpacing: "0.1em" }}>
        NO FURTHER STOPS
      </span>
    );
  }
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
      {plan.map(([lap, compound], i) => (
        <div key={`${lap}-${compound}-${i}`} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {i > 0 && <span style={{ color: C.faint, fontSize: 11 }}>→</span>}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              background: C.raised,
              border: `1px solid ${C.edge}`,
              borderRadius: 3,
              padding: "5px 10px",
            }}
          >
            <span style={{ ...NUM, fontSize: 11, color: C.dim }}>L{lap}</span>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: compoundColor(compound),
                display: "inline-block",
              }}
            />
            <span style={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.12em", color: C.text }}>
              {compound}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function AlternativesTable({
  alternatives,
  topScore,
}: {
  alternatives: ScoredStrategy[];
  topScore: number;
}) {
  if (alternatives.length === 0) return <EmptyState>NO ALTERNATIVE STRATEGIES SCORED</EmptyState>;

  const headers = ["Strategy", "Win", "Podium", "Exp. finish", "Exp. pts", "vs pick"];
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th
                key={h}
                style={{
                  textAlign: i === 0 ? "left" : "right",
                  fontFamily: F.mono,
                  fontSize: 9,
                  letterSpacing: "0.2em",
                  textTransform: "uppercase",
                  color: C.faint,
                  padding: "0 12px 10px",
                  fontWeight: 400,
                  whiteSpace: "nowrap",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {alternatives.map((alt, i) => (
            <tr key={alt.action} style={{ borderTop: `1px solid ${C.rule}` }}>
              <td style={{ padding: "12px", minWidth: 240 }}>
                <PitPlanChips plan={alt.pit_plan} />
              </td>
              <td style={{ ...NUM, padding: 12, textAlign: "right", fontSize: 12, color: C.text }}>
                {pct(alt.win_probability)}
              </td>
              <td style={{ ...NUM, padding: 12, textAlign: "right", fontSize: 12, color: C.dim }}>
                {pct(alt.podium_probability)}
              </td>
              <td style={{ ...NUM, padding: 12, textAlign: "right", fontSize: 12, color: C.dim }}>
                P{alt.expected_finish.toFixed(1)}
              </td>
              <td style={{ ...NUM, padding: 12, textAlign: "right", fontSize: 12, color: C.dim }}>
                {alt.expected_points.toFixed(1)}
              </td>
              {/* Shown relative to the engine's pick, not on an absolute
                  scale: strategy_score is a weighted blend that naturally
                  lands in a narrow low band (~0.02-0.20), so colouring it
                  against a 0-1 quality ramp would paint every row red and
                  say nothing. What a strategist actually wants here is
                  "how much worse than the pick is this?". */}
              <td
                style={{
                  ...NUM,
                  padding: 12,
                  textAlign: "right",
                  fontSize: 12,
                  color: tierColor(topScore > 0 ? alt.strategy_score / topScore : 0),
                  animation: `fadeUp 0.4s ease both`,
                  animationDelay: `${i * 0.04}s`,
                }}
              >
                {topScore > 0 ? `${((alt.strategy_score / topScore) * 100).toFixed(0)}%` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The cross-check panel. This is RaceIQ-specific and worth surfacing
 * prominently: the Win Probability and Final Race Position classifiers
 * are trained independently of the Monte Carlo simulation, so a large
 * disagreement is a genuine signal — it's how this project found a real
 * simulator bug where a race leader was being modelled as likely to lose.
 */
function CrossCheck({
  label,
  simulation,
  model,
  format,
  disagreement,
}: {
  label: string;
  simulation: number;
  model: number;
  format: (v: number) => string;
  disagreement: number;
}) {
  const gap = Math.abs(simulation - model);
  const warn = gap >= disagreement;
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "10px 0" }}>
      <div style={{ fontFamily: F.mono, fontSize: 10, color: C.dim, letterSpacing: "0.1em" }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <span style={{ ...NUM, fontSize: 13, color: C.text }}>{format(simulation)}</span>
        <span style={{ fontSize: 10, color: C.faint }}>vs</span>
        <span style={{ ...NUM, fontSize: 13, color: warn ? C.red : C.dim }}>{format(model)}</span>
        <span
          style={{
            fontFamily: F.mono,
            fontSize: 8.5,
            letterSpacing: "0.14em",
            padding: "3px 7px",
            borderRadius: 2,
            background: warn ? "rgba(220,38,38,0.12)" : "rgba(22,163,74,0.1)",
            color: warn ? C.red : C.green,
          }}
        >
          {warn ? "DISAGREE" : "AGREE"}
        </span>
      </div>
    </div>
  );
}

export function RaceView() {
  const state = useRaceSelection();
  const { selection, ready } = state;
  const [nSimulations, setNSimulations] = useState(2000);

  const recommendation = useAsync(
    () =>
      api.optimalStrategy({
        race_id: selection.raceId,
        lap_number: selection.lap,
        driver_id: selection.driverId,
        n_simulations: nSimulations,
      }),
    [selection.raceId, selection.lap, selection.driverId, nSimulations],
    ready,
  );

  const rec = recommendation.data;
  const accent = teamAccent(rec?.team_id);
  const top = rec?.recommended_strategy;

  return (
    <div style={accentVars(accent)}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 16, gap: 16, flexWrap: "wrap" }}>
        <div>
          <SectionLabel>Race strategy</SectionLabel>
          <h1 style={{ ...DISPLAY, fontSize: 34, margin: "8px 0 0", letterSpacing: "0.02em" }}>
            {state.race?.name ?? "Strategy call"} — lap {selection.lap}
          </h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.2em", color: C.faint }}>SIMS</span>
          <select
            value={nSimulations}
            onChange={(e) => setNSimulations(Number(e.target.value))}
            style={{
              background: C.raised,
              color: C.text,
              border: `1px solid ${C.edge}`,
              borderRadius: 4,
              padding: "7px 10px",
              fontFamily: F.mono,
              fontSize: 11,
            }}
          >
            {[500, 1000, 2000, 5000].map((n) => (
              <option key={n} value={n}>
                {n.toLocaleString()}
              </option>
            ))}
          </select>
        </div>
      </div>

      <RaceSelector state={state} />

      {!ready && <EmptyState>SELECT A RACE, DRIVER AND LAP TO RUN THE ENGINE</EmptyState>}
      {recommendation.error && <ErrorState message={recommendation.error} />}
      {ready && recommendation.loading && <TableSkeleton rows={6} />}

      {rec && top && !recommendation.loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {/* Hero card — identity established once, at the top. */}
          <Card accent={accent} style={{ padding: "22px 26px" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 32, alignItems: "flex-start" }}>
              <div style={{ minWidth: 220, flex: "1 1 240px" }}>
                <SectionLabel>Recommended action</SectionLabel>
                <div style={{ marginTop: 14, marginBottom: 16 }}>
                  <PitPlanChips plan={top.pit_plan} />
                </div>
                <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                  <span style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, letterSpacing: "0.12em" }}>
                    {rec.driver_id.toUpperCase()} · {rec.team_id.toUpperCase()}
                  </span>
                  <span style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, letterSpacing: "0.12em" }}>
                    LAP {rec.current_lap} · {rec.laps_remaining} REMAINING
                  </span>
                </div>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(96px, 1fr))",
                  gap: 26,
                  flex: "2 1 460px",
                }}
              >
                <Stat label="Win" value={pct(top.win_probability)} color={accent} />
                <Stat label="Podium" value={pct(top.podium_probability)} />
                <Stat label="Points" value={pct(top.points_probability)} />
                <Stat label="Exp. finish" value={`P${top.expected_finish.toFixed(1)}`} />
                <Stat label="Exp. points" value={top.expected_points.toFixed(1)} />
              </div>
            </div>
          </Card>

          {/* Two-column detail: distribution + reasoning/cross-checks. */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))", gap: 18 }}>
            <Card style={{ padding: "20px 22px" }}>
              <SectionLabel style={{ marginBottom: 16 }}>Outcome distribution</SectionLabel>
              <FinishDistribution distribution={top.finish_distribution} />
              <div
                style={{
                  marginTop: 14,
                  paddingTop: 14,
                  borderTop: `1px solid ${C.rule}`,
                  display: "flex",
                  justifyContent: "space-between",
                  fontFamily: F.mono,
                  fontSize: 10,
                  color: C.faint,
                  letterSpacing: "0.1em",
                }}
              >
                <span>
                  {rec.n_simulations.toLocaleString()} SIMS · {rec.n_candidates_evaluated} CANDIDATES
                </span>
                <span>SC IN {pct(top.safety_car_encounter_rate)} OF RUNS</span>
              </div>
            </Card>

            <Card style={{ padding: "20px 22px" }}>
              <SectionLabel style={{ marginBottom: 14 }}>Independent model cross-check</SectionLabel>
              <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.6, marginBottom: 8 }}>
                The Win Probability and Final Race Position classifiers are trained separately from the
                simulation. Where they disagree with it, one of them is wrong — and that's worth knowing
                before acting on the call.
              </div>
              <CrossCheck
                label="WIN PROBABILITY"
                simulation={top.win_probability}
                model={rec.win_probability_model_estimate}
                format={pct}
                disagreement={0.15}
              />
              <CrossCheck
                label="EXPECTED FINISH"
                simulation={top.expected_finish}
                model={rec.expected_finish_model_estimate}
                format={(v) => `P${v.toFixed(1)}`}
                disagreement={3}
              />

              <SectionLabel style={{ margin: "20px 0 12px" }}>Reasoning</SectionLabel>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 10 }}>
                {rec.reasoning.map((line, i) => (
                  <li
                    key={i}
                    style={{
                      fontFamily: F.body,
                      fontSize: 12.5,
                      lineHeight: 1.6,
                      color: C.dim,
                      paddingLeft: 14,
                      borderLeft: `2px solid ${line.startsWith("Note:") ? C.red : accent}`,
                      animation: "fadeUp 0.4s ease both",
                      animationDelay: `${i * 0.05}s`,
                    }}
                  >
                    {line}
                  </li>
                ))}
              </ul>
            </Card>
          </div>

          <StrategyLab rec={rec} selection={selection} />

          <Card style={{ padding: "20px 22px" }}>
            <SectionLabel style={{ marginBottom: 18 }}>
              Alternatives considered ({rec.n_candidates_evaluated} evaluated)
            </SectionLabel>
            <AlternativesTable alternatives={rec.alternatives} topScore={top.strategy_score} />
          </Card>

          <div
            style={{
              fontFamily: F.mono,
              fontSize: 9.5,
              color: C.faint,
              letterSpacing: "0.12em",
              textAlign: "right",
              animation: `statPop 0.5s ${SPRING} both`,
            }}
          >
            HISTORICAL SNAPSHOT — NO LIVE TELEMETRY FEED (PRD §4)
          </div>
        </div>
      )}
    </div>
  );
}
