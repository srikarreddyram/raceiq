/**
 * Strategy Simulation View — PRD Section 13.2: "candidate strategies
 * side-by-side ... risk score vs expected points scatter".
 *
 * The scatter is the honest shape for this data: strategy choice is a
 * trade, and the engine's own scoring function weights expected points
 * against variance rather than maximising either alone. Plotting both axes
 * shows you the trade instead of hiding it behind a single ranked number.
 *
 * PRD also names box plots for the outcome distributions. Recharts has no
 * box plot primitive and hand-rolling one here would duplicate what the
 * Race View's distribution chart already shows well, so this shows the
 * spread via the risk axis instead — noted rather than silently dropped.
 */

import { useState } from "react";
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { api } from "../../api/client";
import type { PitPlan, SimulationOutcome } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, ErrorState, SectionLabel, Stat, TableSkeleton } from "../../design/primitives";
import { accentVars, teamAccent } from "../../design/theme";
import { compoundColor, C, F, NUM } from "../../design/tokens";
import { FinishDistribution } from "../components/FinishDistribution";
import { RaceSelector, useRaceSelection } from "../components/RaceSelector";

const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];

type PlottedStrategy = {
  action: string;
  risk: number;
  points: number;
  win: number;
  recommended: boolean;
};

function StrategyTooltip({ payload }: { payload?: { payload: PlottedStrategy }[] }) {
  const point = payload?.[0]?.payload;
  if (!point) return null;
  return (
    <div
      style={{
        background: C.raised,
        border: `1px solid ${C.edge}`,
        borderLeft: `3px solid ${point.recommended ? C.gold : C.edge}`,
        borderRadius: 4,
        padding: "10px 12px",
        fontFamily: F.mono,
        fontSize: 11,
        maxWidth: 280,
      }}
    >
      <div style={{ color: C.text, marginBottom: 7, lineHeight: 1.45 }}>{point.action}</div>
      {point.recommended && (
        <div style={{ color: C.gold, fontSize: 8.5, letterSpacing: "0.18em", marginBottom: 7 }}>
          ENGINE'S PICK
        </div>
      )}
      <div style={{ color: C.dim }}>
        {point.points.toFixed(2)} pts · risk {point.risk.toFixed(1)} · {(point.win * 100).toFixed(1)}% win
      </div>
    </div>
  );
}

function PitPlanBuilder({
  plan,
  onChange,
  maxLap,
}: {
  plan: PitPlan;
  onChange: (plan: PitPlan) => void;
  maxLap: number;
}) {
  const update = (index: number, lap: number, compound: string) => {
    const next = [...plan];
    next[index] = [lap, compound];
    onChange(next);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {plan.map(([lap, compound], i) => (
        <div key={i} style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontFamily: F.mono, fontSize: 9, color: C.faint, width: 42 }}>STOP {i + 1}</span>
          <input
            type="number"
            min={1}
            max={maxLap}
            value={lap}
            onChange={(e) => update(i, Number(e.target.value), compound)}
            style={{
              ...NUM,
              width: 72,
              background: C.raised,
              color: C.text,
              border: `1px solid ${C.edge}`,
              borderRadius: 3,
              padding: "7px 9px",
              fontSize: 12,
            }}
          />
          <select
            value={compound}
            onChange={(e) => update(i, lap, e.target.value)}
            style={{
              background: C.raised,
              color: compoundColor(compound),
              border: `1px solid ${C.edge}`,
              borderRadius: 3,
              padding: "7px 9px",
              fontFamily: F.mono,
              fontSize: 11,
              flex: 1,
            }}
          >
            {COMPOUNDS.map((c) => (
              <option key={c} value={c} style={{ color: C.text }}>
                {c}
              </option>
            ))}
          </select>
          <button
            onClick={() => onChange(plan.filter((_, j) => j !== i))}
            aria-label={`Remove stop ${i + 1}`}
            style={{
              background: "transparent",
              border: `1px solid ${C.edge}`,
              color: C.muted,
              borderRadius: 3,
              width: 30,
              height: 32,
              cursor: "pointer",
              fontSize: 14,
            }}
          >
            ×
          </button>
        </div>
      ))}

      <button
        onClick={() => onChange([...plan, [Math.min(maxLap, 30), "HARD"]])}
        style={{
          background: "transparent",
          border: `1px dashed ${C.edge}`,
          color: C.dim,
          borderRadius: 3,
          padding: "9px 12px",
          fontFamily: F.mono,
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          cursor: "pointer",
        }}
      >
        + Add stop
      </button>
      {plan.length === 0 && (
        <div style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, letterSpacing: "0.1em" }}>
          EMPTY PLAN = RUN TO THE FLAG ON CURRENT TYRES
        </div>
      )}
    </div>
  );
}

export function SimulationView() {
  const state = useRaceSelection();
  const { selection, ready } = state;
  const [customPlan, setCustomPlan] = useState<PitPlan>([[30, "HARD"]]);
  const [customResult, setCustomResult] = useState<SimulationOutcome | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const recommendation = useAsync(
    () =>
      api.optimalStrategy({
        race_id: selection.raceId,
        lap_number: selection.lap,
        driver_id: selection.driverId,
        n_simulations: 1000,
      }),
    [selection.raceId, selection.lap, selection.driverId],
    ready,
  );

  const rec = recommendation.data;
  const accent = teamAccent(rec?.team_id);

  const runCustom = async () => {
    setRunning(true);
    setCustomError(null);
    try {
      const result = await api.simulate({
        race_id: selection.raceId,
        lap_number: selection.lap,
        driver_id: selection.driverId,
        pit_plan: customPlan,
        n_simulations: 1000,
      });
      setCustomResult(result);
    } catch (err) {
      setCustomError(err instanceof Error ? err.message : String(err));
      setCustomResult(null);
    } finally {
      setRunning(false);
    }
  };

  const plotted: PlottedStrategy[] = rec
    ? [
        {
          action: rec.recommended_strategy.action,
          risk: rec.recommended_strategy.risk_score,
          points: rec.recommended_strategy.expected_points,
          win: rec.recommended_strategy.win_probability,
          recommended: true,
        },
        ...rec.alternatives.map((alt) => ({
          action: alt.action,
          risk: alt.risk_score,
          points: alt.expected_points,
          win: alt.win_probability,
          recommended: false,
        })),
      ]
    : [];

  return (
    <div style={accentVars(accent)}>
      <SectionLabel>Strategy simulation</SectionLabel>
      <h1 style={{ fontFamily: F.display, fontSize: 34, margin: "8px 0 16px", letterSpacing: "0.02em" }}>
        Compare the trade
      </h1>

      <RaceSelector state={state} />

      {!ready && <EmptyState>SELECT A RACE, DRIVER AND LAP TO COMPARE STRATEGIES</EmptyState>}
      {recommendation.error && <ErrorState message={recommendation.error} />}
      {ready && recommendation.loading && <TableSkeleton rows={5} />}

      {rec && !recommendation.loading && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 18 }}>
          <Card style={{ padding: "20px 22px" }}>
            <SectionLabel style={{ marginBottom: 6 }}>Risk vs reward</SectionLabel>
            <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.6, marginBottom: 14 }}>
              Risk is the variance of simulated finishing positions. Up and to the left is better: more
              expected points for less spread in where you actually end up.
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: -12 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number"
                  dataKey="risk"
                  name="Risk"
                  tick={{ fill: "rgba(240,240,240,0.38)", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                  axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                  tickLine={false}
                  label={{
                    value: "RISK (POSITION VARIANCE)",
                    position: "insideBottom",
                    offset: -10,
                    fill: "rgba(240,240,240,0.38)",
                    fontSize: 9,
                    fontFamily: "'JetBrains Mono', monospace",
                    letterSpacing: "0.18em",
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="points"
                  name="Expected points"
                  tick={{ fill: "rgba(240,240,240,0.38)", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
                  axisLine={false}
                  tickLine={false}
                />
                <ZAxis type="number" dataKey="win" range={[60, 320]} />
                {/* A custom tooltip rather than recharts' default: the
                    default labels a point by its axis values, and "which
                    strategy is this dot" is the entire question someone
                    hovers a scatter to answer. */}
                <Tooltip cursor={{ stroke: "rgba(255,255,255,0.12)" }} content={<StrategyTooltip />} />
                <Scatter data={plotted} isAnimationActive={false}>
                  {plotted.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.recommended ? accent : "rgba(240,240,240,0.28)"}
                      stroke={entry.recommended ? accent : "transparent"}
                      strokeWidth={entry.recommended ? 2 : 0}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <div style={{ fontFamily: F.mono, fontSize: 9, color: C.faint, letterSpacing: "0.14em", marginTop: 6 }}>
              MARKER SIZE = WIN PROBABILITY · HIGHLIGHTED = ENGINE'S PICK
            </div>
          </Card>

          <Card style={{ padding: "20px 22px" }}>
            <SectionLabel style={{ marginBottom: 6 }}>Test your own call</SectionLabel>
            <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.6, marginBottom: 16 }}>
              Build any pit plan and run it through the same simulation the engine used, against the same
              race state.
            </div>

            <PitPlanBuilder plan={customPlan} onChange={setCustomPlan} maxLap={selection.lap + rec.laps_remaining} />

            <button
              onClick={runCustom}
              disabled={running}
              style={{
                marginTop: 16,
                width: "100%",
                background: running ? C.raised : C.gold,
                color: running ? C.faint : "var(--rq-on-accent, #000)",
                border: "none",
                borderRadius: 4,
                padding: "12px",
                fontFamily: F.mono,
                fontSize: 10,
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                cursor: running ? "wait" : "pointer",
              }}
            >
              {running ? "Simulating…" : "Simulate this plan"}
            </button>

            {customError && <div style={{ marginTop: 14 }}><ErrorState message={customError} /></div>}

            {customResult && (
              <div style={{ marginTop: 20, paddingTop: 18, borderTop: `1px solid ${C.rule}` }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(84px, 1fr))",
                    gap: 18,
                    marginBottom: 20,
                  }}
                >
                  <Stat label="Win" value={`${(customResult.win_probability * 100).toFixed(1)}%`} size={24} color={accent} />
                  <Stat label="Podium" value={`${(customResult.podium_probability * 100).toFixed(1)}%`} size={24} />
                  <Stat label="Exp. finish" value={`P${customResult.expected_finish.toFixed(1)}`} size={24} />
                  <Stat label="Exp. pts" value={customResult.expected_points.toFixed(1)} size={24} />
                </div>

                <div
                  style={{
                    fontFamily: F.mono,
                    fontSize: 10,
                    letterSpacing: "0.12em",
                    color:
                      customResult.expected_points >= rec.recommended_strategy.expected_points ? C.green : C.red,
                    marginBottom: 16,
                  }}
                >
                  {customResult.expected_points >= rec.recommended_strategy.expected_points
                    ? `BEATS THE ENGINE'S PICK BY ${(customResult.expected_points - rec.recommended_strategy.expected_points).toFixed(2)} PTS`
                    : `${(rec.recommended_strategy.expected_points - customResult.expected_points).toFixed(2)} PTS BEHIND THE ENGINE'S PICK`}
                </div>

                <FinishDistribution distribution={customResult.finish_distribution} height={180} />
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
