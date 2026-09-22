/**
 * The simulated finishing-position distribution.
 *
 * Uses the §8 "alternate view" pattern: the cumulative view is nearly free
 * once the per-position probabilities are already computed and sorted, and
 * it answers a different real question ("what are the odds of P5 or
 * better?" vs "what are the odds of exactly P5?"). One render path
 * parameterised by the view, not two copy-pasted ones.
 *
 * Bar colour is the data, not a fixed brand colour (§9): podium positions
 * in the accent, points positions in carbon, out-of-points light grey.
 */

import { useMemo, useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Segmented } from "../../design/primitives";
import { C, F } from "../../design/tokens";
import { AXIS_LINE, AXIS_TICK, TOOLTIP_STYLE } from "./chartStyle";

function positionColor(position: number): string {
  if (position <= 3) return C.accent;
  if (position <= 10) return C.carbon;
  return C.inactive;
}

export function FinishDistribution({
  distribution,
  height = 230,
}: {
  distribution: Record<string, number>;
  height?: number;
}) {
  const [view, setView] = useState<"exact" | "cumulative">("exact");

  const data = useMemo(() => {
    const rows = Object.entries(distribution)
      .map(([position, probability]) => ({ position: Number(position), probability }))
      .sort((a, b) => a.position - b.position);

    if (view === "exact") return rows;

    let running = 0;
    return rows.map((row) => {
      running += row.probability;
      return { ...row, probability: running };
    });
  }, [distribution, view]);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <div style={{ fontFamily: F.mono, fontWeight: 600, fontSize: 11, letterSpacing: "0.08em", color: C.faint }}>
          {view === "exact" ? "P(FINISH EXACTLY HERE)" : "P(FINISH HERE OR BETTER)"}
        </div>
        <Segmented
          value={view}
          onChange={(v) => setView(v as "exact" | "cumulative")}
          options={[
            { value: "exact", label: "Exact" },
            { value: "cumulative", label: "Cumulative" },
          ]}
        />
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
          <XAxis
            dataKey="position"
            tick={AXIS_TICK}
            axisLine={AXIS_LINE}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={AXIS_TICK}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: C.fill }}
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(position) => `P${position}`}
            formatter={(value) => [
              `${(Number(value) * 100).toFixed(1)}%`,
              view === "exact" ? "Probability" : "Cumulative",
            ]}
          />
          <Bar dataKey="probability" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {data.map((row) => (
              <Cell key={row.position} fill={positionColor(row.position)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
