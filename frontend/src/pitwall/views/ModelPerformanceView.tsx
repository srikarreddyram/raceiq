/**
 * Model Performance View — PRD Section 13.2: live metrics, prediction vs
 * actual overlays, drift indicators per model, and CarProfile inference
 * confidence over the season.
 *
 * "Live" here means the 2026 season: never trained or tested on, growing
 * every race weekend, and scored by monitoring/run.py against each
 * promoted model's own test-season figure. This view reads that job's
 * output; it doesn't evaluate anything itself.
 *
 * Input drift is judged against each feature's own noise, not PSI's
 * textbook thresholds — see monitoring/run.py. A race-level feature like
 * air temperature has 14 independent values in 2026, not 16,000, and
 * shows a huge PSI from sampling alone.
 */

import { useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../../api/client";
import type { ModelHealth } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Card, EmptyState, ErrorState, SectionLabel, TableSkeleton } from "../../design/primitives";
import { C, DISPLAY, F, NUM } from "../../design/tokens";
import { ACCENT, AXIS_LINE, AXIS_TICK, GRID_STROKE, TOOLTIP_STYLE } from "../components/chartStyle";

const METRIC_LABELS: Record<string, string> = {
  auc_roc: "AUC-ROC",
  log_loss: "Log loss",
  top3_accuracy: "Top-3 accuracy",
  mean_abs_position_error: "Position error",
  "precision_at_0.5": "Precision @0.5",
  "recall_at_0.5": "Recall @0.5",
  pr_auc: "PR-AUC",
  mae: "MAE",
  rmse: "RMSE",
  mae_stable_regime: "Stable-lap MAE",
};
const metricLabel = (key: string) => METRIC_LABELS[key.replace(/^current_test_/, "")] ?? key;

function StatusBadge({ status }: { status: ModelHealth["status"] }) {
  // Status is never colour alone: a symbol and a word ride with it.
  const spec = {
    ok: { colour: C.green, text: "✓ HOLDING" },
    warn: { colour: C.red, text: "▲ DRIFTING" },
    skipped: { colour: C.faint, text: "— NO 2026 DATA" },
  }[status];
  return (
    <span
      style={{
        fontFamily: F.mono,
        fontSize: 9,
        letterSpacing: "0.16em",
        color: spec.colour,
        border: `1px solid ${spec.colour}`,
        borderRadius: 3,
        padding: "3px 7px",
        whiteSpace: "nowrap",
      }}
    >
      {spec.text}
    </span>
  );
}

function ModelCard({ m, active, onSelect }: { m: ModelHealth; active: boolean; onSelect: () => void }) {
  const drifted = m.feature_drift.filter((d) => d.drifted).length;
  return (
    <button
      onClick={onSelect}
      style={{
        textAlign: "left",
        background: active ? C.hover : C.surface,
        border: `1px solid ${active ? C.accent : C.edge}`,
        borderRadius: 6,
        padding: "16px 18px",
        cursor: "pointer",
        transition: "all 200ms",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
        <div style={{ ...DISPLAY, fontSize: 17, color: C.text }}>{m.label}</div>
        <StatusBadge status={m.status} />
      </div>
      <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 5 }}>
        {Object.entries(m.metrics).map(([key, v]) => (
          <div key={key} style={{ display: "flex", justifyContent: "space-between", fontFamily: F.mono, fontSize: 11 }}>
            <span style={{ color: C.faint }}>{metricLabel(key)}</span>
            <span style={{ ...NUM, color: v.status === "WARN" ? C.red : C.dim }}>
              {v.current.toFixed(3)}
              <span style={{ color: C.faint }}> / {v.baseline == null ? "—" : v.baseline.toFixed(3)}</span>
            </span>
          </div>
        ))}
      </div>
      <div style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.12em", color: C.faint, marginTop: 12 }}>
        {m.n_races ?? 0} RACES · {drifted} INPUT{drifted === 1 ? "" : "S"} DRIFTED
      </div>
    </button>
  );
}

function PerRace({ m }: { m: ModelHealth }) {
  const data = m.per_race.map((r) => ({ race: `R${r.race_id.split("_")[1]}`, value: r.value, n: r.n_rows }));
  const missing = m.per_race.filter((r) => r.value == null).length;
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 4 }}>
        {m.label} — {metricLabel(m.per_race_metric)} race by race
      </SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        Each 2026 race scored on its own; dashed is the promoted model's figure on its 2025 test season.{" "}
        {m.per_race_lower_is_better ? "Lower is better." : "Higher is better."}
        {missing > 0 && ` ${missing} race${missing === 1 ? "" : "s"} left blank where the metric is undefined (e.g. no safety car to detect).`}
      </div>
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: -6 }}>
          <CartesianGrid stroke={GRID_STROKE} vertical={false} />
          <XAxis dataKey="race" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
          <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={48} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(2)} />
          {m.per_race_baseline != null && (
            <ReferenceLine
              y={m.per_race_baseline}
              stroke={C.muted}
              strokeDasharray="4 4"
              label={{ value: "2025 TEST", position: "insideTopRight", fill: C.faint, fontSize: 9, fontFamily: F.mono }}
            />
          )}
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(value, _n, item) => [
              value == null ? "undefined" : `${Number(value).toFixed(3)} (${item?.payload?.n} rows)`,
              metricLabel(m.per_race_metric),
            ]}
          />
          <Line dataKey="value" stroke={ACCENT} strokeWidth={2} dot={{ r: 3.5, fill: ACCENT }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}

function Drift({ m }: { m: ModelHealth }) {
  const rows = m.feature_drift.slice(0, 12);
  const max = Math.max(...rows.map((d) => Math.max(d.psi, d.null_p95)), 0.1);
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 4 }}>Input drift — 2026 vs training seasons</SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        Bar: PSI of the feature's 2026 distribution. Tick: the 95th percentile of PSI for random sets of the same
        number of training races — the noise floor. Only a bar past its tick is drift.
      </div>
      {rows.length === 0 ? (
        <EmptyState>NO NUMERIC FEATURES WITH ENOUGH DATA TO COMPARE</EmptyState>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {rows.map((d) => (
            <div key={d.feature} style={{ display: "grid", gridTemplateColumns: "190px 1fr 92px", gap: 12, alignItems: "center" }}>
              <span style={{ fontFamily: F.mono, fontSize: 10.5, color: d.drifted ? C.text : C.faint, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {d.feature}
              </span>
              <div style={{ position: "relative", height: 10, background: C.fill, borderRadius: 5 }}>
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: `${(d.psi / max) * 100}%`,
                    background: d.drifted ? ACCENT : C.line,
                    borderRadius: 5,
                  }}
                />
                <div style={{ position: "absolute", top: -3, bottom: -3, left: `${(d.null_p95 / max) * 100}%`, width: 2, background: C.muted }} />
              </div>
              <span style={{ ...NUM, fontSize: 10.5, color: d.drifted ? C.text : C.faint, textAlign: "right" }}>
                {d.psi.toFixed(2)} {d.drifted ? "▲" : ""}
              </span>
            </div>
          ))}
        </div>
      )}
      {m.notes.length > 0 && (
        <div style={{ fontFamily: F.mono, fontSize: 10, color: C.faint, marginTop: 14, letterSpacing: "0.06em" }}>
          {m.notes.map((n) => `NOTE: ${n.toUpperCase()}`).join(" · ")}
        </div>
      )}
    </Card>
  );
}

function LapTimeOverlayCard() {
  const races = useAsync(() => api.races(2026), []);
  const drivers = useAsync(() => api.drivers(2026), []);
  const [raceId, setRaceId] = useState("");
  const [driverId, setDriverId] = useState("leclerc");

  useEffect(() => {
    if (!raceId && races.data?.length) {
      setRaceId([...races.data].sort((a, b) => b.round - a.round)[0].race_id);
    }
  }, [races.data, raceId]);

  const overlay = useAsync(() => api.lapTimeOverlay(raceId, driverId), [raceId, driverId], Boolean(raceId && driverId));
  const select = {
    background: C.raised,
    color: C.text,
    border: `1px solid ${C.edge}`,
    borderRadius: 4,
    padding: "7px 9px",
    fontFamily: F.mono,
    fontSize: 11,
    outline: "none",
  };

  return (
    <Card style={{ padding: "20px 22px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 6 }}>
        <SectionLabel>Lap Time — predicted vs actual</SectionLabel>
        <div style={{ display: "flex", gap: 8 }}>
          <select value={raceId} onChange={(e) => setRaceId(e.target.value)} style={select} aria-label="Race">
            {[...(races.data ?? [])]
              .sort((a, b) => a.round - b.round)
              .map((r) => (
                <option key={r.race_id} value={r.race_id}>
                  R{r.round} {r.name.replace(" Grand Prix", " GP")}
                </option>
              ))}
          </select>
          <select value={driverId} onChange={(e) => setDriverId(e.target.value)} style={select} aria-label="Driver">
            {[...(drivers.data ?? [])]
              .sort((a, b) => (a.family_name ?? "").localeCompare(b.family_name ?? ""))
              .map((d) => (
                <option key={d.driver_id} value={d.driver_id}>
                  {d.family_name}
                </option>
              ))}
          </select>
        </div>
      </div>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        The promoted Lap Time model's prediction of each next lap against what the driver actually did. Laps with a
        pit stop, safety car or flag in play are drawn but not scored — no current-lap feature can foresee them.
        {overlay.data?.mae_stable != null && (
          <span style={{ ...NUM, color: C.text }}> Stable-lap MAE: {overlay.data.mae_stable.toFixed(3)}s.</span>
        )}
      </div>
      {overlay.error && <EmptyState>{overlay.error.toUpperCase()}</EmptyState>}
      {overlay.data && (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={overlay.data.laps} margin={{ top: 6, right: 12, bottom: 0, left: -2 }}>
            <CartesianGrid stroke={GRID_STROKE} vertical={false} />
            <XAxis dataKey="lap_number" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
            <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={52} domain={["dataMin - 1", "dataMax + 1"]} tickFormatter={(v: number) => v.toFixed(0)} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelFormatter={(lap, payload) => `Lap ${lap}${payload?.[0]?.payload?.stable ? "" : " · not scored"}`}
              formatter={(value, name) => [`${Number(value).toFixed(3)}s`, String(name)]}
            />
            <Legend wrapperStyle={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.1em" }} />
            <Line dataKey="actual" name="Actual" stroke={C.muted} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="predicted" name="Predicted" stroke={ACCENT} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

function ProfileConfidence() {
  const data = useAsync(() => api.profileConfidence(2026), []);
  const rows = useMemo(
    () =>
      (data.data ?? []).map((p) => ({
        races: p.races,
        median: p.median_relative_halfwidth,
        band: [p.p25_relative_halfwidth, p.p75_relative_halfwidth] as [number, number],
        distinct: p.distinct_share * 100,
      })),
    [data.data],
  );
  // Two small charts, never one with two y-axes.
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 4 }}>CarProfile confidence over the 2026 season</SectionLabel>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
        Left: each team's 95% interval on each characteristic, as a fraction of how far apart the whole grid ended up —
        1.0 means the uncertainty is as wide as the grid's entire spread. Median and interquartile band across every
        team and characteristic. Right: the share whose interval already excludes the grid mean.
      </div>
      {data.error && <ErrorState message={data.error} />}
      {rows.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 18 }}>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={rows} margin={{ top: 6, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid stroke={GRID_STROKE} vertical={false} />
              <XAxis dataKey="races" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={44} domain={[0, 2]} allowDataOverflow tickFormatter={(v: number) => v.toFixed(1)} />
              <ReferenceLine y={1} stroke={C.muted} strokeDasharray="4 4" />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                labelFormatter={(k) => `After ${k} races`}
                formatter={(value, name) =>
                  Array.isArray(value)
                    ? [`${Number(value[0]).toFixed(2)} – ${Number(value[1]).toFixed(2)}`, "Interquartile"]
                    : [Number(value).toFixed(2), String(name)]
                }
              />
              <Area dataKey="band" stroke="none" fill={ACCENT} fillOpacity={0.14} isAnimationActive={false} />
              <Line dataKey="median" name="Median" stroke={ACCENT} strokeWidth={2} dot={{ r: 3, fill: ACCENT }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rows} margin={{ top: 6, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid stroke={GRID_STROKE} vertical={false} />
              <XAxis dataKey="races" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={44} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={(k) => `After ${k} races`} formatter={(v) => [`${Number(v).toFixed(0)}%`, "Distinct from grid"]} />
              <Line dataKey="distinct" name="Distinct" stroke={C.text} strokeWidth={2} dot={{ r: 3, fill: C.text }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

export function ModelPerformanceView() {
  const health = useAsync(() => api.modelHealth(), []);
  const [selected, setSelected] = useState("lap_time_prediction");
  const models = health.data ?? [];
  const active = models.find((m) => m.experiment === selected) ?? models[0];

  return (
    <div>
      <SectionLabel>Model performance</SectionLabel>
      <h1 style={{ ...DISPLAY, fontSize: 34, margin: "8px 0 8px", letterSpacing: "0.02em" }}>
        How the models are holding up in 2026
      </h1>
      <div style={{ fontFamily: F.mono, fontSize: 10, letterSpacing: "0.14em", color: C.faint, marginBottom: 20 }}>
        {active ? `EVALUATED ${active.evaluated_at.slice(0, 16).replace("T", " ")} UTC · ` : ""}CURRENT / PROMOTED BASELINE ·
        REFRESH WITH MONITORING.RUN
      </div>

      {health.error && <ErrorState message={health.error} />}
      {health.loading && <TableSkeleton rows={6} columns={3} />}

      {models.length > 0 && active && (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: 14 }}>
            {models.map((m) => (
              <ModelCard key={m.experiment} m={m} active={m.experiment === active.experiment} onSelect={() => setSelected(m.experiment)} />
            ))}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(460px, 1fr))", gap: 18, alignItems: "start" }}>
            <PerRace m={active} />
            <Drift m={active} />
          </div>
          <LapTimeOverlayCard />
          <ProfileConfidence />
        </div>
      )}
    </div>
  );
}
