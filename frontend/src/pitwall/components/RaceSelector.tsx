/**
 * The shared "which lap of which race, for which driver" control used by
 * every view that asks the strategy engine a question.
 *
 * All three prediction endpoints take the same (race_id, lap_number,
 * driver_id) pointer into real historical data — deliberately, per
 * serving/api/routers/predictions.py: there's no live telemetry feed, so
 * "race state" means a real lap that actually happened. This control is
 * the UI being honest about that rather than dressing a historical
 * snapshot up as a live timing screen.
 */

import { useMemo, useState } from "react";
import { api } from "../../api/client";
import type { Driver, Race } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { C, F, NUM } from "../../design/tokens";

export type RaceSelection = {
  season: number;
  raceId: string;
  lap: number;
  driverId: string;
};

const SEASONS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018];

export function useRaceSelection(initial?: Partial<RaceSelection>) {
  // Defaults to a real, verified scenario: Bahrain 2025 lap 20, where
  // piastri leads. Every strategy-engine fix this project made was checked
  // against this exact snapshot, so it's the one state guaranteed to show
  // the dashboard working rather than an empty shell.
  const [season, setSeason] = useState(initial?.season ?? 2025);
  const [raceId, setRaceId] = useState(initial?.raceId ?? "2025_4");
  const [lap, setLap] = useState(initial?.lap ?? 20);
  const [driverId, setDriverId] = useState(initial?.driverId ?? "piastri");

  const races = useAsync(() => api.races(season), [season]);
  const drivers = useAsync(() => api.drivers(season), [season]);

  const changeSeason = (next: number) => {
    setSeason(next);
    setRaceId("");
    setDriverId("");
  };

  return {
    selection: { season, raceId, lap, driverId } as RaceSelection,
    races,
    drivers,
    setSeason: changeSeason,
    setRaceId,
    setLap,
    setDriverId,
    ready: Boolean(raceId && driverId && lap > 0),
  };
}

const labelStyle = {
  fontFamily: F.mono,
  fontSize: 9,
  letterSpacing: "0.22em",
  textTransform: "uppercase" as const,
  color: C.faint,
  marginBottom: 7,
  display: "block",
};

const controlStyle = {
  background: C.raised,
  color: C.text,
  border: `1px solid ${C.edge}`,
  borderRadius: 4,
  padding: "9px 11px",
  fontFamily: F.mono,
  fontSize: 12,
  width: "100%",
  outline: "none",
};

export function RaceSelector({
  state,
  onRun,
  running,
}: {
  state: ReturnType<typeof useRaceSelection>;
  onRun?: () => void;
  running?: boolean;
}) {
  const { selection, races, drivers, setSeason, setRaceId, setLap, setDriverId } = state;

  const sortedRaces = useMemo(
    () => [...(races.data ?? [])].sort((a: Race, b: Race) => a.round - b.round),
    [races.data],
  );
  const sortedDrivers = useMemo(
    () =>
      [...(drivers.data ?? [])].sort((a: Driver, b: Driver) =>
        (a.family_name ?? a.driver_id).localeCompare(b.family_name ?? b.driver_id),
      ),
    [drivers.data],
  );

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
        gap: 14,
        alignItems: "end",
        background: C.surface,
        border: `1px solid ${C.edge}`,
        borderRadius: 6,
        padding: "16px 18px",
        marginBottom: 20,
      }}
    >
      <div>
        <label style={labelStyle}>Season</label>
        <select value={selection.season} onChange={(e) => setSeason(Number(e.target.value))} style={controlStyle}>
          {SEASONS.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div style={{ gridColumn: "span 2" }}>
        <label style={labelStyle}>Race</label>
        <select
          value={selection.raceId}
          onChange={(e) => setRaceId(e.target.value)}
          style={controlStyle}
          disabled={races.loading}
        >
          <option value="">{races.loading ? "Loading…" : "Select a race"}</option>
          {sortedRaces.map((r) => (
            <option key={r.race_id} value={r.race_id}>
              R{String(r.round).padStart(2, "0")} — {r.name}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label style={labelStyle}>Driver</label>
        <select
          value={selection.driverId}
          onChange={(e) => setDriverId(e.target.value)}
          style={controlStyle}
          disabled={drivers.loading}
        >
          <option value="">{drivers.loading ? "Loading…" : "Select a driver"}</option>
          {sortedDrivers.map((d) => (
            <option key={d.driver_id} value={d.driver_id}>
              {d.driver_code ?? d.driver_id} — {d.family_name ?? d.driver_id}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label style={labelStyle}>Lap</label>
        <input
          type="number"
          min={1}
          max={80}
          value={selection.lap}
          onChange={(e) => setLap(Number(e.target.value))}
          style={{ ...controlStyle, ...NUM }}
        />
      </div>

      {onRun && (
        <button
          onClick={onRun}
          disabled={running}
          style={{
            background: running ? C.raised : C.accent,
            color: running ? C.faint : "var(--rq-on-accent, #fff)",
            border: "none",
            borderRadius: 4,
            padding: "11px 18px",
            fontFamily: F.mono,
            fontSize: 10,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            cursor: running ? "wait" : "pointer",
            transition: "all 200ms",
          }}
        >
          {running ? "Simulating…" : "Run"}
        </button>
      )}
    </div>
  );
}
