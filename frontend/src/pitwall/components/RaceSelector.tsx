/**
 * The race-weekend control shared by every view that asks the strategy
 * engine a question: which race of the current season, which driver, and
 * (for in-race calls) which lap.
 *
 * It reads and writes the Pit Wall's shared weekend (WeekendContext), so
 * the planner, the strategy call and the simulator stay on the same race.
 * There's no live telemetry feed, so "race state" means a real lap that
 * actually happened — the control is honest about that rather than
 * dressing a completed race up as a live timing screen.
 */

import { useMemo } from "react";
import type { Driver, Race } from "../../api/types";
import { C, F, NUM } from "../../design/tokens";
import { useWeekend } from "../WeekendContext";

export type RaceSelection = {
  season: number | null;
  raceId: string;
  lap: number;
  driverId: string;
};

export function useRaceSelection() {
  const w = useWeekend();
  return {
    selection: { season: w.season, raceId: w.raceId, lap: w.lap, driverId: w.driverId } as RaceSelection,
    races: w.races,
    drivers: w.drivers,
    setRaceId: w.setRaceId,
    setLap: w.setLap,
    setDriverId: w.setDriverId,
    race: w.race,
    ready: Boolean(w.raceId && w.driverId && w.lap > 0),
  };
}

export const labelStyle = {
  fontFamily: F.mono,
  fontWeight: 600,
  fontSize: 10.5,
  letterSpacing: "0.1em",
  textTransform: "uppercase" as const,
  color: C.faint,
  marginBottom: 7,
  display: "block",
};

export const controlStyle = {
  background: C.raised,
  color: C.text,
  border: `1px solid ${C.edge}`,
  borderRadius: 6,
  padding: "9px 11px",
  fontFamily: F.body,
  fontWeight: 600,
  fontSize: 13,
  width: "100%",
  outline: "none",
};

export function RaceSelector({
  state,
  onRun,
  running,
  showLap = true,
}: {
  state: ReturnType<typeof useRaceSelection>;
  onRun?: () => void;
  running?: boolean;
  showLap?: boolean;
}) {
  const { selection, races, drivers, setRaceId, setLap, setDriverId } = state;

  const sortedRaces = useMemo(
    () => [...(races.data ?? [])].sort((a: Race, b: Race) => b.round - a.round),
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
        gridTemplateColumns: showLap ? "2fr 1.3fr 0.7fr auto" : "2fr 1.3fr",
        gap: 14,
        alignItems: "end",
        background: C.surface,
        border: `1px solid ${C.edge}`,
        borderRadius: 8,
        padding: "14px 18px",
        marginBottom: 20,
      }}
    >
      <div>
        <label style={labelStyle}>{selection.season ?? ""} race weekend</label>
        <select value={selection.raceId} onChange={(e) => setRaceId(e.target.value)} style={controlStyle} disabled={races.loading}>
          {races.loading && <option value="">Loading…</option>}
          {sortedRaces.map((r) => (
            <option key={r.race_id} value={r.race_id}>
              R{String(r.round).padStart(2, "0")} — {r.name}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label style={labelStyle}>Driver</label>
        <select value={selection.driverId} onChange={(e) => setDriverId(e.target.value)} style={controlStyle} disabled={drivers.loading}>
          {drivers.loading && <option value="">Loading…</option>}
          {sortedDrivers.map((d) => (
            <option key={d.driver_id} value={d.driver_id}>
              {d.given_name} {d.family_name}
            </option>
          ))}
        </select>
      </div>

      {showLap && (
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
      )}

      {showLap && onRun && (
        <button
          onClick={onRun}
          disabled={running}
          style={{
            background: running ? C.raised : C.carbon,
            color: running ? C.faint : "#fff",
            border: "none",
            borderRadius: 6,
            padding: "11px 20px",
            fontFamily: F.mono,
            fontWeight: 700,
            fontSize: 12,
            letterSpacing: "0.08em",
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
