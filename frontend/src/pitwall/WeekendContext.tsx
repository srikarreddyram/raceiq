/**
 * The race weekend the Pit Wall is working on — one race, one driver,
 * shared by the planner, the in-race strategy call and the strategy
 * simulator, so picking "Leclerc at Madring" once carries through all
 * three.
 *
 * Deliberately the CURRENT season, its whole calendar — run and unrun.
 * These are tools for the weekend in front of you, not a browser of every
 * race since 2018 (the Car and Driver pages are where history lives). It
 * defaults to the NEXT race on the calendar (the latest run one once the
 * season is over) and the championship leader — never a hard-coded team.
 */

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { CalendarRound, Driver, Standings } from "../api/types";
import { useAsync, type AsyncState } from "../api/useAsync";

export type Weekend = {
  season: number | null;
  raceId: string;
  driverId: string;
  lap: number;
  setRaceId: (id: string) => void;
  setDriverId: (id: string) => void;
  setLap: (lap: number) => void;
  races: AsyncState<CalendarRound[]>;
  drivers: AsyncState<Driver[]>;
  standings: AsyncState<Standings>;
  race: CalendarRound | undefined;
  ready: boolean;
};

const WeekendContext = createContext<Weekend | null>(null);

export function WeekendProvider({ children }: { children: ReactNode }) {
  const standings = useAsync(() => api.standings(), []);
  const season = standings.data?.season ?? null;
  const races = useAsync(() => api.calendar(season ?? undefined), [season], season != null);
  const drivers = useAsync(() => api.drivers(season ?? undefined), [season], season != null);

  const [raceId, setRaceId] = useState("");
  const [driverId, setDriverId] = useState("");
  const [lap, setLap] = useState(20);

  // Defaults once the season is known: the next race, the points leader.
  useEffect(() => {
    if (!raceId && races.data?.length) {
      const next = races.data.find((r) => !r.has_results);
      setRaceId(next?.race_id ?? races.data[races.data.length - 1].race_id);
    }
    if (!driverId && standings.data?.drivers.length) setDriverId(standings.data.drivers[0].id);
  }, [races.data, standings.data, raceId, driverId]);

  const race = races.data?.find((r) => r.race_id === raceId);
  const value = useMemo<Weekend>(
    () => ({
      season,
      raceId,
      driverId,
      lap,
      setRaceId,
      setDriverId,
      setLap,
      races,
      drivers,
      standings,
      race,
      ready: Boolean(raceId && driverId),
    }),
    [season, raceId, driverId, lap, races, drivers, standings, race],
  );
  return <WeekendContext.Provider value={value}>{children}</WeekendContext.Provider>;
}

export function useWeekend(): Weekend {
  const ctx = useContext(WeekendContext);
  if (!ctx) throw new Error("useWeekend must be used inside WeekendProvider");
  return ctx;
}
