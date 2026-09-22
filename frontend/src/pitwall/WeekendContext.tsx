/**
 * The race weekend the Pit Wall is working on — one race, one driver,
 * shared by the planner, the in-race strategy call and the strategy
 * simulator, so picking "Leclerc at Madring" once carries through all
 * three.
 *
 * Deliberately the CURRENT season only. These are tools for the weekend
 * in front of you, not a browser of every race since 2018 (the Driver,
 * Car Profile and Tyre views are where history lives). It defaults to the
 * latest race with data and the championship leader — never a hard-coded
 * team, which used to paint every view Ferrari red.
 */

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { Driver, Race, Standings } from "../api/types";
import { useAsync, type AsyncState } from "../api/useAsync";

export type Weekend = {
  season: number | null;
  raceId: string;
  driverId: string;
  lap: number;
  setRaceId: (id: string) => void;
  setDriverId: (id: string) => void;
  setLap: (lap: number) => void;
  races: AsyncState<Race[]>;
  drivers: AsyncState<Driver[]>;
  standings: AsyncState<Standings>;
  race: Race | undefined;
  ready: boolean;
};

const WeekendContext = createContext<Weekend | null>(null);

export function WeekendProvider({ children }: { children: ReactNode }) {
  const standings = useAsync(() => api.standings(), []);
  const season = standings.data?.season ?? null;
  const races = useAsync(() => api.races(season ?? undefined), [season], season != null);
  const drivers = useAsync(() => api.drivers(season ?? undefined), [season], season != null);

  const [raceId, setRaceId] = useState("");
  const [driverId, setDriverId] = useState("");
  const [lap, setLap] = useState(20);

  // Defaults once the season is known: the latest race, the points leader.
  useEffect(() => {
    if (!raceId && standings.data) setRaceId(standings.data.through_race_id);
    if (!driverId && standings.data?.drivers.length) setDriverId(standings.data.drivers[0].id);
  }, [standings.data, raceId, driverId]);

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
