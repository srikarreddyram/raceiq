/**
 * Thin fetch wrapper over the RaceIQ FastAPI layer (serving/api/).
 *
 * No react-query or similar: the whole app makes a handful of requests
 * against a local API with no auth, no pagination and no cache
 * invalidation story, so a fetch wrapper plus the `useAsync` hook next to
 * it is the entire requirement. This is the same reasoning the design
 * system template applies to animation libraries — don't take a build
 * dependency for something a few lines cover.
 *
 * FastAPI surfaces its errors as `{"detail": "..."}`, so that's unpacked
 * into a real Error message rather than showing a raw status code to
 * someone who then has to go read the server log.
 */

import type {
  CarProfile,
  Circuit,
  CircuitMap,
  CircuitProfile,
  Driver,
  DriverProfile,
  LapTimeOverlay,
  LapTimePrediction,
  ModelHealth,
  PitPlan,
  ProfileConfidencePoint,
  RacePlan,
  SpeedProfile,
  Standings,
  Race,
  SimulationOutcome,
  StrategyRecommendation,
  Team,
  TyreReport,
} from "./types";

const BASE_URL = import.meta.env.VITE_RACEIQ_API ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    // A network-level failure (API not running) is by far the most likely
    // error in local use, and it deserves a message that says so rather
    // than "Failed to fetch".
    throw new Error(`Could not reach the RaceIQ API at ${BASE_URL}`, { cause });
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // Non-JSON error body — keep the status line we already have.
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  circuits: () => request<Circuit[]>("/circuits"),
  circuitProfile: (circuitId: string) => request<CircuitProfile>(`/circuits/${circuitId}/profile`),
  circuitMap: (circuitId: string) => request<CircuitMap>(`/circuits/${circuitId}/map`),

  teams: (season?: number) => request<Team[]>(`/teams${season ? `?season=${season}` : ""}`),
  carProfile: (teamId: string, season: number) =>
    request<CarProfile>(`/teams/${teamId}/car-profile?season=${season}`),
  speed: (season: number) => request<SpeedProfile>(`/teams/speed?season=${season}`),
  tyres: (teamId: string, season: number) => request<TyreReport>(`/teams/${teamId}/tyres?season=${season}`),

  drivers: (season?: number) => request<Driver[]>(`/drivers${season ? `?season=${season}` : ""}`),
  driverProfile: (driverId: string, season: number) =>
    request<DriverProfile>(`/drivers/${driverId}/profile?season=${season}`),
  races: (season?: number) => request<Race[]>(`/races${season ? `?season=${season}` : ""}`),

  standings: (season?: number) => request<Standings>(`/standings${season ? `?season=${season}` : ""}`),
  racePlan: (raceId: string, driverId: string, grid?: number | null) =>
    request<RacePlan>(`/races/${raceId}/plan?driver_id=${driverId}${grid ? `&grid=${grid}` : ""}`),

  modelHealth: () => request<ModelHealth[]>("/monitoring/models"),
  lapTimeOverlay: (raceId: string, driverId: string) =>
    request<LapTimeOverlay>(`/monitoring/lap-time?race_id=${raceId}&driver_id=${driverId}`),
  profileConfidence: (season: number) =>
    request<ProfileConfidencePoint[]>(`/monitoring/car-profile-confidence?season=${season}`),

  predictLapTime: (body: { race_id: string; lap_number: number; driver_id: string }) =>
    post<LapTimePrediction>("/predict/laptime", body),

  optimalStrategy: (body: {
    race_id: string;
    lap_number: number;
    driver_id: string;
    n_simulations?: number;
  }) => post<StrategyRecommendation>("/predict/strategy/optimal", body),

  simulate: (body: {
    race_id: string;
    lap_number: number;
    driver_id: string;
    pit_plan: PitPlan;
    n_simulations?: number;
  }) => post<SimulationOutcome>("/simulate", body),
};

export { BASE_URL };
