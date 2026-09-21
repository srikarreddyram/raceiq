/**
 * Mirrors serving/api/schemas.py exactly. Any field that's optional there
 * is optional here — the Gold layer genuinely has nulls (a circuit's
 * first-ever race has no prior-race history, a driver mid-season may have
 * no constructor yet), and pretending otherwise just moves the crash from
 * the API boundary into a component.
 */

export type Circuit = {
  circuit_id: string;
  name: string;
  country: string;
  locality: string;
  latitude: number;
  longitude: number;
};

export type CircuitProfile = Circuit & {
  prior_races_at_circuit: number | null;
  historical_sc_rate: number | null;
  circuit_baseline_track_temp: number | null;
};

export type CornerNode = {
  number: string;
  x: number;
  y: number;
  speed_class: "slow" | "medium" | "fast";
  min_speed_kph: number;
  radius_m: number | null;
};

/** GET /circuits/{id}/map — coordinates live in a square viewBox of side `viewbox`. */
export type CircuitMap = {
  circuit_id: string;
  reference_race_id: string;
  reference_season: number;
  viewbox: number;
  svg_path: string;
  start_finish: [number, number];
  sector_boundaries: [number, number][];
  corners: CornerNode[];
  // null when the reference race's regulations have no DRS (2026+) or DRS
  // wasn't observed — never [] meaning "has none".
  drs_zones: string[] | null;
  lap_length_m: number;
  corner_count: number;
  slow_corner_count: number;
  medium_corner_count: number;
  fast_corner_count: number;
  slow_corner_pct: number | null;
  avg_corner_radius_m: number | null;
  total_braking_distance_m: number;
  elevation_range_m: number;
  elevation_variance: number;
  sector1_avg_speed: number | null;
  sector2_avg_speed: number | null;
  sector3_avg_speed: number | null;
  downforce_demand_index: number;
  drs_zone_total_length_m: number | null;
  pit_lane_delta: number;
  tyre_stress_index: number | null;
  track_evolution_rate: number | null;
  detection_recall: number | null;
  detected_corner_count: number;
};

export type Driver = {
  driver_id: string;
  driver_code: string | null;
  given_name: string | null;
  family_name: string | null;
  nationality: string | null;
  season: number;
  constructor_id: string | null;
};

export type Race = {
  race_id: string;
  season: number;
  round: number;
  circuit_id: string;
  name: string;
  date: string;
};

export type LapTimePrediction = {
  driver_id: string;
  lap_number: number;
  predicted_next_lap_time_seconds: number;
  actual_current_lap_time_seconds: number;
};

export type ScoredStrategy = {
  action: string;
  /** The same strategy as data, so it can be re-simulated or charted
   *  without parsing `action`'s display text. */
  pit_plan: PitPlan;
  win_probability: number;
  podium_probability: number;
  expected_finish: number;
  expected_points: number;
  risk_score: number;
  strategy_score: number;
};

/** The top-ranked strategy carries its full outcome distribution too. */
export type RecommendedStrategy = ScoredStrategy & {
  points_probability: number;
  safety_car_encounter_rate: number;
  finish_distribution: Record<string, number>;
};

export type StrategyRecommendation = {
  circuit_id: string;
  driver_id: string;
  team_id: string;
  current_lap: number;
  laps_remaining: number;
  n_simulations: number;
  n_candidates_evaluated: number;
  /** Independent cross-checks from the Win Probability / Race Position
   *  classifiers — deliberately NOT the simulation's own numbers. */
  win_probability_model_estimate: number;
  expected_finish_model_estimate: number;
  recommended_strategy: RecommendedStrategy;
  reasoning: string[];
  alternatives: ScoredStrategy[];
};

export type SimulationOutcome = {
  action: string;
  win_probability: number;
  podium_probability: number;
  points_probability: number;
  expected_finish: number;
  expected_points: number;
  risk_score: number;
  /** Keyed by finishing position; JSON object keys arrive as strings. */
  finish_distribution: Record<string, number>;
};

export type PitPlan = [number, string][];
