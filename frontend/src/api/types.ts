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

export type Team = { team_id: string; name: string; season: number };

export type EstimatePoint = { race_id: string; value: number | null; ci95: [number, number] | null; n: number };

export type CarCharacteristic = {
  key: string;
  label: string;
  unit: string;
  kind: "mean" | "correlation";
  better: "lower" | "higher" | null;
  description: string;
  value: number | null;
  ci95: [number, number] | null;
  n: number;
  field_mean: number | null;
  field_min: number | null;
  field_max: number | null;
  rank: number | null;
  teams_ranked: number;
  trajectory: EstimatePoint[];
};

export type TyreWindowPoint = {
  race_id: string;
  race_name: string;
  track_temp: number | null;
  circuit_baseline_track_temp: number | null;
  degradation_vs_field: number | null;
};

/** GET /teams/{id}/car-profile — every completed race of the season included. */
export type CarProfile = {
  team_id: string;
  season: number;
  races_observed: number;
  last_race_id: string;
  min_races_for_confidence: number;
  characteristics: CarCharacteristic[];
  tyre_window: TyreWindowPoint[];
  seeded_priors: Record<string, unknown> | null;
  seeded_priors_note: string;
};

export type CurvePoint = { tyre_age: number; median_delta_s: number; p25_delta_s: number; p75_delta_s: number; laps: number };

export type CompoundReport = {
  compound: "SOFT" | "MEDIUM" | "HARD";
  stints: number;
  wear_s_per_lap: number | null;
  wear_ci95: [number, number] | null;
  field_wear_s_per_lap: number | null;
  curve: CurvePoint[];
  field_curve: CurvePoint[];
  by_race: { race_id: string; track_temp: number; wear_s_per_lap: number; laps: number }[];
};

export type RemainingLifeLap = {
  driver_id: string;
  lap_number: number;
  stint_number: number;
  compound: string;
  tyre_age: number;
  predicted_remaining: number;
  actual_remaining: number;
};

/** GET /teams/{id}/tyres — fuel-corrected wear, see car_profiles/degradation_curves.py. */
export type TyreReport = {
  team_id: string;
  season: number;
  fuel_track_seconds_per_lap: number;
  compounds: CompoundReport[];
  remaining_life: { race_id: string; laps: RemainingLifeLap[] } | null;
};

export type DriverRaceResult = {
  race_id: string;
  race_name: string | null;
  circuit_id: string | null;
  team_id: string | null;
  grid: number | null;
  position: number | null;
  classified: boolean;
  status: string | null;
  points: number | null;
  teammate_id: string | null;
  teammate_position: number | null;
  teammate_gap_s: number | null;
  teammate_gap_laps: number | null;
};

export type DriverProfile = {
  driver_id: string;
  code: string | null;
  given_name: string | null;
  family_name: string | null;
  nationality: string | null;
  season: number;
  seasons: number[];
  summary: {
    races: number;
    points: number | null;
    wins: number;
    podiums: number;
    not_classified: number;
    avg_grid: number | null;
    avg_finish: number | null;
    ahead_of_teammate: number;
    head_to_head_races: number;
    median_teammate_gap_s: number | null;
  };
  races: DriverRaceResult[];
  career: {
    season: number;
    era: string;
    teams: string[];
    races: number;
    points: number | null;
    wins: number;
    podiums: number;
    avg_finish: number | null;
    not_classified: number;
    median_teammate_gap_s: number | null;
  }[];
  circuits: {
    circuit_id: string;
    circuit_name: string | null;
    races: number;
    best_finish: number | null;
    avg_finish: number | null;
    wins: number;
    last_race_id: string;
    last_position: number | null;
    last_status: string | null;
    median_teammate_gap_s: number | null;
  }[];
  wet_dry: {
    era: string;
    dry_laps: number;
    wet_laps: number;
    wet_races: number;
    dry_teammate_gap_s: number | null;
    wet_teammate_gap_s: number | null;
    dry_field_delta_s: number | null;
    wet_field_delta_s: number | null;
  }[];
  min_wet_laps: number;
};

export type MetricComparison = { current: number; baseline: number | null; status: "ok" | "WARN" | "no baseline" };

export type ModelHealth = {
  experiment: string;
  label: string;
  kind: string;
  evaluated_at: string;
  status: "ok" | "warn" | "skipped";
  n_rows: number | null;
  n_races: number | null;
  per_race_metric: string;
  per_race_lower_is_better: boolean;
  per_race_baseline: number | null;
  metrics: Record<string, MetricComparison>;
  per_race: { race_id: string; n_rows: number; value: number | null }[];
  feature_drift: { feature: string; psi: number; null_p95: number; drifted: boolean }[];
  notes: string[];
};

export type LapTimeOverlay = {
  race_id: string;
  driver_id: string;
  laps: { lap_number: number; predicted: number; actual: number; stable: boolean }[];
  mae_stable: number | null;
};

export type ProfileConfidencePoint = {
  races: number;
  median_relative_halfwidth: number;
  p25_relative_halfwidth: number;
  p75_relative_halfwidth: number;
  distinct_share: number;
  pairs: number;
};

export type WaitWindow = {
  open_lap: number;
  pull_the_plug_lap: number;
  latest_safe_lap: number;
  caution_saving_seconds: number;
  per_lap_caution_probability: number;
  has_window: boolean;
  reason: string;
};

export type PlannedStop = {
  stop_number: number;
  compound: string;
  window_open: number;
  window_close: number;
  nominal_lap: number;
  wait: WaitWindow | null;
};

/** GET /races/{id}/plan — the race weekend planner (race_plan/). */
export type RacePlan = {
  race_id: string;
  race_name: string;
  circuit_id: string;
  circuit_name: string | null;
  date: string;
  driver_id: string;
  team_id: string;
  grid_position: number;
  actual_grid_position: number | null;
  total_laps: number;
  n_simulations: number;
  conditions: {
    track_temp: number;
    air_temp: number;
    rain_expected: boolean;
    circuit_baseline_track_temp: number | null;
    historical_sc_rate: number | null;
    historical_overtaking_rate: number | null;
    pit_loss_seconds: number;
    cold_start_circuit: boolean;
  };
  starting_compound: string;
  compound_sequence: string[];
  compound_choice_is_decisive: boolean;
  stops: PlannedStop[];
  expected_finish: number;
  win_probability: number;
  points_probability: number;
  expected_points: number;
  starting_options: { starting_compound: string; plan: string; strategy_score: number; expected_finish: number; expected_points: number }[];
  tyres: {
    allocation: Record<string, number>;
    race_reserved: Record<string, number>;
    quali_reserved: Record<string, number>;
    practice_budget: Record<string, number>;
    used_in_practice: { compound: string; laps_run: number; first_session: string; sessions: string[] }[];
    warnings: string[];
  };
};

export type StandingEntry = { id: string; name: string; team_id: string | null; points: number; wins: number };
export type Standings = { season: number; through_race_id: string; drivers: StandingEntry[]; constructors: StandingEntry[] };

export type TeamSpeed = {
  team_id: string;
  name: string;
  races: number;
  top_speed_delta_kph: number;
  top_speed_ci95: [number, number] | null;
  median_trap_kph: number;
  pace_delta_s: number;
  pace_ci95: [number, number] | null;
  reading: string;
  by_race: { race_id: string; circuit_name: string; trap_kph: number; trap_delta_kph: number; pace_delta_s: number }[];
};

/** GET /teams/speed — straight-line speed against overall pace, 2023 onwards. */
export type SpeedProfile = { season: number; races: number; teams: TeamSpeed[] };
