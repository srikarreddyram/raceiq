"""Pydantic request/response models — PRD Section 14."""

from __future__ import annotations

from datetime import date as date_type

from pydantic import BaseModel


class Circuit(BaseModel):
    circuit_id: str
    name: str
    country: str
    locality: str
    latitude: float
    longitude: float


class CircuitProfile(Circuit):
    prior_races_at_circuit: int | None = None
    historical_sc_rate: float | None = None
    circuit_baseline_track_temp: float | None = None


class CornerNode(BaseModel):
    number: str
    x: float
    y: float
    speed_class: str  # slow / medium / fast, PRD 7.1 thresholds
    min_speed_kph: float
    radius_m: float | None = None


class CircuitMap(BaseModel):
    """PRD Section 7.2's dashboard map plus Section 10.3's geometry
    features. Coordinates are in a square viewBox of side `viewbox`."""

    circuit_id: str
    reference_race_id: str
    # Which regulation era the lap came from — speeds and aero behaviour
    # differ between them, and DRS doesn't exist from 2026.
    reference_season: int
    viewbox: float
    svg_path: str
    start_finish: list[float]
    sector_boundaries: list[list[float]]
    corners: list[CornerNode]
    # null when the reference race's regulations have no DRS (2026+) or when
    # DRS wasn't observed in that race — never [] meaning "has none".
    drs_zones: list[str] | None
    lap_length_m: float
    corner_count: int
    slow_corner_count: int
    medium_corner_count: int
    fast_corner_count: int
    slow_corner_pct: float | None
    avg_corner_radius_m: float | None
    total_braking_distance_m: float
    elevation_range_m: float
    elevation_variance: float
    sector1_avg_speed: float | None
    sector2_avg_speed: float | None
    sector3_avg_speed: float | None
    downforce_demand_index: float
    drs_zone_total_length_m: float | None
    pit_lane_delta: float
    tyre_stress_index: float | None
    track_evolution_rate: float | None
    detection_recall: float | None
    detected_corner_count: int


class Driver(BaseModel):
    driver_id: str
    driver_code: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    nationality: str | None = None
    season: int
    constructor_id: str | None = None


class Race(BaseModel):
    race_id: str
    season: int
    round: int
    circuit_id: str
    name: str
    date: date_type


class LapTimePredictionRequest(BaseModel):
    race_id: str
    lap_number: int
    driver_id: str


class LapTimePredictionResponse(BaseModel):
    driver_id: str
    lap_number: int
    predicted_next_lap_time_seconds: float
    actual_current_lap_time_seconds: float


class StrategyRequest(BaseModel):
    race_id: str
    lap_number: int
    driver_id: str
    n_simulations: int = 5000


class ScoredStrategy(BaseModel):
    action: str
    # `action` is display text; `pit_plan` is the same strategy as data, so
    # a client can re-simulate or chart it without parsing that sentence.
    pit_plan: list[tuple[int, str]]
    win_probability: float
    podium_probability: float
    expected_finish: float
    expected_points: float
    risk_score: float
    strategy_score: float


class RecommendedStrategy(ScoredStrategy):
    """The top-ranked strategy carries more than the alternatives do: the
    simulation already computes its full outcome distribution and
    safety-car encounter rate, and a dashboard charting the recommendation
    needs both. They were being dropped at this boundary purely because
    the schema didn't name them.
    """

    points_probability: float
    safety_car_encounter_rate: float
    finish_distribution: dict[int, float]


class StrategyRecommendation(BaseModel):
    circuit_id: str
    driver_id: str
    team_id: str
    current_lap: int
    laps_remaining: int
    n_simulations: int
    n_candidates_evaluated: int
    # The Win Probability and Final Race Position models' own estimates
    # for the driver's actual current state, independent of the simulation
    # below — cross-checks, not replacements (see
    # strategy_engine/oracles.py's docstrings).
    win_probability_model_estimate: float
    expected_finish_model_estimate: float
    recommended_strategy: RecommendedStrategy
    reasoning: list[str]
    alternatives: list[ScoredStrategy]


class SimulateRequest(BaseModel):
    race_id: str
    lap_number: int
    driver_id: str
    pit_plan: list[tuple[int, str]]  # [(pit_lap, compound), ...]
    n_simulations: int = 5000


class SimulateResponse(BaseModel):
    action: str
    win_probability: float
    podium_probability: float
    points_probability: float
    expected_finish: float
    expected_points: float
    risk_score: float
    finish_distribution: dict[int, float]


class Team(BaseModel):
    team_id: str
    name: str
    season: int


class EstimatePoint(BaseModel):
    race_id: str
    value: float | None
    ci95: list[float] | None
    n: int


class CarCharacteristic(BaseModel):
    key: str
    label: str
    unit: str
    kind: str  # "mean" or "correlation"
    better: str | None  # "lower" / "higher" / None when neither end is better
    description: str
    value: float | None
    ci95: list[float] | None
    n: int
    field_mean: float | None
    field_min: float | None
    field_max: float | None
    rank: int | None
    teams_ranked: int
    trajectory: list[EstimatePoint]


class TyreWindowPoint(BaseModel):
    race_id: str
    race_name: str
    track_temp: float | None
    circuit_baseline_track_temp: float | None
    degradation_vs_field: float | None


class CarProfileResponse(BaseModel):
    """PRD Section 13.2's Car Profile View — see car_profiles/season_profile.py.
    Every completed race of the season is included, unlike
    gold.car_profiles' pre-race rows that the models use."""

    team_id: str
    season: int
    races_observed: int
    last_race_id: str
    min_races_for_confidence: int
    characteristics: list[CarCharacteristic]
    tyre_window: list[TyreWindowPoint]
    # Always null today: PRD 8.1's Layer 2 priors aren't ingested.
    seeded_priors: dict | None
    seeded_priors_note: str


class CurvePoint(BaseModel):
    tyre_age: int
    median_delta_s: float
    p25_delta_s: float
    p75_delta_s: float
    laps: int


class RaceWear(BaseModel):
    race_id: str
    track_temp: float
    wear_s_per_lap: float
    laps: int


class CompoundReport(BaseModel):
    compound: str
    stints: int
    wear_s_per_lap: float | None
    wear_ci95: list[float] | None
    field_wear_s_per_lap: float | None
    curve: list[CurvePoint]
    field_curve: list[CurvePoint]
    by_race: list[RaceWear]


class RemainingLifeLap(BaseModel):
    driver_id: str
    lap_number: int
    stint_number: int
    compound: str
    tyre_age: int
    predicted_remaining: float
    actual_remaining: int


class RemainingLifeTrace(BaseModel):
    race_id: str
    laps: list[RemainingLifeLap]


class TyreReport(BaseModel):
    """PRD Section 13.2's Tyre View — fuel-corrected, see
    car_profiles/degradation_curves.py — plus the Tyre Degradation model's
    remaining-life predictions over the team's latest race."""

    team_id: str
    season: int
    fuel_track_seconds_per_lap: float
    compounds: list[CompoundReport]
    remaining_life: RemainingLifeTrace | None


class DriverRaceResult(BaseModel):
    race_id: str
    race_name: str | None
    circuit_id: str | None
    team_id: str | None
    grid: int | None  # null for a pit-lane start (Ergast grid 0)
    position: int | None  # finishing ORDER, retirements included
    classified: bool
    status: str | None
    points: float | None
    teammate_id: str | None
    teammate_position: int | None
    # Median lap-time gap to the teammate on shared clean green laps; < 0 = faster.
    teammate_gap_s: float | None
    teammate_gap_laps: int | None


class DriverSeasonSummary(BaseModel):
    races: int
    points: float | None
    wins: int
    podiums: int
    not_classified: int
    avg_grid: float | None
    avg_finish: float | None  # classified finishes only
    ahead_of_teammate: int
    head_to_head_races: int
    median_teammate_gap_s: float | None


class DriverCareerSeason(BaseModel):
    season: int
    era: str
    teams: list[str]
    races: int
    points: float | None
    wins: int
    podiums: int
    avg_finish: float | None
    not_classified: int
    median_teammate_gap_s: float | None


class DriverCircuitRecord(BaseModel):
    circuit_id: str
    circuit_name: str | None
    races: int
    best_finish: int | None
    avg_finish: float | None
    wins: int
    last_race_id: str
    last_position: int | None
    last_status: str | None
    median_teammate_gap_s: float | None


class WetDrySplit(BaseModel):
    era: str
    dry_laps: int
    wet_laps: int
    wet_races: int
    dry_teammate_gap_s: float | None
    wet_teammate_gap_s: float | None  # null below min_wet_laps
    dry_field_delta_s: float | None
    wet_field_delta_s: float | None


class DriverProfile(BaseModel):
    """PRD Section 13.2's Driver View — see driver_profiles/profile.py."""

    driver_id: str
    code: str | None
    given_name: str | None
    family_name: str | None
    nationality: str | None
    season: int
    seasons: list[int]
    summary: DriverSeasonSummary
    races: list[DriverRaceResult]
    career: list[DriverCareerSeason]
    circuits: list[DriverCircuitRecord]
    wet_dry: list[WetDrySplit]
    min_wet_laps: int


class MetricComparison(BaseModel):
    current: float
    baseline: float | None  # the promoted run's test-season figure
    status: str  # "ok", "WARN" or "no baseline"


class RaceMetric(BaseModel):
    race_id: str
    n_rows: int
    value: float | None  # null where undefined, e.g. an AUC in a race with no safety car


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    null_p95: float  # PSI of random same-size sets of training races — the noise floor
    drifted: bool


class ModelHealth(BaseModel):
    """One model's monitoring row — written by monitoring/run.py."""

    experiment: str
    label: str
    kind: str
    evaluated_at: str
    status: str  # "ok", "warn" or "skipped"
    n_rows: int | None
    n_races: int | None
    per_race_metric: str
    per_race_lower_is_better: bool
    per_race_baseline: float | None
    metrics: dict[str, MetricComparison]
    per_race: list[RaceMetric]
    feature_drift: list[FeatureDrift]
    notes: list[str]


class LapTimeOverlayPoint(BaseModel):
    lap_number: int
    predicted: float
    actual: float
    stable: bool


class LapTimeOverlay(BaseModel):
    race_id: str
    driver_id: str
    laps: list[LapTimeOverlayPoint]
    mae_stable: float | None


class ProfileConfidencePoint(BaseModel):
    races: int
    median_relative_halfwidth: float
    p25_relative_halfwidth: float
    p75_relative_halfwidth: float
    distinct_share: float
    pairs: int


class WaitWindowOut(BaseModel):
    open_lap: int
    pull_the_plug_lap: int
    latest_safe_lap: int
    caution_saving_seconds: float
    per_lap_caution_probability: float
    has_window: bool
    reason: str


class PlannedStopOut(BaseModel):
    stop_number: int
    compound: str
    window_open: int
    window_close: int
    nominal_lap: int
    wait: WaitWindowOut | None


class StartingOption(BaseModel):
    starting_compound: str
    plan: str
    strategy_score: float
    expected_finish: float
    expected_points: float


class TyreSetOut(BaseModel):
    compound: str
    laps_run: int
    first_session: str
    sessions: list[str]


class TyrePlanOut(BaseModel):
    allocation: dict[str, int]
    race_reserved: dict[str, int]
    quali_reserved: dict[str, int]
    practice_budget: dict[str, int]
    used_in_practice: list[TyreSetOut]
    warnings: list[str]


class RaceConditions(BaseModel):
    track_temp: float
    air_temp: float
    rain_expected: bool
    circuit_baseline_track_temp: float | None
    historical_sc_rate: float | None
    historical_overtaking_rate: float | None
    pit_loss_seconds: float
    cold_start_circuit: bool  # no prior races here: circuit priors are unknown
    source: str  # "measured" (a run race), "forecast", "typical" or "override"
    note: str
    rain_probability: float | None


class RacePlanResponse(BaseModel):
    """The race weekend planner — race_plan/. Everything a strategist needs
    for one driver at one race: conditions, tyre sets, starting compound,
    pit windows and the hold-for-a-caution window per stop."""

    race_id: str
    race_name: str
    circuit_id: str
    circuit_name: str | None
    date: date_type
    driver_id: str
    team_id: str
    grid_position: int
    actual_grid_position: int | None
    grid_is_expected: bool  # an unrun race's slot from season-average grid, not chosen
    total_laps: int
    laps_known: bool  # False only for a first race at a circuit
    is_future: bool
    prior_races_at_circuit: int | None
    n_simulations: int
    conditions: RaceConditions
    starting_compound: str
    compound_sequence: list[str]
    compound_choice_is_decisive: bool
    stops: list[PlannedStopOut]
    expected_finish: float
    win_probability: float
    points_probability: float
    expected_points: float
    starting_options: list[StartingOption]
    tyres: TyrePlanOut


class StandingEntry(BaseModel):
    id: str
    name: str
    team_id: str | None
    points: float
    wins: int


class Standings(BaseModel):
    season: int
    through_race_id: str
    drivers: list[StandingEntry]
    constructors: list[StandingEntry]


class SpeedRace(BaseModel):
    race_id: str
    circuit_name: str
    trap_kph: float
    trap_delta_kph: float
    pace_delta_s: float


class TeamSpeed(BaseModel):
    team_id: str
    name: str
    races: int
    top_speed_delta_kph: float
    top_speed_ci95: list[float] | None
    median_trap_kph: float
    pace_delta_s: float
    pace_ci95: list[float] | None
    reading: str
    by_race: list[SpeedRace]


class SpeedProfile(BaseModel):
    """Straight-line speed against overall pace — car_profiles/speed_profile.py."""

    season: int
    races: int
    teams: list[TeamSpeed]


class CalendarRound(BaseModel):
    race_id: str
    season: int
    round: int
    name: str
    circuit_id: str
    circuit_name: str
    country: str
    date: date_type
    has_results: bool
