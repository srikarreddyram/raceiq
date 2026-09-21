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
