"""Race-state representation — PRD Section 12.2's inputs, built from
whatever a real `gold.race_features` row (plus a rival snapshot) gives us.

Two PRD-named inputs aren't modeled and default to a fixed assumption
rather than being faked from data that doesn't exist:

- `available_compounds` — which tyre sets remain in a team's allocation is
  team-internal logistics data, not published by any source this project
  ingests. Defaults to all three dry compounds always being available.
- `weather_state` beyond current conditions — there's no forecast source
  (see lap_features.py's docstring on `rain_probability_next_10_laps`),
  so weather is treated as constant for the duration of one simulated
  race rather than sampled as changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass
class RivalState:
    driver_id: str
    team_id: str
    compound: str
    tyre_age: float
    gap_to_car_ahead: float | None  # this rival's own gap to the car ahead of them


@dataclass
class RaceState:
    circuit_id: str
    team_id: str
    driver_id: str
    current_lap: int
    race_total_laps: int
    compound: str
    tyre_age: float
    stint_number: int
    current_position: float
    gap_to_leader: float
    gap_to_car_ahead: float | None
    gap_to_car_behind: float | None
    lap_time_seconds: float
    field_avg_lap_time_seconds: float
    degradation_rate: float | None
    grip_estimate: float | None

    # Weather — held constant for the simulation (see module docstring).
    air_temp: float
    track_temp: float
    humidity: float
    wind_speed: float
    wind_direction: float
    rainfall_flag: bool

    # Historical driver/circuit context — also held constant; these are
    # season-to-date facts, not something that changes lap to lap.
    driver_avg_pace_delta: float | None
    driver_consistency_score: float | None
    driver_overtaking_score: float | None
    condition_delta: float | None
    historical_sc_rate: float | None
    historical_dnf_rate: float | None
    circuit_baseline_track_temp: float | None

    # The car directly ahead — the only rival our Gold features actually
    # describe (rival_driver_id/team_id/compound/tyre_age all mean "car
    # ahead" per pipelines/gold/lap_features.py). A fuller rival_states
    # list (PRD 12.2) would need per-rival strategy simulation, which is
    # out of scope for this version — see simulation/monte_carlo.py.
    rival_ahead: RivalState | None = None

    available_compounds: tuple[str, ...] = DRY_COMPOUNDS
    compounds_used_this_race: set[str] = field(default_factory=set)

    @property
    def laps_remaining(self) -> int:
        return self.race_total_laps - self.current_lap

    @classmethod
    def from_gold_row(cls, row: pd.Series, race_total_laps: int) -> "RaceState":
        """Build a RaceState from one row of `models.common.data.load_race_features()`
        — i.e. a real historical (driver, lap) snapshot. Used both for testing
        the engine against real races and as the template for a live API request.
        """
        rival = None
        if pd.notna(row.get("rival_driver_id")):
            rival = RivalState(
                driver_id=row["rival_driver_id"],
                team_id=row["rival_team_id"],
                compound=row["rival_compound"],
                tyre_age=row["rival_tyre_age"],
                gap_to_car_ahead=None,
            )

        return cls(
            circuit_id=row["circuit_id"],
            team_id=row["team_id"],
            driver_id=row["driver_id"],
            current_lap=int(row["lap_number"]),
            race_total_laps=race_total_laps,
            compound=row["compound"],
            tyre_age=float(row["tyre_age"]),
            stint_number=int(row["stint_number"]),
            current_position=float(row["current_position"]),
            gap_to_leader=float(row["gap_to_leader"]),
            gap_to_car_ahead=row.get("gap_to_car_ahead"),
            gap_to_car_behind=row.get("gap_to_car_behind"),
            lap_time_seconds=float(row["lap_time_seconds"]),
            field_avg_lap_time_seconds=float(row["field_avg_lap_time_seconds"]),
            degradation_rate=row.get("degradation_rate"),
            grip_estimate=row.get("grip_estimate"),
            air_temp=row["air_temp"],
            track_temp=row["track_temp"],
            humidity=row["humidity"],
            wind_speed=row["wind_speed"],
            wind_direction=row["wind_direction"],
            rainfall_flag=bool(row["rainfall_flag"]),
            driver_avg_pace_delta=row.get("driver_avg_pace_delta"),
            driver_consistency_score=row.get("driver_consistency_score"),
            driver_overtaking_score=row.get("driver_overtaking_score"),
            condition_delta=row.get("condition_delta"),
            historical_sc_rate=row.get("historical_sc_rate"),
            historical_dnf_rate=row.get("historical_dnf_rate"),
            circuit_baseline_track_temp=row.get("circuit_baseline_track_temp"),
            rival_ahead=rival,
            compounds_used_this_race={row["compound"]},
        )
