"""Shared fixtures for RaceIQ's test suite.

These tests are integration-style, on purpose, matching how this project
has been built and verified all along: against the real local DuckDB
warehouse (`pipelines/bronze/db.py`'s file), not synthetic or mocked data.
That means they need the Bronze/Silver/Gold pipelines already built
locally (`uv run python -m pipelines.<layer>.run`) and at least the six
models trained (`uv run python -m models.<name>.train`) — there is no
CI-friendly fixture data checked into the repo, since the whole point of
this project's testing discipline has been catching bugs that only show
up against real F1 data's actual messiness (DSQs, red flags, mid-season
debuts, rookie drivers). Running this suite with none of that built will
fail at collection time with a clear "no data" error, not a silent skip.
"""

from __future__ import annotations

import pytest

from models.common.data import load_race_features

# A real, manually-verified historical race snapshot used across this
# session's whole strategy-engine debugging history: Bahrain 2025, lap 20.
# piastri is the actual race leader at this point (current_position=1,
# gap_to_leader=0), hamilton is mid-pack, bortoleto is genuinely last
# (current_position=20) — a spread deliberately covering the front, middle,
# and back of the field with real, not synthetic, data.
SCENARIO_RACE_ID = "2025_4"
SCENARIO_LAP = 20
SCENARIO_LEADER = "piastri"
SCENARIO_MIDPACK = "hamilton"
SCENARIO_LAST = "bortoleto"

# A second real scenario used to stress-test the LSTM pace oracle and the
# Monte Carlo simulation's pace-deviation cap: a wet, drying-track race.
WET_RACE_ID = "2025_10"
WET_LAP = 30
WET_DRIVER = "piastri"


@pytest.fixture(scope="session")
def race_features():
    return load_race_features()


def _build_state(race_features, race_id: str, lap: int, driver_id: str):
    from strategy_engine.state import RaceState

    race_total_laps = int(race_features[race_features.race_id == race_id]["lap_number"].max())
    row = race_features[
        (race_features.race_id == race_id)
        & (race_features.lap_number == lap)
        & (race_features.driver_id == driver_id)
    ]
    if row.empty:
        pytest.fail(
            f"No gold.race_features row for race_id={race_id!r} lap={lap} driver_id={driver_id!r} — "
            "has the Gold layer been built? (uv run python -m pipelines.gold.run)"
        )
    return RaceState.from_gold_row(row.iloc[0], race_total_laps=race_total_laps)


@pytest.fixture
def leader_state(race_features):
    return _build_state(race_features, SCENARIO_RACE_ID, SCENARIO_LAP, SCENARIO_LEADER)


@pytest.fixture
def midpack_state(race_features):
    return _build_state(race_features, SCENARIO_RACE_ID, SCENARIO_LAP, SCENARIO_MIDPACK)


@pytest.fixture
def last_place_state(race_features):
    return _build_state(race_features, SCENARIO_RACE_ID, SCENARIO_LAP, SCENARIO_LAST)


@pytest.fixture
def wet_race_state(race_features):
    return _build_state(race_features, WET_RACE_ID, WET_LAP, WET_DRIVER)


@pytest.fixture
def leader_rivals():
    from strategy_engine.field import build_field_snapshot_from_gold

    return build_field_snapshot_from_gold(SCENARIO_RACE_ID, SCENARIO_LAP, exclude_driver_id=SCENARIO_LEADER)


@pytest.fixture
def last_place_rivals():
    from strategy_engine.field import build_field_snapshot_from_gold

    return build_field_snapshot_from_gold(SCENARIO_RACE_ID, SCENARIO_LAP, exclude_driver_id=SCENARIO_LAST)
