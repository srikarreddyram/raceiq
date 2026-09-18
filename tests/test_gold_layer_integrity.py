"""Data-integrity regression tests for the Gold layer, one per real bug
this session found and fixed at the SQL source:

- Every race must resolve to a real circuit_id (regression test for the
  Monaco -> monza / Yas Marina -> marina_bay silent fuzzy-match
  mislabelings found via a manual circuit-coverage audit).
- No lap time should be large enough to be a red-flag session-clock
  artifact rather than real pace (regression test for the ~2,500-second
  "lap times" a 2024 red flag left in an early version of this table).
- historical_sc_rate / historical_dnf_rate must be valid probabilities
  wherever they're populated at all.
- No duplicate (race_id, driver_id, lap_number) rows — the grain
  gold.race_features promises everything downstream.
"""

from __future__ import annotations

import pytest

from models.common.data import load_race_features
from models.common.db import get_connection

# A red-flag-stopped session clock produces lap "times" in the thousands
# of seconds; a real F1 lap, even at the slowest permanent circuit under
# a full-course caution, is well under this.
IMPLAUSIBLE_LAP_TIME_SECONDS = 600.0


@pytest.fixture(scope="module")
def df():
    return load_race_features()


def test_no_duplicate_driver_laps(df):
    duplicated = df.duplicated(subset=["race_id", "driver_id", "lap_number"])
    assert not duplicated.any(), (
        f"{duplicated.sum()} duplicate (race_id, driver_id, lap_number) rows in gold.race_features"
    )


def test_every_race_has_a_resolved_circuit(df):
    missing = df[df["circuit_id"].isna()]["race_id"].unique()
    assert len(missing) == 0, f"races with no resolved circuit_id: {sorted(missing)}"


def test_implausible_lap_times_are_all_flagged_red_flag(df):
    """A red-flag-stopped session clock genuinely does produce lap "times"
    in the thousands of seconds (e.g. 2024 Monaco, lap 1) — this project
    deliberately keeps that raw value honest rather than scrubbing it, and
    instead excludes it from aggregates/targets via `red_flag_active`
    (see pipelines/gold/lap_features.py and models/lap_time/train.py's
    `next_red_flag_active` exclusion). The invariant that actually matters
    is that every implausible value is one of *those*, not a new, unflagged
    corruption slipping in some other way.
    """
    bad = df[df["lap_time_seconds"] > IMPLAUSIBLE_LAP_TIME_SECONDS]
    unflagged = bad[~bad["red_flag_active"]]
    assert unflagged.empty, (
        f"{len(unflagged)} rows with lap_time_seconds > {IMPLAUSIBLE_LAP_TIME_SECONDS}s "
        f"NOT flagged red_flag_active (race_ids {sorted(unflagged['race_id'].unique())[:5]}) — "
        "an unexplained corrupted lap time, not the known red-flag session-clock artifact"
    )


def test_historical_rates_are_valid_probabilities(df):
    for column in ["historical_sc_rate", "historical_dnf_rate"]:
        values = df[column].dropna()
        assert len(values) > 0, f"{column} is entirely null"
        assert values.between(0.0, 1.0).all(), f"{column} has values outside [0, 1]"


def test_circuit_mapping_has_no_known_bad_mismatches():
    """Direct regression test for the exact two silent mismatches found
    via manual audit: Monaco's races must resolve to circuit_id 'monaco'
    (not 'monza'), and Yas Island/Yas Marina races must resolve to
    'yas_marina' (not 'marina_bay', which is Singapore's own circuit_id).
    """
    con = get_connection()
    try:
        races = con.execute("SELECT race_id, name, circuit_id FROM silver.races").df()
    finally:
        con.close()

    monaco_races = races[races["name"].str.contains("Monaco", case=False, na=False)]
    assert (monaco_races["circuit_id"] == "monaco").all(), (
        f"Monaco race(s) resolved to the wrong circuit_id: "
        f"{monaco_races[monaco_races['circuit_id'] != 'monaco'][['race_id', 'circuit_id']].to_dict('records')}"
    )

    yas_races = races[races["name"].str.contains("Abu Dhabi", case=False, na=False)]
    if not yas_races.empty:
        assert (yas_races["circuit_id"] == "yas_marina").all(), (
            f"Abu Dhabi race(s) resolved to the wrong circuit_id: "
            f"{yas_races[yas_races['circuit_id'] != 'yas_marina'][['race_id', 'circuit_id']].to_dict('records')}"
        )
