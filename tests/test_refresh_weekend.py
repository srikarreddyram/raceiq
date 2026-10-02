"""Which rounds a weekend refresh picks (ingestion/refresh_weekend.py)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from ingestion.config import load_config
from ingestion.refresh_weekend import RoundReport, _has_races, rounds_to_refresh

RACES = [
    {"round": "1", "date": "2026-03-08"},
    {"round": "2", "date": "2026-03-15"},
    {"round": "3", "date": "2026-03-29"},
    {"round": "4", "date": "2026-04-12"},
]


def _ingested(root, season, rnd):
    (root / "ergast" / str(season) / str(rnd)).mkdir(parents=True)
    (root / "ergast" / str(season) / str(rnd) / "results.json").write_text("{}")
    (root / "fastf1" / str(season) / str(rnd) / "R").mkdir(parents=True)
    (root / "fastf1" / str(season) / str(rnd) / "R" / "results.parquet").write_text("")


def test_catches_up_run_rounds_and_takes_a_weekend_underway(tmp_path):
    config = replace(load_config(), raw_data_root=tmp_path)
    _ingested(tmp_path, 2026, 1)
    # Friday of round 3's weekend: round 2 was never ingested, round 3 is
    # two days away, round 4 is two weeks off.
    chosen = rounds_to_refresh(RACES, 2026, date(2026, 3, 27), config)
    assert [r["round"] for r in chosen] == ["2", "3"]


def test_nothing_to_do_between_weekends(tmp_path):
    config = replace(load_config(), raw_data_root=tmp_path)
    for rnd in (1, 2):
        _ingested(tmp_path, 2026, rnd)
    assert rounds_to_refresh(RACES, 2026, date(2026, 3, 20), config) == []


def test_a_round_with_results_but_no_race_laps_is_still_refreshed(tmp_path):
    config = replace(load_config(), raw_data_root=tmp_path)
    (tmp_path / "ergast" / "2026" / "1").mkdir(parents=True)
    (tmp_path / "ergast" / "2026" / "1" / "results.json").write_text("{}")
    chosen = rounds_to_refresh(RACES[:1], 2026, date(2026, 3, 10), config)
    assert [r["round"] for r in chosen] == ["1"]


def test_an_empty_response_is_not_content():
    assert not _has_races({"MRData": {"RaceTable": {"Races": []}}})
    assert _has_races({"MRData": {"RaceTable": {"Races": [{"round": "1"}]}}})


def test_report_says_what_is_missing():
    report = RoundReport(2026, 16, "Singapore Grand Prix", "2026-10-11", fetched=["FP1"], not_available=["Q", "R"])
    assert "FP1" in report.line() and "not available yet: Q, R" in report.line()
