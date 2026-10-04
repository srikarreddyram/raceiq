"""The wet-race plan (race_plan/wet.py)."""

from __future__ import annotations

import pytest

from race_plan.wet import WET_PLAN_LEADS, WET_PLAN_PROBABILITY, build_wet_plan, wet_race_facts


def test_wet_race_facts_say_what_teams_do():
    f = wet_race_facts()
    if f.wet_races < 5:
        pytest.skip("too few wet races in this warehouse")
    assert f.start_on_inters > f.start_on_wets  # intermediates are the usual wet start
    assert f.sc_rate[0] > f.sc_rate[1] and f.red_flag_rate[0] > f.red_flag_rate[1]
    assert f.stops[0] > f.stops[1]  # wet races take more stops
    assert f.to_slicks_median_laps is not None and 0 <= f.to_slicks_median_laps <= 20
    assert abs(sum(f.slick_after_wet.values()) - 1) < 1e-6


def test_no_wet_plan_when_rain_is_unlikely_and_it_leads_when_likely():
    assert build_wet_plan(WET_PLAN_PROBABILITY - 0.05, [{"hour_utc": "13:00", "rain_probability": 0.1, "rain_mm": 0.0}]) is None
    maybe = build_wet_plan(0.35, [{"hour_utc": "13:00", "rain_probability": 0.35, "rain_mm": 0.2}])
    assert maybe is not None and not maybe.leads
    likely = build_wet_plan(0.9, [{"hour_utc": "13:00", "rain_probability": WET_PLAN_LEADS + 0.4, "rain_mm": 1.0}])
    assert likely.leads and likely.start_tyre == "INTERMEDIATE"
    assert [c["title"].split()[0] for c in likely.calls][:2] == ["Start", "Back"]


def test_rain_later_in_the_race_still_brings_the_plan():
    # Dry at the start, a storm forecast for the second hour.
    plan = build_wet_plan(0.1, [{"hour_utc": "13:00", "rain_probability": 0.1, "rain_mm": 0.0},
                                {"hour_utc": "14:00", "rain_probability": 0.8, "rain_mm": 2.5}])
    assert plan is not None and plan.leads and plan.chance_during == pytest.approx(0.8)


def test_the_planner_carries_it_for_wet_and_not_for_dry_races():
    from race_plan.plan import build_race_plan

    wet = build_race_plan("2025_12", "hamilton", n_simulations=150)  # Silverstone 2025, started on intermediates
    assert wet.wet is not None and wet.wet.source == "measured" and wet.wet.leads
    dry = build_race_plan("2025_14", "hamilton", n_simulations=150)  # Hungary 2025
    assert dry.wet is None
    forced = build_race_plan("2025_14", "hamilton", n_simulations=150, rain=True)
    assert forced.wet is not None and forced.wet.source == "override"
