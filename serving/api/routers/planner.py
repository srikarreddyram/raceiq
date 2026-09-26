"""GET /races/{race_id}/plan and GET /standings — the race weekend planner
(race_plan/), the project's centrepiece, and the championship table the
views default from.

A plan runs ~150,000 simulated races and takes several seconds, so results
are cached per (race, driver, grid slot, simulation count): the question a
strategist asks repeatedly — "what if we start P8 instead?" — is answered
once per slot.
"""

from __future__ import annotations

from functools import lru_cache

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query

from models.common.data import is_classified
from race_plan.plan import STRATEGY_EDGE_EVIDENCE, build_race_plan
from race_plan.tyre_allocation import recommend_tyre_allocation
from race_plan.weekend_pace import qualifying_gaps
from serving.api.db import get_db
from serving.api.schemas import CalendarRound, RacePlanResponse, Standings, StandingEntry
from strategy_engine.pit_loss import typical_pit_loss_seconds

router = APIRouter(tags=["planner"])


@lru_cache(maxsize=256)
def _cached_plan(
    race_id: str, driver_id: str, grid: int | None, n_simulations: int, track_temp: float | None, rain: bool | None, hour: str
):
    # `hour` refreshes cached upcoming-race plans hourly, as their forecast moves.
    plan = build_race_plan(
        race_id, driver_id, n_simulations=n_simulations, grid_override=grid, track_temp=track_temp, rain=rain
    )
    tyres = recommend_tyre_allocation(race_id, driver_id, plan.compound_sequence)
    return plan, tyres


@router.get("/races/{race_id}/plan", response_model=RacePlanResponse)
def get_race_plan(
    race_id: str,
    driver_id: str = Query(...),
    grid: int | None = Query(None, ge=1, le=22, description="Plan from this grid slot instead of the real one"),
    n_simulations: int = Query(1200, ge=200, le=5000),
    track_temp: float | None = Query(None, ge=0, le=75, description="What-if track temperature, °C"),
    rain: bool | None = Query(None, description="What-if: plan for rain (true) or a dry race (false)"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> RacePlanResponse:
    # The calendar, not silver.races: an upcoming round has no results yet.
    race = con.execute(
        """
        SELECT cal.name, cal.circuit_id, cal.circuit_name, cal.date,
               ch.circuit_baseline_track_temp, ch.historical_overtaking_rate
        FROM silver.calendar cal
        LEFT JOIN gold.circuit_history ch ON ch.race_id = cal.race_id
        WHERE cal.race_id = ?
        """,
        [race_id],
    ).fetchone()
    if race is None:
        raise HTTPException(404, detail=f"No race {race_id!r}")
    season, rnd = race_id.split("_")
    actual = con.execute(
        "SELECT grid FROM bronze.ergast_results WHERE season = ? AND round = ? AND driver_id = ?",
        [int(season), int(rnd), driver_id],
    ).fetchone()

    try:
        from datetime import datetime

        plan, tyres = _cached_plan(race_id, driver_id, grid, n_simulations, track_temp, rain, datetime.now().strftime("%Y%m%d%H"))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc

    def finite(x):
        return None if x is None or x != x else float(x)

    return RacePlanResponse(
        race_id=race_id,
        race_name=race[0],
        circuit_id=race[1],
        circuit_name=race[2],
        date=race[3],
        driver_id=driver_id,
        team_id=plan.team_id,
        grid_position=plan.grid_position,
        actual_grid_position=int(actual[0]) if actual and actual[0] else None,
        grid_is_expected=plan.grid_is_expected,
        total_laps=plan.total_laps,
        laps_known=plan.laps_known,
        is_future=plan.is_future,
        prior_races_at_circuit=plan.prior_races_at_circuit,
        n_simulations=n_simulations,
        conditions={
            "track_temp": plan.track_temp,
            "air_temp": plan.air_temp,
            "rain_expected": plan.rain_expected,
            "circuit_baseline_track_temp": finite(plan.circuit_baseline_track_temp),
            "historical_sc_rate": finite(plan.historical_sc_rate),
            "historical_overtaking_rate": finite(plan.historical_overtaking_rate),
            "pit_loss_seconds": typical_pit_loss_seconds(plan.circuit_id),
            "cold_start_circuit": plan.cold_start_circuit,
            "source": plan.conditions_source,
            "note": plan.conditions_note,
            "rain_probability": plan.rain_probability,
        },
        starting_compound=plan.starting_compound,
        compound_sequence=plan.compound_sequence,
        compound_choice_is_decisive=plan.compound_choice_is_decisive,
        stops=[
            {
                "stop_number": s.stop_number,
                "compound": s.compound,
                "window_open": s.window_open,
                "window_close": s.window_close,
                "nominal_lap": s.nominal_lap,
                "wait": None
                if s.wait is None
                else {
                    "open_lap": s.wait.open_lap,
                    "pull_the_plug_lap": s.wait.pull_the_plug_lap,
                    "latest_safe_lap": s.wait.latest_safe_lap,
                    "caution_saving_seconds": s.wait.caution_saving_seconds,
                    "per_lap_caution_probability": s.wait.per_lap_caution_probability,
                    "has_window": s.wait.has_window,
                    "reason": s.wait.reason,
                },
            }
            for s in plan.stops
        ],
        expected_finish=plan.expected_finish,
        win_probability=plan.win_probability,
        points_probability=plan.points_probability,
        expected_points=plan.expected_points,
        typical_expected_finish=finite(plan.typical_expected_finish),
        typical_win_probability=finite(plan.typical_win_probability),
        typical_points_probability=finite(plan.typical_points_probability),
        typical_expected_points=finite(plan.typical_expected_points),
        strategy_edge_note=STRATEGY_EDGE_EVIDENCE,
        rival_stops_source=plan.rival_stops_source,
        qualifying_in_pace=bool(qualifying_gaps(race_id)),
        starting_options=plan.considered,
        tyres={
            "allocation": tyres.allocation,
            "race_reserved": tyres.race_reserved,
            "quali_reserved": tyres.quali_reserved,
            "practice_budget": tyres.practice_budget,
            "used_in_practice": [
                {"compound": t.compound, "laps_run": t.laps_run, "first_session": t.first_session, "sessions": list(t.sessions)}
                for t in tyres.used_in_practice
            ],
            # Before an unrun weekend, "no practice laps" isn't a data gap —
            # practice hasn't happened, and this is the plan to follow in it.
            "warnings": [w for w in tyres.warnings if not (plan.is_future and w.startswith("No practice"))],
        },
    )


@router.get("/standings", response_model=Standings)
def get_standings(
    season: int | None = Query(None, description="Defaults to the latest season"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> Standings:
    """Race points and wins, drivers and constructors. Sprint points aren't
    in this data, so it's the Grand Prix table, not the official one."""
    if season is None:
        season = con.execute("SELECT MAX(season) FROM bronze.ergast_results").fetchone()[0]
    rows = con.execute(
        """
        SELECT round, driver_id, driver_given_name || ' ' || driver_family_name AS name,
               constructor_id, points, position, status
        FROM bronze.ergast_results WHERE season = ?
        """,
        [season],
    ).df()
    if rows.empty:
        raise HTTPException(404, detail=f"No results for {season}")
    rows["won"] = (rows["position"] == 1) & rows["status"].map(is_classified)
    latest = rows.sort_values("round").groupby("driver_id").last()
    drivers = (
        rows.groupby("driver_id")
        .agg(points=("points", "sum"), wins=("won", "sum"))
        .join(latest[["name", "constructor_id"]])
        .sort_values(["points", "wins"], ascending=False)
    )
    teams = con.execute(
        "SELECT constructor_id, name FROM silver.constructors WHERE season = ?", [season]
    ).df().set_index("constructor_id")["name"].to_dict()
    constructors = (
        rows.groupby("constructor_id").agg(points=("points", "sum"), wins=("won", "sum")).sort_values(["points", "wins"], ascending=False)
    )
    return Standings(
        season=season,
        through_race_id=f"{season}_{int(rows['round'].max())}",
        drivers=[
            StandingEntry(id=d, name=r["name"], team_id=r["constructor_id"], points=float(r["points"]), wins=int(r["wins"]))
            for d, r in drivers.iterrows()
        ],
        constructors=[
            StandingEntry(id=t, name=teams.get(t, t), team_id=t, points=float(r.points), wins=int(r.wins))
            for t, r in constructors.iterrows()
        ],
    )


@router.get("/calendar", response_model=list[CalendarRound])
def get_calendar(
    season: int | None = Query(None, description="Defaults to the latest season on the calendar"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[CalendarRound]:
    """Every round of a season, run or not — the planner plans both."""
    if season is None:
        season = con.execute("SELECT MAX(season) FROM silver.calendar").fetchone()[0]
    rows = con.execute(
        """
        SELECT race_id, season, round, name, circuit_id, circuit_name, country, date, has_results
        FROM silver.calendar WHERE season = ? ORDER BY round
        """,
        [season],
    ).df()
    return [CalendarRound(**r) for r in rows.to_dict(orient="records")]
