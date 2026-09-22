"""GET /teams, /teams/{id}/car-profile, /teams/{id}/tyres — PRD Section 14
and the Car Profile / Tyre views of Section 13.2."""

from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query

from car_profiles.degradation_curves import team_tyre_report
from car_profiles.season_profile import season_profile, seasons_for
from models.tyre_degradation.trace import remaining_life_trace
from serving.api.db import get_db
from serving.api.schemas import CarProfileResponse, Team, TyreReport

router = APIRouter(prefix="/teams", tags=["teams"])


def _latest_season(con: duckdb.DuckDBPyConnection, team_id: str) -> int:
    seasons = seasons_for(con, team_id)
    if not seasons:
        raise HTTPException(404, detail=f"No races for team {team_id!r}")
    return seasons[-1]


@router.get("", response_model=list[Team])
def list_teams(
    season: int | None = Query(None, description="Defaults to the latest season in the warehouse"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[Team]:
    if season is None:
        season = con.execute("SELECT MAX(season) FROM gold.car_race_observations").fetchone()[0]
    rows = con.execute(
        """
        SELECT DISTINCT o.team_id, COALESCE(c.name, o.team_id) AS name, o.season
        FROM gold.car_race_observations o
        LEFT JOIN silver.constructors c ON c.constructor_id = o.team_id AND c.season = o.season
        WHERE o.season = ?
        ORDER BY name
        """,
        [season],
    ).df()
    return [Team(**r) for r in rows.to_dict(orient="records")]


@router.get("/{team_id}/car-profile", response_model=CarProfileResponse)
def get_car_profile(
    team_id: str,
    season: int | None = Query(None, description="Defaults to the team's latest season"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> CarProfileResponse:
    season = season or _latest_season(con, team_id)
    try:
        return CarProfileResponse(**season_profile(con, team_id, season))
    except LookupError as exc:
        raise HTTPException(404, detail=str(exc)) from exc


@router.get("/{team_id}/tyres", response_model=TyreReport)
def get_tyres(
    team_id: str,
    season: int | None = Query(None, description="Defaults to the team's latest season"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> TyreReport:
    season = season or _latest_season(con, team_id)
    try:
        return TyreReport(**team_tyre_report(team_id, season), remaining_life=remaining_life_trace(team_id, season))
    except (LookupError, ValueError) as exc:
        raise HTTPException(404, detail=str(exc)) from exc
