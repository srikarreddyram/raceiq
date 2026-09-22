"""GET /drivers and /drivers/{id}/profile — PRD Section 14 and the Driver
View of Section 13.2."""

from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query

from serving.api.db import get_db
from driver_profiles.profile import driver_profile
from serving.api.schemas import Driver, DriverProfile

router = APIRouter(prefix="/drivers", tags=["drivers"])


@router.get("", response_model=list[Driver])
def list_drivers(
    season: int | None = Query(None, description="Filter to a single season; defaults to each driver's most recent"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[Driver]:
    if season is not None:
        rows = con.execute(
            """
            SELECT driver_id, driver_code, given_name, family_name, nationality, season, constructor_id
            FROM silver.drivers WHERE season = ? ORDER BY family_name
            """,
            [season],
        ).df()
    else:
        # One row per driver: their most recent season, so a retired
        # driver's last-known team shows rather than every season they
        # ever raced.
        rows = con.execute(
            """
            SELECT driver_id, driver_code, given_name, family_name, nationality, season, constructor_id
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY driver_id ORDER BY season DESC) AS rn
                FROM silver.drivers
            ) WHERE rn = 1
            ORDER BY family_name
            """
        ).df()
    return [Driver(**row) for row in rows.to_dict(orient="records")]


@router.get("/{driver_id}/profile", response_model=DriverProfile)
def get_driver_profile(
    driver_id: str,
    season: int | None = Query(None, description="Defaults to the driver's latest season"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> DriverProfile:
    if season is None:
        season = con.execute(
            "SELECT MAX(season) FROM bronze.ergast_results WHERE driver_id = ?", [driver_id]
        ).fetchone()[0]
        if season is None:
            raise HTTPException(404, detail=f"No results for driver {driver_id!r}")
    try:
        return DriverProfile(**driver_profile(con, driver_id, season))
    except LookupError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
