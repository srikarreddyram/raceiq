"""GET /races — PRD Section 14."""

from __future__ import annotations

import duckdb
from fastapi import APIRouter, Depends, Query

from serving.api.db import get_db
from serving.api.schemas import Race

router = APIRouter(prefix="/races", tags=["races"])


@router.get("", response_model=list[Race])
def list_races(
    season: int | None = Query(None, description="Filter to a single season"),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[Race]:
    query = "SELECT race_id, season, round, circuit_id, name, date FROM silver.races"
    params = []
    if season is not None:
        query += " WHERE season = ?"
        params.append(season)
    query += " ORDER BY season, round"

    rows = con.execute(query, params).df()
    return [Race(**row) for row in rows.to_dict(orient="records")]
