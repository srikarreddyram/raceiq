"""GET /monitoring/* — the Model Performance View of PRD Section 13.2.

Reads what monitoring/run.py writes; it doesn't evaluate models itself.
Scoring six models means loading six datasets, which belongs in a job
(`uv run python -m monitoring.run`), not in a request.
"""

from __future__ import annotations

import json
import math

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query

from car_profiles.season_profile import confidence_over_season
from serving.api.db import get_db
from serving.api.schemas import LapTimeOverlay, ModelHealth, ProfileConfidencePoint

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_NOT_RUN = "No monitoring results yet — run `uv run python -m monitoring.run`"


def _clean(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


@router.get("/models", response_model=list[ModelHealth])
def model_health(con: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[ModelHealth]:
    try:
        rows = con.execute("SELECT * FROM monitoring.model_health").df()
    except duckdb.CatalogException as exc:
        raise HTTPException(404, detail=_NOT_RUN) from exc
    out = []
    for r in rows.to_dict(orient="records"):
        r = {k: _clean(v) for k, v in r.items()}
        out.append(
            ModelHealth(
                experiment=r["experiment"],
                label=r["label"],
                kind=r["kind"],
                evaluated_at=str(r["evaluated_at"]),
                status=r["status"],
                n_rows=None if r["n_rows"] is None else int(r["n_rows"]),
                n_races=None if r["n_races"] is None else int(r["n_races"]),
                per_race_metric=r["per_race_metric"],
                per_race_lower_is_better=bool(r["per_race_lower_is_better"]),
                per_race_baseline=r["per_race_baseline"],
                metrics=json.loads(r["metrics_json"]),
                per_race=json.loads(r["per_race_json"]),
                feature_drift=json.loads(r["drift_json"]),
                notes=json.loads(r["notes_json"]),
            )
        )
    return out


@router.get("/lap-time", response_model=LapTimeOverlay)
def lap_time_overlay(
    race_id: str = Query(...),
    driver_id: str = Query(...),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> LapTimeOverlay:
    try:
        rows = con.execute(
            """
            SELECT lap_number, predicted, actual, COALESCE(stable, FALSE) AS stable
            FROM monitoring.lap_time_predictions
            WHERE race_id = ? AND driver_id = ?
            ORDER BY lap_number
            """,
            [race_id, driver_id],
        ).df()
    except duckdb.CatalogException as exc:
        raise HTTPException(404, detail=_NOT_RUN) from exc
    if rows.empty:
        raise HTTPException(404, detail=f"No 2026 lap-time predictions for {driver_id!r} in {race_id!r}")
    stable = rows[rows["stable"]]
    mae = float((stable["predicted"] - stable["actual"]).abs().mean()) if len(stable) else None
    return LapTimeOverlay(
        race_id=race_id,
        driver_id=driver_id,
        laps=rows.to_dict(orient="records"),
        mae_stable=None if mae is None else round(mae, 3),
    )


@router.get("/car-profile-confidence", response_model=list[ProfileConfidencePoint])
def car_profile_confidence(
    season: int = Query(2026),
    con: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[ProfileConfidencePoint]:
    return [ProfileConfidencePoint(**r) for r in confidence_over_season(con, season)]
