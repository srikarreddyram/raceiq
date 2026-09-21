"""GET /circuits, /circuits/{circuit_id}/profile, /circuits/{circuit_id}/map — PRD Section 14."""

from __future__ import annotations

import json
import math

import duckdb
from fastapi import APIRouter, Depends, HTTPException

from serving.api.db import get_db
from serving.api.schemas import Circuit, CircuitMap, CircuitProfile

router = APIRouter(prefix="/circuits", tags=["circuits"])


@router.get("", response_model=list[Circuit])
def list_circuits(con: duckdb.DuckDBPyConnection = Depends(get_db)) -> list[Circuit]:
    rows = con.execute(
        "SELECT circuit_id, name, country, locality, latitude, longitude FROM silver.circuits ORDER BY name"
    ).df()
    return [Circuit(**row) for row in rows.to_dict(orient="records")]


@router.get("/{circuit_id}/profile", response_model=CircuitProfile)
def get_circuit_profile(circuit_id: str, con: duckdb.DuckDBPyConnection = Depends(get_db)) -> CircuitProfile:
    circuit = con.execute(
        "SELECT circuit_id, name, country, locality, latitude, longitude FROM silver.circuits WHERE circuit_id = ?",
        [circuit_id],
    ).df()
    if circuit.empty:
        raise HTTPException(status_code=404, detail=f"Unknown circuit_id: {circuit_id!r}")

    # Most recent circuit_history row for this circuit — that race's
    # expanding-window priors are the closest thing to "this circuit's
    # profile as of today" (see pipelines/gold/circuit_history.py).
    history = con.execute(
        """
        SELECT ch.prior_races_at_circuit, ch.historical_sc_rate, ch.circuit_baseline_track_temp
        FROM gold.circuit_history ch
        JOIN silver.races r ON r.race_id = ch.race_id
        WHERE ch.circuit_id = ?
        ORDER BY r.date DESC
        LIMIT 1
        """,
        [circuit_id],
    ).df()

    payload = circuit.iloc[0].to_dict()
    if not history.empty:
        payload.update(history.iloc[0].to_dict())
    return CircuitProfile(**payload)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        return None if math.isnan(value) else value
    except TypeError:
        return value


@router.get("/{circuit_id}/map", response_model=CircuitMap)
def get_circuit_map(circuit_id: str, con: duckdb.DuckDBPyConnection = Depends(get_db)) -> CircuitMap:
    """Track map reconstructed from FastF1 telemetry — track_maps/."""
    rows = con.execute("SELECT * FROM gold.circuit_geometry WHERE circuit_id = ?", [circuit_id]).df()
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No track map for {circuit_id!r} — build it with `uv run python -m track_maps.run`",
        )
    row = {key: _finite_or_none(value) for key, value in rows.iloc[0].to_dict().items()}
    overlay = json.loads(row.pop("overlay_json"))
    return CircuitMap(
        **{k: v for k, v in row.items() if k in CircuitMap.model_fields},
        viewbox=overlay["viewbox"],
        start_finish=overlay["start_finish"],
        sector_boundaries=overlay["sector_boundaries"],
        corners=overlay["corners"],
        drs_zones=overlay["drs_zones"],
    )
