"""Bronze loader for Ergast-schema data (results, qualifying, constructor
standings, circuits) fetched via the Jolpica-F1 mirror.

Each raw file is one JSON envelope `{source, ingested_at, payload}` where
`payload` is Ergast's native (and deeply nested) response shape. This
loader's job is entirely to flatten that nesting into flat rows with typed
columns — it does not yet reconcile Ergast's `driverId` (e.g.
"max_verstappen") against FastF1's `Abbreviation` (e.g. "VER"); that
identifier normalisation is Silver's job (PRD Section 7), though Ergast's
`Driver.code` field — also "VER" — is what makes that join possible, so
it's carried through here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pipelines.bronze.common import write_bronze_table

logger = logging.getLogger(__name__)


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    return envelope["payload"], envelope["ingested_at"]


def load_results(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for path in sorted(raw_root.glob("ergast/*/*/results.json")):
        season, round_number = int(path.parts[-3]), int(path.parts[-2])
        payload, ingested_at = _load_envelope(path)
        races = payload["MRData"]["RaceTable"]["Races"]
        if not races:
            continue

        for result in races[0]["Results"]:
            rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": result["Driver"]["driverId"],
                    "driver_code": result["Driver"].get("code"),
                    "driver_given_name": result["Driver"].get("givenName"),
                    "driver_family_name": result["Driver"].get("familyName"),
                    "driver_nationality": result["Driver"].get("nationality"),
                    "driver_date_of_birth": result["Driver"].get("dateOfBirth"),
                    "constructor_id": result["Constructor"]["constructorId"],
                    "grid": int(result["grid"]),
                    "laps_completed": int(result["laps"]),
                    "status": result["status"],
                    "position": pd.to_numeric(result.get("position"), errors="coerce"),
                    "points": float(result["points"]),
                    "finish_time_millis": pd.to_numeric(
                        result.get("Time", {}).get("millis"), errors="coerce"
                    ),
                    "_source": "ergast",
                    "_ingested_at": ingested_at,
                }
            )

    write_bronze_table(
        con, "ergast_results", pd.DataFrame(rows), dedup_keys=["season", "round", "driver_id"]
    )


def load_qualifying(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for path in sorted(raw_root.glob("ergast/*/*/qualifying.json")):
        season, round_number = int(path.parts[-3]), int(path.parts[-2])
        payload, ingested_at = _load_envelope(path)
        races = payload["MRData"]["RaceTable"]["Races"]
        if not races:
            continue

        for result in races[0]["QualifyingResults"]:
            rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": result["Driver"]["driverId"],
                    "driver_code": result["Driver"].get("code"),
                    "constructor_id": result["Constructor"]["constructorId"],
                    "position": int(result["position"]),
                    "q1": result.get("Q1"),
                    "q2": result.get("Q2"),
                    "q3": result.get("Q3"),
                    "_source": "ergast",
                    "_ingested_at": ingested_at,
                }
            )

    write_bronze_table(
        con, "ergast_qualifying", pd.DataFrame(rows), dedup_keys=["season", "round", "driver_id"]
    )


def load_constructor_standings(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for path in sorted(raw_root.glob("ergast/*/*/constructor_standings.json")):
        season, round_number = int(path.parts[-3]), int(path.parts[-2])
        payload, ingested_at = _load_envelope(path)
        lists_ = payload["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists_:
            continue

        for standing in lists_[0]["ConstructorStandings"]:
            rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "constructor_id": standing["Constructor"]["constructorId"],
                    "constructor_name": standing["Constructor"]["name"],
                    "position": int(standing["position"]),
                    "points": float(standing["points"]),
                    "wins": int(standing["wins"]),
                    "_source": "ergast",
                    "_ingested_at": ingested_at,
                }
            )

    write_bronze_table(
        con,
        "ergast_constructor_standings",
        pd.DataFrame(rows),
        dedup_keys=["season", "round", "constructor_id"],
    )


def load_circuits(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    rows = []
    for path in sorted(raw_root.glob("ergast/*/circuits.json")):
        payload, ingested_at = _load_envelope(path)
        for circuit in payload["MRData"]["CircuitTable"]["Circuits"]:
            rows.append(
                {
                    "circuit_id": circuit["circuitId"],
                    "name": circuit["circuitName"],
                    "country": circuit["Location"]["country"],
                    "locality": circuit["Location"]["locality"],
                    "latitude": float(circuit["Location"]["lat"]),
                    "longitude": float(circuit["Location"]["long"]),
                    "_source": "ergast",
                    "_ingested_at": ingested_at,
                }
            )

    write_bronze_table(con, "ergast_circuits", pd.DataFrame(rows), dedup_keys=["circuit_id"])


def load_all(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    load_circuits(raw_root, con)
    load_results(raw_root, con)
    load_qualifying(raw_root, con)
    load_constructor_standings(raw_root, con)
