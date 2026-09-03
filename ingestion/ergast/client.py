"""Ergast-schema historical data client (PRD Section 6.2).

The original Ergast API (ergast.com) stopped receiving updates after the
2024 season and is being sunset — this was flagged as a deprecation risk in
the PRD (Section 6.2 / Open Question 18). The fallback is already wired in:
`ingestion.config` points `ergast_base_url` at the Jolpica-F1 project
(api.jolpi.ca/ergast), a community-run mirror that serves the same
request/response schema as the original API. If that also disappears, only
`ERGAST_BASE_URL` needs to change — every function below just appends paths
to `config.ergast_base_url`.
"""

from __future__ import annotations

from typing import Any

from ingestion.config import IngestionConfig
from ingestion.http import get_json


def get_race_results(season: int, round_number: int, config: IngestionConfig) -> dict[str, Any]:
    url = f"{config.ergast_base_url}/{season}/{round_number}/results.json"
    return get_json(url, config)


def get_qualifying_results(season: int, round_number: int, config: IngestionConfig) -> dict[str, Any]:
    url = f"{config.ergast_base_url}/{season}/{round_number}/qualifying.json"
    return get_json(url, config)


def get_constructor_standings(season: int, round_number: int, config: IngestionConfig) -> dict[str, Any]:
    url = f"{config.ergast_base_url}/{season}/{round_number}/constructorStandings.json"
    return get_json(url, config)


def get_circuits(season: int, config: IngestionConfig) -> dict[str, Any]:
    url = f"{config.ergast_base_url}/{season}/circuits.json"
    return get_json(url, config)
