"""Weather client, keyed to circuit coordinates and session dates (PRD 6.4).

Defaults to Open-Meteo's historical archive API: free, keyless, and queried
by raw latitude/longitude + date, which is exactly how weather needs to be
keyed to a circuit. It returns hourly observations — collapsing that to
lap-level granularity is a Silver-layer join/interpolation job (Section 7),
not something ingestion does.

Circuit coordinates themselves come from the Ergast `circuits` payload
(Section 6.2), not from here — this client only knows how to turn
(lat, lon, date range) into weather observations.
"""

from __future__ import annotations

from typing import Any

from ingestion.config import IngestionConfig
from ingestion.http import get_json

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
]


def get_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: IngestionConfig,
) -> dict[str, Any]:
    """Fetch hourly weather for a circuit over a date range (YYYY-MM-DD)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }
    return get_json(config.weather_base_url, config, params=params)
