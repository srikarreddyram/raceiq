"""How far to trust a race-day rain forecast: the archived forecast for
every race since 2022 against what happened on track.

Open-Meteo keeps the forecasts it issued (the historical-forecast API),
so for each race this fetches the chance of rain around the start — the
window race_plan/future.py reads, an hour either side of the start time
— and compares it with the race: whether rain fell (the track's sensor),
whether the field started on wet-weather tyres, and whether it ran on
them at any point. The archive has no rain-chance figure before 2024.

The table it prints is the one race_plan/wet.py's thresholds and
FORECAST_TRACK_RECORD come from; re-run it as races are added.

Usage:
    uv run python -m race_plan.rain_forecast_check
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from models.common.db import get_connection

ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _forecast(lat: float, lon: float, date: str) -> list | None:
    for _ in range(3):
        try:
            r = requests.get(
                ARCHIVE_URL,
                timeout=30,
                params={"latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
                        "hourly": "precipitation_probability", "timezone": "UTC"},
            )
            r.raise_for_status()
            return r.json().get("hourly", {}).get("precipitation_probability")
        except requests.RequestException:
            time.sleep(2)
    return None


def main() -> None:
    con = get_connection()
    try:
        races = con.execute(
            "SELECT race_id, date, time_utc, latitude, longitude FROM silver.calendar WHERE has_results AND season >= 2022 ORDER BY date"
        ).df()
        laps = con.execute("SELECT race_id, lap_number, compound FROM gold.lap_features WHERE race_id LIKE '202%'").df()
        rain = con.execute("SELECT race_id, BOOL_OR(rainfall) AS rain FROM silver.weather WHERE race_id LIKE '202%' GROUP BY 1").df()
    finally:
        con.close()

    rows = []
    for r in races.itertuples():
        hour = int(str(r.time_utc or "13:00:00")[:2])
        p = _forecast(float(r.latitude), float(r.longitude), str(r.date)[:10])
        window = [x for x in (p or [])[max(0, hour - 1): hour + 2] if x is not None]
        if window:
            rows.append({"race_id": r.race_id, "chance": max(window) / 100})
        time.sleep(0.2)
    f = pd.DataFrame(rows)
    wet = laps.assign(w=laps["compound"].isin(["INTERMEDIATE", "WET"])).groupby(["race_id", "lap_number"])["w"].mean() >= 0.5
    f = f.merge(wet.groupby("race_id").any().rename("wet_tyres").reset_index(), on="race_id", how="left")
    f = f.merge(rain, on="race_id", how="left")
    f["band"] = pd.cut(f["chance"], [-0.01, 0.1, 0.3, 0.5, 0.9, 1.0], labels=["0-10%", "10-30%", "30-50%", "50-90%", "90-100%"])
    table = f.groupby("band", observed=True).agg(races=("race_id", "size"), rain_fell=("rain", "mean"), wet_tyres=("wet_tyres", "sum"))
    print(f"{len(f)} races with an archived forecast")
    print(table.to_string())


if __name__ == "__main__":
    main()
