"""FastAPI app — PRD Section 14.

Endpoints not implemented here, with reasons (not silently dropped):
- `GET /circuits/{id}/map` needs track_maps/ (GPS-reconstructed SVG
  geometry), not built.
- `GET /teams/{id}/car-profile` and `PUT .../seed` need car_profiles/,
  not built.
- `GET /telemetry/{lap_id}` needs per-lap telemetry ingestion, skipped so
  far to stay under FastF1's rate limit (see ingestion/fastf1/client.py).
- `POST /retrain` needs the nightly retraining pipeline (PRD Section 15,
  Airflow-orchestrated), not built — there's nothing to trigger yet.

Run with:
    uv run uvicorn serving.api.main:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.api.routers import circuits, drivers, predictions, races, teams

app = FastAPI(
    title="RaceIQ API",
    description="Circuit-adaptive F1 race strategy intelligence — PRD Section 14",
    version="0.1.0",
)

# The frontend (PRD Section 13) is served by Vite on its own port in
# development, so every browser request to this API is cross-origin.
# Defaults to the local Vite ports only — an explicit list rather than
# `allow_origins=["*"]`, since a wildcard here would be a habit worth not
# forming before this is ever deployed anywhere real.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("RACEIQ_CORS_ORIGINS", _DEFAULT_ORIGINS).split(","),
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type"],
)

app.include_router(circuits.router)
app.include_router(drivers.router)
app.include_router(races.router)
app.include_router(teams.router)
app.include_router(predictions.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
