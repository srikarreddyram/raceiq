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

from fastapi import FastAPI

from serving.api.routers import circuits, drivers, predictions, races

app = FastAPI(
    title="RaceIQ API",
    description="Circuit-adaptive F1 race strategy intelligence — PRD Section 14",
    version="0.1.0",
)

app.include_router(circuits.router)
app.include_router(drivers.router)
app.include_router(races.router)
app.include_router(predictions.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
