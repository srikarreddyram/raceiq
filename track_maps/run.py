"""Build every circuit's track map and geometry features — PRD Section 7.

For each circuit fetched by reconstruction/fetch.py: rebuild the smoothed
centreline, detect corners from curvature and grade that detection against
FastF1's official corner list, compute the Section 10.3 features, and
write one row per circuit to gold.circuit_geometry plus a standalone SVG to
track_maps/assets/.

Features use the OFFICIAL corners, characterised with this track's own
speed and curvature. Curvature detection is implemented as PRD 7.1 asks
and its agreement with the official list is stored per circuit
(`detection_recall`, `detection_precision`) — but a count of "slow
corners" should not move because a smoothing window caught a kink, so the
authoritative list is what the features count.

`tyre_stress_index` is the one feature PRD 10.3 defines on race outcomes
rather than geometry. It's the circuit's median in-stint lap-time slope
MINUS the global median, in seconds per lap. A ratio (the PRD's "vs global
average") doesn't work: race stint slopes are mostly negative — fuel burns
off faster than tyres wear, median -0.08 s/lap across this data — and a
ratio of two small, mostly-negative numbers swings wildly. As a
difference, the fuel effect common to every circuit largely cancels and
what's left is how much harder this circuit is on tyres than typical. It
is computed over all history, so it is a display/context value, not a
leakage-safe model feature.

Usage:
    uv run python -m track_maps.run
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from models.common.db import get_connection as get_read_connection
from pipelines.gold.db import get_connection
from track_maps.geometry_features import features
from track_maps.reconstruction.fetch import RAW_ROOT
from track_maps.reconstruction.path import (
    build_track,
    corners_from_official,
    detect_corners,
    index_at,
    match_to_official,
    to_svg,
)

logger = logging.getLogger(__name__)

ASSETS = RAW_ROOT.parent.parent.parent / "track_maps" / "assets"

CLASS_COLOURS = {"slow": "#DC2626", "medium": "#C9A84C", "fast": "#16A34A"}


def _tyre_stress() -> dict[str, float]:
    con = get_read_connection()
    try:
        rows = con.execute(
            """
            WITH d AS (
                SELECT r.circuit_id, lf.degradation_rate
                FROM gold.lap_features lf JOIN silver.races r ON r.race_id = lf.race_id
                WHERE lf.degradation_rate IS NOT NULL AND NOT lf.is_pit_lap
                    AND NOT lf.safety_car_active AND NOT lf.vsc_active AND NOT lf.red_flag_active
            )
            SELECT circuit_id,
                   MEDIAN(degradation_rate) - (SELECT MEDIAN(degradation_rate) FROM d) AS stress
            FROM d GROUP BY circuit_id
            """
        ).df()
    finally:
        con.close()
    return {row.circuit_id: float(row.stress) for row in rows.itertuples()}


def _overlay(track, corners, meta, svg) -> dict:
    project = svg["project"]

    def point_at(distance_m: float) -> list[float]:
        i = index_at(track, distance_m)
        px, py = project(np.array([track.x[i]]), np.array([track.y[i]]))
        return [round(float(px[0]), 1), round(float(py[0]), 1)]

    def segment(start_m: float, end_m: float) -> str:
        lap = track.lap_length_m
        distances = np.arange(start_m, end_m, 20.0) % lap
        idx = [index_at(track, d) for d in distances]
        px, py = project(track.x[idx], track.y[idx])
        return "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in zip(px, py))

    corner_nodes = []
    for number, corner in zip(meta.get("_official_numbers", []), corners):
        px, py = project(np.array([corner.x]), np.array([corner.y]))
        corner_nodes.append(
            {
                "number": number,
                "x": round(float(px[0]), 1),
                "y": round(float(py[0]), 1),
                "speed_class": corner.speed_class,
                "min_speed_kph": round(corner.min_speed_kph, 1),
                "radius_m": round(corner.radius_m, 1) if np.isfinite(corner.radius_m) else None,
            }
        )

    _, drs_zones = features.drs_total_length(meta.get("drs_zones_m"), track.lap_length_m)
    return {
        "viewbox": svg["viewbox"],
        "start_finish": point_at(0.0),
        "sector_boundaries": [point_at(d) for d in meta.get("sector_boundaries_m") or []],
        "corners": corner_nodes,
        # null (not []) when the reference race's regulations have no DRS.
        "drs_zones": None if meta.get("drs_zones_m") is None else [segment(s, e) for s, e in drs_zones],
    }


def _standalone_svg(svg: dict, overlay: dict, circuit_id: str) -> str:
    size = svg["viewbox"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size:.0f} {size:.0f}">',
        f"<title>{circuit_id}</title>",
        f'<path d="{svg["path"]}" fill="none" stroke="#F0F0F0" stroke-width="10" '
        'stroke-linejoin="round" stroke-linecap="round"/>',
    ]
    for zone in overlay["drs_zones"] or []:
        parts.append(f'<path d="{zone}" fill="none" stroke="#16A34A" stroke-width="10" stroke-linecap="round"/>')
    for c in overlay["corners"]:
        parts.append(
            f'<circle cx="{c["x"]}" cy="{c["y"]}" r="9" fill="{CLASS_COLOURS[c["speed_class"]]}"/>'
        )
    sx, sy = overlay["start_finish"]
    parts.append(f'<rect x="{sx - 4}" y="{sy - 14}" width="8" height="28" fill="#E10600"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_all() -> pd.DataFrame:
    stress = _tyre_stress()
    rows = []
    ASSETS.mkdir(parents=True, exist_ok=True)

    for circuit_path in sorted(p for p in RAW_ROOT.iterdir() if (p / "meta.json").exists()):
        circuit_id = circuit_path.name
        try:
            meta = json.loads((circuit_path / "meta.json").read_text())
            telemetry = pd.read_parquet(circuit_path / "telemetry.parquet")
            official = pd.read_parquet(circuit_path / "corners.parquet")

            track = build_track(telemetry)
            detected = detect_corners(track)
            grading = match_to_official(detected, official, track.lap_length_m)
            corners = corners_from_official(track, official) if len(official) else detected
            meta["_official_numbers"] = [
                f"{int(n)}{l}" for n, l in zip(official["Number"], official["Letter"].fillna(""))
            ] if len(official) else [str(i + 1) for i in range(len(detected))]

            row = features.compute(track, corners, meta, circuit_id)
            row["tyre_stress_index"] = round(stress[circuit_id], 4) if circuit_id in stress else None
            row["detected_corner_count"] = grading["detected_corners"]
            row["detection_recall"] = round(grading["recall"], 3)
            row["detection_precision"] = round(grading["precision"], 3)

            svg = to_svg(track, meta["rotation_degrees"])
            overlay = _overlay(track, corners, meta, svg)
            row["svg_path"] = svg["path"]
            row["overlay_json"] = json.dumps(overlay)
            rows.append(row)

            (ASSETS / f"{circuit_id}.svg").write_text(_standalone_svg(svg, overlay, circuit_id))
        except Exception:
            logger.exception("track map build failed for %s", circuit_id)

    return pd.DataFrame(rows)


def write_gold(df: pd.DataFrame) -> None:
    con = get_connection()
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        con.register("geometry_df", df)
        con.execute("CREATE OR REPLACE TABLE gold.circuit_geometry AS SELECT * FROM geometry_df")
        count = con.execute("SELECT COUNT(*) FROM gold.circuit_geometry").fetchone()[0]
        logger.info("gold.circuit_geometry: %s rows", count)
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_all()
    write_gold(df)


if __name__ == "__main__":
    main()
