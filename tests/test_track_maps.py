"""Track maps and circuit geometry — PRD Section 7 and 10.3.

Integration tests against the built gold.circuit_geometry table (build it
with `uv run python -m track_maps.run`), plus unit tests for the two
pieces of logic whose failure would be silent: DRS null-vs-zero handling
and start/finish-wrap merging.
"""

from __future__ import annotations

import json

import pytest

from models.common.db import get_connection
from serving.api.routers.circuits import get_circuit_map
from track_maps.geometry_features.features import drs_total_length


@pytest.fixture(scope="module")
def geometry():
    con = get_connection()
    try:
        return con.execute("SELECT * FROM gold.circuit_geometry").df().set_index("circuit_id")
    finally:
        con.close()


def test_drs_null_means_no_drs_not_zero_zones():
    # 2026 regulations: no DRS at all.
    assert drs_total_length(None, 5000.0) == (None, [])
    # DRS era, flap never opened (Istanbul 2021 in the wet) — not observed.
    assert drs_total_length([], 5000.0) == (None, [])


def test_drs_zone_crossing_start_finish_is_merged():
    total, zones = drs_total_length([[0.0, 300.0], [1000.0, 1500.0], [4800.0, 5000.0]], 5000.0)
    assert len(zones) == 2
    assert total == pytest.approx(300 + 500 + 200)


def test_every_2026_reference_has_null_drs(geometry):
    modern = geometry[geometry["reference_season"] >= 2026]
    assert not modern.empty
    assert modern["drs_zone_total_length_m"].isna().all()
    for overlay in modern["overlay_json"]:
        assert json.loads(overlay)["drs_zones"] is None


def test_corner_detection_recovers_official_corners(geometry):
    recall = geometry["detection_recall"].dropna()
    assert len(recall) >= 25
    assert recall.mean() >= 0.9


def test_geometry_features_match_known_circuit_character(geometry):
    dfi = geometry["downforce_demand_index"]
    # Monza is the low-downforce circuit on the calendar; Monaco and the
    # Hungaroring are the high-downforce extremes.
    assert dfi.idxmin() == "monza"
    assert dfi["monaco"] > dfi.median()
    assert dfi["hungaroring"] > dfi.median()
    # Monaco's climb from Sainte Dévote to Casino is ~40 m.
    assert 30 < geometry.loc["monaco", "elevation_range_m"] < 55
    assert geometry.loc["monaco", "slow_corner_count"] >= 6


def test_map_endpoint_serves_overlay():
    con = get_connection()
    try:
        m = get_circuit_map("bahrain", con)
    finally:
        con.close()
    assert m.svg_path.startswith("M ") and m.svg_path.endswith("Z")
    assert len(m.corners) == m.corner_count
    assert all(0 <= c.x <= m.viewbox and 0 <= c.y <= m.viewbox for c in m.corners)
    assert m.drs_zones  # a 2025 reference: DRS era
    assert len(m.sector_boundaries) == 2


def test_lap_time_model_uses_circuit_id_not_geometry():
    # Geometry was tried and reverted (models/lap_time/train.py): it acted as
    # a circuit fingerprint and failed 2-3x at unlike-anything circuits.
    from models.common.circuit_geometry import CIRCUIT_GEOMETRY_COLUMNS
    from models.common.registry import load_latest_model
    from models.lap_time.train import FEATURE_COLUMNS

    assert "circuit_id" in FEATURE_COLUMNS
    assert not set(CIRCUIT_GEOMETRY_COLUMNS) & set(FEATURE_COLUMNS)
    # Whatever the oracle loads must have been trained on exactly this list.
    assert list(load_latest_model("lap_time_prediction").feature_name_) == FEATURE_COLUMNS


def test_every_raced_circuit_has_a_track_map():
    # The Circuit View and any geometry experiment need a map for every
    # circuit raced since 2018 — Madring included.
    from models.common.circuit_geometry import add_circuit_geometry
    from models.common.data import load_race_features

    df = add_circuit_geometry(load_race_features())
    missing = df[df["lap_length_m"].isna()]
    assert missing.empty, f"no geometry for: {sorted(missing['circuit_id'].unique())}"
