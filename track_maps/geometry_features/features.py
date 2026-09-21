"""Per-circuit geometry features — PRD Section 10.3.

Everything here describes the circuit, computed from one reference lap
(see reconstruction/fetch.py), except the two features the PRD itself
defines on race outcomes, which are marked as such below.

Deliberate departures from PRD 10.3, each for the same reason — a number
that looks right but measures something else is worse than a gap:

- `track_evolution_rate` is left null. The PRD defines it as lap-time
  improvement from lap 1 to lap 10 "as rubber lays down". Measured on race
  laps, that improvement is dominated by fuel burning off, not by the
  track, so it would carry the right name and the wrong meaning. It needs
  practice sessions at scale, and this project ingests those for only one
  race so far.
- `drs_zone_total_length_m` is null for any circuit whose reference race is
  2026 or later. DRS was abolished by the 2026 regulations; zero would
  claim the circuit has DRS and no zones.
- `downforce_demand_index` is defined here as the share of the lap NOT
  spent at full throttle. PRD 10.3 only says "weighted index from corner
  speed distribution". The reasoning for this choice: a high-downforce
  setup costs time on the straights, so how much of a lap is flat-out is
  what decides how much downforce a team can afford to run. It is checked
  against reality in run.py (Monza should come out lowest, the tight
  street and twisty circuits highest).

Corner speed classes use PRD 7.1's fixed thresholds (slow < 120 km/h,
fast > 200 km/h). Those were written for one generation of car; a circuit
last raced in 2026 was lapped by a different car than one last raced in
2025, so `reference_season` is stored beside every row.
"""

from __future__ import annotations

import numpy as np

from strategy_engine.pit_loss import typical_pit_loss_seconds
from track_maps.reconstruction.path import RESAMPLE_STEP_M, Corner, Track

FULL_THROTTLE = 98.0  # percent; FastF1 throttle rarely reads exactly 100 when flat


def _distance_weighted_mean(values: np.ndarray) -> float:
    # The track is resampled at uniform distance, so a plain mean IS the
    # distance-weighted mean — which is what "average speed through a
    # sector" should be. A time-weighted mean would over-count slow corners.
    return float(np.mean(values)) if len(values) else float("nan")


def sector_speeds(track: Track, boundaries_m: list[float]) -> tuple[float, float, float]:
    if len(boundaries_m) != 2:
        return (float("nan"),) * 3
    b1, b2 = boundaries_m
    s1 = track.speed[track.distance < b1]
    s2 = track.speed[(track.distance >= b1) & (track.distance < b2)]
    s3 = track.speed[track.distance >= b2]
    return _distance_weighted_mean(s1), _distance_weighted_mean(s2), _distance_weighted_mean(s3)


def drs_total_length(zones_m: list | None, lap_length_m: float) -> tuple[float | None, list]:
    """Total DRS length, with the start/finish line handled.

    A zone that runs across the line shows up as two pieces — one starting
    at 0 m and one ending near the lap length. Bahrain's main straight did
    exactly that. Those are one zone and are merged.
    """
    if zones_m is None:
        return None, []
    if not zones_m:
        # A DRS-era race where the flap never opened on any fastest lap —
        # Istanbul 2021, run on intermediates with DRS disabled for most of
        # it, is the real case. Every DRS-era circuit has at least one zone,
        # so zero is never the truth; "not observed" is, and that's null.
        return None, []
    zones = [list(z) for z in zones_m]
    if len(zones) >= 2 and zones[0][0] < 50 and zones[-1][1] > lap_length_m - 250:
        wrap = [zones[-1][0], zones[0][1] + lap_length_m]
        zones = [wrap] + zones[1:-1]
    return float(sum(end - start for start, end in zones)), zones


def compute(track: Track, corners: list[Corner], meta: dict, circuit_id: str) -> dict:
    classes = [c.speed_class for c in corners]
    corner_count = len(corners)
    slow = classes.count("slow")
    medium = classes.count("medium")
    fast = classes.count("fast")
    finite_radii = [c.radius_m for c in corners if np.isfinite(c.radius_m)]

    s1, s2, s3 = sector_speeds(track, meta.get("sector_boundaries_m") or [])
    full_throttle_fraction = float(np.mean(track.throttle >= FULL_THROTTLE))
    drs_length, _ = drs_total_length(meta.get("drs_zones_m"), track.lap_length_m)

    return {
        "circuit_id": circuit_id,
        "reference_race_id": meta["reference_race_id"],
        "reference_season": int(meta["reference_season"]),
        "lap_length_m": round(track.lap_length_m, 1),
        "corner_count": corner_count,
        "slow_corner_count": slow,
        "medium_corner_count": medium,
        "fast_corner_count": fast,
        "slow_corner_pct": round(slow / corner_count, 4) if corner_count else None,
        "avg_corner_radius_m": round(float(np.mean(finite_radii)), 1) if finite_radii else None,
        "total_braking_distance_m": round(float(track.brake.sum() * RESAMPLE_STEP_M), 1),
        "elevation_range_m": round(float(track.z.max() - track.z.min()), 2),
        "elevation_variance": round(float(np.var(track.z)), 3),
        "sector1_avg_speed": round(s1, 2),
        "sector2_avg_speed": round(s2, 2),
        "sector3_avg_speed": round(s3, 2),
        "full_throttle_fraction": round(full_throttle_fraction, 4),
        "downforce_demand_index": round(1.0 - full_throttle_fraction, 4),
        "drs_zone_total_length_m": None if drs_length is None else round(drs_length, 1),
        "pit_lane_delta": round(typical_pit_loss_seconds(circuit_id), 2),
        "track_evolution_rate": None,
    }
