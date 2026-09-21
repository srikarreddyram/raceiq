"""Turn one lap of telemetry into a track: a smooth centreline, its
curvature, detected corners, and an SVG path — PRD Section 7.1 steps 1-3
and 6.

The path is resampled to uniform 5 m spacing along the lap *before*
smoothing or differentiating. Raw telemetry arrives at uneven intervals
(faster on straights, where the car covers more ground per sample), and
curvature taken by differencing unevenly-spaced points is dominated by
the spacing rather than the shape.

Smoothing is the Savitzky-Golay filter PRD 7.1 names, over a window of
~75 m. That's long enough to remove the jitter in position samples and
short enough to keep a hairpin a hairpin; the window is odd-length and the
filter treats the path as periodic, since a lap is a closed loop and
padding the ends with anything else would distort turn 1 and the final
corner.

Corners are detected as the PRD specifies — local maxima of curvature —
and then checked against FastF1's official corner list for the same
circuit (see `match_to_official`). Detection is the method the PRD asks
for; the official list is what it gets graded against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

RESAMPLE_STEP_M = 5.0
SMOOTHING_WINDOW_M = 75.0
SMOOTHING_POLYORDER = 3

# A corner is a curvature peak tighter than a 400 m radius. Anything gentler
# is a kink a car takes flat, which is not what "corner" means for tyre or
# downforce demand.
CORNER_MAX_RADIUS_M = 400.0
# Two curvature peaks closer than this are one corner, not two: most
# corners show separate entry and exit peaks. Tuned against FastF1's
# official corner lists rather than chosen — across 245 official corners:
#
#   separation   detected/official   recall
#      60 m            1.77           0.978
#     100 m            1.22           0.978
#     140 m            0.96           0.960   <- chosen
#     180 m            0.80           0.897
#
# 140 m is where the count matches the official list without starting to
# merge genuinely separate corners (recall only begins falling beyond it).
CORNER_MIN_SEPARATION_M = 140.0

# PRD 7.1's speed classes.
SLOW_CORNER_MAX_KPH = 120.0
FAST_CORNER_MIN_KPH = 200.0


@dataclass
class Track:
    distance: np.ndarray  # metres along the lap, uniform spacing
    x: np.ndarray  # metres, centred on the circuit's centroid
    y: np.ndarray
    z: np.ndarray  # elevation, metres, relative
    speed: np.ndarray  # km/h
    throttle: np.ndarray
    brake: np.ndarray  # bool
    curvature: np.ndarray  # 1/m, signed
    lap_length_m: float


@dataclass
class Corner:
    distance_m: float
    x: float
    y: float
    radius_m: float
    min_speed_kph: float

    @property
    def speed_class(self) -> str:
        if self.min_speed_kph < SLOW_CORNER_MAX_KPH:
            return "slow"
        if self.min_speed_kph > FAST_CORNER_MIN_KPH:
            return "fast"
        return "medium"


def build_track(telemetry: pd.DataFrame) -> Track:
    tel = telemetry.dropna(subset=["Distance", "X", "Y"]).sort_values("Distance")
    tel = tel.drop_duplicates(subset="Distance")

    lap_length = float(tel["Distance"].max())
    grid = np.arange(0.0, lap_length, RESAMPLE_STEP_M)

    def resample(column: str) -> np.ndarray:
        return np.interp(grid, tel["Distance"].to_numpy(), tel[column].to_numpy(dtype=float))

    # FastF1 positions are decimetres; everything downstream is metres.
    x = resample("X") / 10.0
    y = resample("Y") / 10.0
    z = resample("Z") / 10.0
    x -= x.mean()
    y -= y.mean()

    window = int(SMOOTHING_WINDOW_M / RESAMPLE_STEP_M) | 1  # force odd
    x = savgol_filter(x, window, SMOOTHING_POLYORDER, mode="wrap")
    y = savgol_filter(y, window, SMOOTHING_POLYORDER, mode="wrap")
    z = savgol_filter(z, window, SMOOTHING_POLYORDER, mode="wrap")

    # Signed curvature of a plane curve, with derivatives taken per metre of
    # arc length (the grid spacing), so the result is in 1/m.
    dx = np.gradient(x, RESAMPLE_STEP_M)
    dy = np.gradient(y, RESAMPLE_STEP_M)
    ddx = np.gradient(dx, RESAMPLE_STEP_M)
    ddy = np.gradient(dy, RESAMPLE_STEP_M)
    curvature = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5).clip(min=1e-9)

    return Track(
        distance=grid,
        x=x,
        y=y,
        z=z - z.min(),
        speed=resample("Speed"),
        throttle=resample("Throttle"),
        brake=resample("Brake") > 0.5,
        curvature=curvature,
        lap_length_m=lap_length,
    )


def _corner_at(track: Track, index: int) -> Corner:
    # Minimum speed in a ±60 m window: the apex speed, which is what PRD
    # 7.1's slow/medium/fast classes are defined on, and which usually sits
    # a few metres off the exact curvature peak.
    half = int(60 / RESAMPLE_STEP_M)
    lo, hi = max(0, index - half), min(len(track.speed), index + half + 1)
    kappa = abs(track.curvature[index])
    return Corner(
        distance_m=float(track.distance[index]),
        x=float(track.x[index]),
        y=float(track.y[index]),
        radius_m=float(1.0 / kappa) if kappa > 0 else float("inf"),
        min_speed_kph=float(track.speed[lo:hi].min()),
    )


def detect_corners(track: Track) -> list[Corner]:
    peaks, _ = find_peaks(
        np.abs(track.curvature),
        height=1.0 / CORNER_MAX_RADIUS_M,
        distance=int(CORNER_MIN_SEPARATION_M / RESAMPLE_STEP_M),
    )
    return [_corner_at(track, int(i)) for i in peaks]


def corners_from_official(track: Track, official: pd.DataFrame) -> list[Corner]:
    """Characterise FastF1's official corners using this track's own
    speed and curvature at each one's distance along the lap."""
    corners = []
    for row in official.itertuples():
        index = int(np.clip(round(row.Distance / RESAMPLE_STEP_M), 0, len(track.distance) - 1))
        # The official marker is placed at the corner's number board, not its
        # apex. Take the tightest point within ±40 m so radius describes the
        # corner rather than the approach to it.
        half = int(40 / RESAMPLE_STEP_M)
        lo, hi = max(0, index - half), min(len(track.curvature), index + half + 1)
        apex = lo + int(np.argmax(np.abs(track.curvature[lo:hi])))
        corners.append(_corner_at(track, apex))
    return corners


def match_to_official(detected: list[Corner], official: pd.DataFrame, lap_length: float, tolerance_m: float = 120.0) -> dict:
    """How well curvature detection recovers the official corner list.

    Recall: share of official corners with a detected corner within
    `tolerance_m` along the lap. Precision: share of detected corners that
    land near an official one. Distances wrap at the start/finish line.
    """
    official_d = official["Distance"].to_numpy(dtype=float)
    detected_d = np.array([c.distance_m for c in detected], dtype=float)

    def near(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(b) == 0:
            return np.zeros(len(a), dtype=bool)
        gap = np.abs(a[:, None] - b[None, :])
        gap = np.minimum(gap, lap_length - gap)
        return gap.min(axis=1) <= tolerance_m

    recall = float(near(official_d, detected_d).mean()) if len(official_d) else float("nan")
    precision = float(near(detected_d, official_d).mean()) if len(detected_d) else float("nan")
    return {
        "official_corners": int(len(official_d)),
        "detected_corners": int(len(detected_d)),
        "recall": recall,
        "precision": precision,
    }


def to_svg(track: Track, rotation_degrees: float, viewbox: float = 1000.0, padding: float = 40.0) -> dict:
    """Project the centreline into a square SVG viewBox.

    Rotated by FastF1's circuit rotation so the map faces the way the
    official broadcast graphic does, and y flipped because SVG's y axis
    points down. Returns the path plus the same transform as a function, so
    corners, sector boundaries and DRS zones land in the same coordinates.
    """
    theta = np.radians(rotation_degrees)
    cos, sin = np.cos(theta), np.sin(theta)

    def rotate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x * cos - y * sin, x * sin + y * cos

    rx, ry = rotate(track.x, track.y)
    ry = -ry
    span = max(rx.max() - rx.min(), ry.max() - ry.min())
    scale = (viewbox - 2 * padding) / span
    offset_x = (viewbox - (rx.max() - rx.min()) * scale) / 2 - rx.min() * scale
    offset_y = (viewbox - (ry.max() - ry.min()) * scale) / 2 - ry.min() * scale

    def project(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, py = rotate(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return px * scale + offset_x, -py * scale + offset_y

    px, py = project(track.x, track.y)
    step = 4  # every 20 m: smooth enough to draw, small enough to ship
    points = " L ".join(f"{a:.1f} {b:.1f}" for a, b in zip(px[::step], py[::step]))
    return {"path": f"M {points} Z", "project": project, "viewbox": viewbox}


def index_at(track: Track, distance_m: float) -> int:
    return int(np.clip(round(distance_m / RESAMPLE_STEP_M), 0, len(track.distance) - 1))
