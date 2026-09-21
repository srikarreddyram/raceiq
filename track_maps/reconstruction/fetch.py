"""Pull the raw material for one circuit's track map out of FastF1.

One reference race per circuit — the most recent one in the warehouse, so
a reconfigured layout (Abu Dhabi 2021, PRD 7.1's own example) is picked
up automatically rather than being averaged with the old one. From it:

- **The fastest lap's telemetry** — position (X/Y/Z), speed, throttle,
  brake and distance along the lap. One clean lap describes the geometry;
  averaging many would blur every corner.
- **DRS zones, from many laps, not that one — and only where DRS exists.**
  The fastest lap in a race is often set in clean air, where DRS never
  opens (Bahrain 2025's reads closed for the entire lap), so zones are the
  union of where it was actually open across every driver's fastest lap.
  But DRS was abolished by the 2026 regulations: FastF1's DRS channel for a
  2026 race is all zeros (checked on Zandvoort — 2025 shows open-flap
  values 12/14, 2026 shows only 0). A circuit whose reference race is 2026
  or later therefore gets `drs_zones_m = null`, not an empty list: an empty
  list would claim "this circuit has DRS and no zones", and the feature
  built from it would read 0 m, when the truth is that the concept doesn't
  apply. 2026's replacements (active aero and the overtake mode) aren't in
  that channel, so they aren't mapped here at all rather than guessed at.
- **Official corners and sector boundaries**, from FastF1's circuit info
  and the lap's own sector timestamps. The corners are the ground truth the
  PRD's curvature-based detection is validated against.

FastF1 position data is already in a local Cartesian frame (decimetres),
not GPS latitude/longitude, so PRD 7.1's "convert lat/lng to Cartesian"
step reduces to re-centring and converting units — done in path.py.

Loading telemetry is the most expensive call this project makes against
FastF1's ~500/hour limit, so every circuit is cached to disk once and
skipped on re-runs, with the same backoff ingestion/backfill.py uses.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from ingestion.fastf1.client import enable_cache
from models.common.db import get_connection

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw" / "track_maps"
FASTF1_CACHE = REPO_ROOT / ".fastf1_cache"

RATE_LIMIT_BACKOFF_SECONDS = 20 * 60
RATE_LIMIT_MAX_RETRIES = 4

# FastF1's DRS channel: 10, 12 and 14 mean the flap is open; lower values
# are closed or "eligible but not deployed".
DRS_OPEN_THRESHOLD = 10

# DRS was removed by the 2026 regulations. Seasons from this one on have no
# DRS zones to find.
FIRST_SEASON_WITHOUT_DRS = 2026

TELEMETRY_COLUMNS = ["SessionTime", "Distance", "X", "Y", "Z", "Speed", "Throttle", "Brake", "nGear", "DRS"]

# The smallest circuit in this data still spans ~580 m in one direction
# (Jeddah, narrow and long). Position data covering less than 300 m is not
# a track — Monaco 2026's feed spans under half a metre while its speed
# channel is perfectly real, so presence alone is not a quality check.
MIN_TRACK_SPAN_M = 300.0


class NoUsableTelemetry(RuntimeError):
    """Raised when no lap in a session has position data to build from."""


def races_by_recency(circuit_id: str) -> list[tuple[str, int, int]]:
    """Every race at a circuit, newest first — the fallback order when the
    newest one turns out to have no position data."""
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT race_id, season, round FROM silver.races r
            WHERE circuit_id = ?
                AND EXISTS (SELECT 1 FROM gold.lap_features lf WHERE lf.race_id = r.race_id)
            ORDER BY date DESC
            """,
            [circuit_id],
        ).fetchall()
    finally:
        con.close()
    return [(race_id, int(season), int(rnd)) for race_id, season, rnd in rows]


def reference_races() -> pd.DataFrame:
    """The most recent race at every circuit this warehouse has."""
    con = get_connection()
    try:
        return con.execute(
            """
            SELECT circuit_id, race_id, season, round
            FROM (
                SELECT r.*, ROW_NUMBER() OVER (PARTITION BY circuit_id ORDER BY date DESC) AS rn
                FROM silver.races r
                WHERE EXISTS (SELECT 1 FROM gold.lap_features lf WHERE lf.race_id = r.race_id)
            )
            WHERE rn = 1
            ORDER BY circuit_id
            """
        ).df()
    finally:
        con.close()


def circuit_dir(circuit_id: str) -> Path:
    return RAW_ROOT / circuit_id


def is_fetched(circuit_id: str) -> bool:
    return (circuit_dir(circuit_id) / "meta.json").exists()


def _drs_zones(session: fastf1.core.Session) -> list[tuple[float, float]]:
    """Distance ranges where any driver's fastest lap had DRS open."""
    open_distances: list[float] = []
    for driver in session.laps["Driver"].unique():
        try:
            lap = session.laps.pick_drivers(driver).pick_fastest()
            if lap is None or pd.isna(lap["LapTime"]):
                continue
            tel = lap.get_car_data().add_distance()
        except Exception:  # noqa: BLE001 -- one bad driver shouldn't sink the circuit
            continue
        open_distances.extend(tel.loc[tel["DRS"] >= DRS_OPEN_THRESHOLD, "Distance"].tolist())

    if not open_distances:
        return []

    # Merge the scatter of open-flap samples into contiguous zones. A gap of
    # more than 150m between samples is the flap genuinely closing, not
    # sampling jitter between two readings in the same zone.
    open_distances.sort()
    zones: list[list[float]] = [[open_distances[0], open_distances[0]]]
    for distance in open_distances[1:]:
        if distance - zones[-1][1] <= 150:
            zones[-1][1] = distance
        else:
            zones.append([distance, distance])
    # A zone shorter than 100m is a single stray sample, not a DRS zone.
    return [(start, end) for start, end in zones if end - start >= 100]


def _load_session(season: int, round_number: int) -> fastf1.core.Session:
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(season, round_number, "R")
            session.load(telemetry=True, weather=False, messages=False)
            return session
        except RateLimitExceededError:
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            logger.warning("rate limited — sleeping %ss", RATE_LIMIT_BACKOFF_SECONDS)
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
    raise RuntimeError("unreachable")


def _official_corners(
    session: fastf1.core.Session, circuit_id: str, race_id: str, telemetry: pd.DataFrame
) -> tuple[pd.DataFrame, float, str | None]:
    """Official corner numbers and positions, plus the map rotation.

    Returns (corners, rotation_degrees, source_race_id). Two things go wrong
    in practice, and both are handled rather than skipping the circuit:

    - FastF1 measures each corner's distance along the session's *fastest*
      lap, so a session whose fastest lap has no position data (Monaco 2026)
      makes get_circuit_info() crash internally. Corner positions don't
      change year to year on an unchanged layout, so an earlier race at the
      same circuit supplies them instead.
    - A brand-new circuit (Madring, 2026) has no circuit info anywhere yet.
      Then there are no official corners at all, and run.py falls back to
      curvature detection — the one case where the PRD's detection method
      is the only source rather than the one being graded.

    Either way, each corner's distance is recomputed by projecting its X/Y
    onto THIS reference lap, so the corners line up with the telemetry
    they'll be read against regardless of which session supplied them.
    """
    candidates = [(race_id, session)] + [
        (other, None) for other, _, _ in races_by_recency(circuit_id) if other != race_id
    ]
    for source_id, source_session in candidates:
        try:
            if source_session is None:
                season, rnd = (int(part) for part in source_id.split("_"))
                source_session = _load_session(season, rnd)
            info = source_session.get_circuit_info()
            if info is None or info.corners is None or info.corners.empty:
                continue
        except Exception:  # noqa: BLE001 -- FastF1 fails inside for broken sessions
            continue

        corners = info.corners[["Number", "Letter", "X", "Y", "Angle"]].copy()
        tx, ty = telemetry["X"].to_numpy(), telemetry["Y"].to_numpy()
        td = telemetry["Distance"].to_numpy()
        corners["Distance"] = [
            float(td[((tx - cx) ** 2 + (ty - cy) ** 2).argmin()]) for cx, cy in zip(corners["X"], corners["Y"])
        ]
        return corners, float(info.rotation), source_id

    empty = pd.DataFrame(columns=["Number", "Letter", "X", "Y", "Angle", "Distance"])
    return empty, 0.0, None


def _sector_boundaries(lap: pd.Series, telemetry: pd.DataFrame) -> list[float]:
    """Distance along the lap where sectors 1 and 2 end."""
    boundaries = []
    for column in ("Sector1SessionTime", "Sector2SessionTime"):
        moment = lap.get(column)
        if moment is None or pd.isna(moment):
            continue
        nearest = (telemetry["SessionTime"] - moment).abs().idxmin()
        boundaries.append(float(telemetry.loc[nearest, "Distance"]))
    return boundaries


def fetch_circuit(circuit_id: str, season: int, round_number: int, race_id: str) -> None:
    enable_cache(FASTF1_CACHE)
    warnings.filterwarnings("ignore")

    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(season, round_number, "R")
            session.load(telemetry=True, weather=False, messages=False)
            break
        except RateLimitExceededError:
            if attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            logger.warning("rate limited on %s — sleeping %ss", circuit_id, RATE_LIMIT_BACKOFF_SECONDS)
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)

    # The fastest lap first, then the next-fastest: FastF1's position feed
    # is sometimes empty for individual laps (Monaco 2026's fastest lap has
    # no position data at all, so merging it fails outright). A slightly
    # slower lap with real coordinates describes the same circuit.
    lap, telemetry = None, None
    for _, candidate in session.laps.pick_quicklaps().sort_values("LapTime").head(20).iterrows():
        try:
            tel = candidate.get_telemetry()
        except Exception:  # noqa: BLE001 -- empty position feed for this lap
            continue
        present = tel[["X", "Y"]].notna().all(axis=1).mean() > 0.95
        span_m = max(tel["X"].max() - tel["X"].min(), tel["Y"].max() - tel["Y"].min()) / 10.0
        if len(tel) > 100 and present and span_m >= MIN_TRACK_SPAN_M:
            lap, telemetry = candidate, tel
            break
    if lap is None:
        raise NoUsableTelemetry(f"no lap in {race_id} has usable position data")

    telemetry = telemetry[TELEMETRY_COLUMNS].copy()
    telemetry["SessionTime"] = telemetry["SessionTime"].dt.total_seconds()

    corners, rotation, corners_source = _official_corners(session, circuit_id, race_id, telemetry)

    out = circuit_dir(circuit_id)
    out.mkdir(parents=True, exist_ok=True)
    telemetry.to_parquet(out / "telemetry.parquet", index=False)
    corners.to_parquet(out / "corners.parquet", index=False)

    raw_lap_for_sectors = lap.copy()
    sector_telemetry = lap.get_telemetry()[["SessionTime", "Distance"]]
    drs_applies = season < FIRST_SEASON_WITHOUT_DRS
    meta = {
        "circuit_id": circuit_id,
        "reference_race_id": race_id,
        # Recorded so every downstream consumer can tell which ruleset the
        # speeds and aero behaviour in this map come from — circuits last
        # raced in 2025 and circuits already raced in 2026 were driven by
        # genuinely different cars.
        "reference_season": season,
        "reference_driver": str(lap["Driver"]),
        "reference_lap_time_seconds": float(lap["LapTime"].total_seconds()),
        "rotation_degrees": rotation,
        # Which race the official corners came from (may differ from the
        # reference race — see _official_corners), or null if none exist yet.
        "official_corners_source": corners_source,
        "sector_boundaries_m": _sector_boundaries(raw_lap_for_sectors, sector_telemetry),
        "drs_zones_m": _drs_zones(session) if drs_applies else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def fetch_all(force: bool = False) -> None:
    for row in reference_races().itertuples():
        if is_fetched(row.circuit_id) and not force:
            continue
        # Newest race first; if its whole session lacks position data, fall
        # back to the previous race at the same circuit rather than leaving
        # the circuit unmapped. The fallback's season is recorded in
        # meta.json, so its regulation era (and whether DRS applies) follows.
        for race_id, season, rnd in races_by_recency(row.circuit_id):
            logger.info("fetching %s from %s", row.circuit_id, race_id)
            try:
                fetch_circuit(row.circuit_id, season, rnd, race_id)
                break
            except NoUsableTelemetry as reason:
                logger.warning("%s — trying an earlier race", reason)
            except Exception:
                logger.exception("track map fetch failed for %s at %s", row.circuit_id, race_id)
                break
