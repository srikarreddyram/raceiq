"""Planning a race that hasn't happened yet.

For a completed race the planner starts from that race's own first lap:
its lap count, its entry list, its measured weather. None of that exists
for an upcoming round, so this module builds the same inputs from what IS
known on the Thursday before a race:

  the circuit    every race run there so far — safety-car rate, how hard
                 it is to pass, typical temperatures, how often it rains,
                 how many laps the race runs, and an era-aware retirement
                 rate (the same rules gold.circuit_history applies race by
                 race, computed as of today)
  the conditions an Open-Meteo forecast for race start when the race is
                 within its 16-day range; beyond that, what's typical at
                 this circuit. Either can be overridden. Track temperature
                 comes from forecast air temperature plus how much hotter
                 than the air the track has run here, on average.
  the grid       unknown before qualifying — each driver's average
                 starting slot this season, ranked, unless the strategist
                 sets their own driver's slot
  the field      the current season's entry list, each car's season form

A circuit with no race in this project's data (2018 onward) — Sepang in
2026 — has no priors: the grid-wide rates stand in, the lap count comes
from the calendar's typical race length, and the plan says so.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, replace
from functools import lru_cache

import duckdb
import pandas as pd

from models.common.data import load_race_features
from models.common.db import get_connection
from race_plan.field import _form_or_zero, _season_form, starting_gap_for_position
from race_plan.weekend_pace import qualifying_gaps, weekend_pace
from strategy_engine.field import RivalTrend
from strategy_engine.state import RaceState

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 16
RAIN_PROBABILITY_THRESHOLD = 0.5  # plan for rain when the forecast says it's more likely than not
DEFAULT_TRACK_MINUS_AIR = 12.0  # used only if no circuit has both readings
DEFAULT_PIT_START_COMPOUND = "MEDIUM"


def _era(season: int) -> int:
    return 0 if season <= 2021 else 1 if season <= 2025 else 2


def calendar_entry(con: duckdb.DuckDBPyConnection, race_id: str) -> dict | None:
    row = con.execute("SELECT * FROM silver.calendar WHERE race_id = ?", [race_id]).df()
    return None if row.empty else row.iloc[0].to_dict()


def has_race_data(con: duckdb.DuckDBPyConnection, race_id: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM gold.race_features WHERE race_id = ?", [race_id]).fetchone()[0])


def circuit_priors(con: duckdb.DuckDBPyConnection, circuit_id: str, season: int) -> dict:
    """What every race run at this circuit so far says about it."""
    races = con.execute(
        """
        SELECT r.race_id, r.season, r.date,
               AVG(lf.track_temp) AS track_temp, AVG(lf.air_temp) AS air_temp,
               MAX(lf.safety_car_active::INT) AS had_sc,
               MAX(COALESCE(lf.rainfall_flag, FALSE)::INT) AS had_rain,
               MAX(lf.lap_number) AS laps,
               MEDIAN(lf.field_avg_lap_time_seconds) AS pace_scale
        FROM silver.races r JOIN gold.lap_features lf ON lf.race_id = r.race_id
        WHERE r.circuit_id = ?
        GROUP BY r.race_id, r.season, r.date
        ORDER BY r.date
        """,
        [circuit_id],
    ).df()
    overtaking = con.execute(
        """
        SELECT AVG(ch.historical_overtaking_rate) FROM gold.circuit_history ch WHERE ch.circuit_id = ?
          AND ch.race_id = (SELECT race_id FROM silver.races WHERE circuit_id = ? ORDER BY date DESC LIMIT 1)
        """,
        [circuit_id, circuit_id],
    ).fetchone()[0]
    # The latest race's overtaking prior excludes that race itself; for a
    # future race include it by averaging the per-race figure directly.
    per_race_overtaking = con.execute(
        """
        WITH seq AS (
            SELECT rf.race_id, rf.current_position,
                   LAG(rf.current_position) OVER w AS prev, rf.is_pit_lap, LAG(rf.is_pit_lap) OVER w AS prev_pit,
                   rf.safety_car_active, rf.vsc_active, rf.red_flag_active
            FROM gold.lap_features rf JOIN silver.races r ON r.race_id = rf.race_id
            WHERE r.circuit_id = ? AND rf.lap_number > 2
            WINDOW w AS (PARTITION BY rf.race_id, rf.driver_id ORDER BY rf.lap_number)
        )
        SELECT AVG(ABS(current_position - prev)) FROM seq
        WHERE prev IS NOT NULL AND NOT is_pit_lap AND NOT prev_pit
          AND NOT safety_car_active AND NOT vsc_active AND NOT red_flag_active
        """,
        [circuit_id],
    ).fetchone()[0]

    # Retirements: this circuit in this era, else every circuit in this
    # era, else all history — gold.circuit_history's fallback order.
    dnf = con.execute(
        """
        WITH r AS (
            SELECT r.circuit_id, CASE WHEN r.season <= 2021 THEN 0 WHEN r.season <= 2025 THEN 1 ELSE 2 END AS era,
                   AVG(CASE WHEN res.status = 'Finished' OR res.status = 'Lapped' OR res.status LIKE '+%' THEN 0 ELSE 1 END) AS dnf
            FROM silver.races r JOIN bronze.ergast_results res ON res.season = r.season AND res.round = r.round
            WHERE res.status != 'Did not start'
            GROUP BY r.race_id, r.circuit_id, r.season
        )
        SELECT AVG(dnf) FILTER (WHERE circuit_id = ? AND era = ?), AVG(dnf) FILTER (WHERE era = ?), AVG(dnf) FROM r
        """,
        [circuit_id, _era(season), _era(season)],
    ).fetchone()
    offsets = con.execute(
        """
        SELECT AVG(track_temp - air_temp) FROM gold.lap_features
        WHERE track_temp IS NOT NULL AND air_temp IS NOT NULL
        """
    ).fetchone()[0]
    typical_laps = con.execute(
        "SELECT MEDIAN(laps) FROM (SELECT race_id, MAX(lap_number) AS laps FROM gold.lap_features GROUP BY race_id)"
    ).fetchone()[0]

    known = not races.empty
    return {
        "prior_races": int(len(races)),
        "track_temp": float(races.track_temp.mean()) if known else None,
        "air_temp": float(races.air_temp.mean()) if known else None,
        "track_minus_air": float((races.track_temp - races.air_temp).mean()) if known else float(offsets or DEFAULT_TRACK_MINUS_AIR),
        "sc_rate": float(races.had_sc.mean()) if known else None,
        "rain_share": float(races.had_rain.mean()) if known else None,
        "overtaking_rate": float(per_race_overtaking) if per_race_overtaking is not None else (float(overtaking) if overtaking else None),
        "dnf_rate": next((float(v) for v in dnf if v is not None), None),
        "total_laps": int(races.laps.iloc[-1]) if known else int(typical_laps or 57),
        "laps_known": known,
        "pace_scale": float(races.pace_scale.dropna().iloc[-1]) if known and races.pace_scale.notna().any() else None,
    }


@lru_cache(maxsize=64)
def _forecast(lat: float, lon: float, date: str, hour: int, _cache_hour: str) -> dict | None:
    """Open-Meteo's forecast at race start. `_cache_hour` makes the cache
    refresh hourly — a forecast is worth re-fetching, not keeping."""
    from ingestion.config import load_config
    from ingestion.http import get_json

    try:
        data = get_json(
            FORECAST_URL,
            load_config(),
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability",
                "start_date": date,
                "end_date": date,
                "timezone": "UTC",
            },
        )
    except Exception as exc:  # a forecast is a nicety; typical conditions stand in
        logger.warning("forecast unavailable: %s", exc)
        return None
    hourly = data.get("hourly") or {}
    temps, rain = hourly.get("temperature_2m") or [], hourly.get("precipitation_probability") or []
    if not temps:
        return None
    window = range(max(0, hour - 1), min(len(temps), hour + 2))
    t = [temps[i] for i in window if temps[i] is not None]
    p = [rain[i] for i in window if i < len(rain) and rain[i] is not None]
    if not t:
        return None
    return {"air_temp": sum(t) / len(t), "rain_probability": (max(p) / 100.0) if p else None}


@dataclass
class Conditions:
    track_temp: float
    air_temp: float
    rain_expected: bool
    rain_probability: float | None
    source: str  # "forecast", "typical", "override" — or "measured" for a completed race
    note: str


def race_conditions(entry: dict, priors: dict, track_temp: float | None = None, rain: bool | None = None) -> Conditions:
    date = pd.Timestamp(entry["date"]).date()
    hour = int(str(entry.get("time_utc") or "13:00:00")[:2])
    days_out = (date - dt.date.today()).days
    base_air = priors["air_temp"] if priors["air_temp"] is not None else 25.0
    air, rain_p, source = base_air, priors["rain_share"], "typical"
    note = (
        f"Typical for this circuit: the average of {priors['prior_races']} race(s) here."
        if priors["prior_races"]
        else "First race here in this data: grid-wide averages stand in for the circuit."
    )

    if 0 <= days_out <= FORECAST_HORIZON_DAYS:
        fc = _forecast(float(entry["latitude"]), float(entry["longitude"]), date.isoformat(), hour, dt.datetime.now().strftime("%Y%m%d%H"))
        if fc:
            air, rain_p, source = fc["air_temp"], fc["rain_probability"], "forecast"
            note = f"Open-Meteo forecast for race start ({days_out} day{'s' if days_out != 1 else ''} out); track temperature estimated from it."

    track = air + priors["track_minus_air"] if source == "forecast" or priors["track_temp"] is None else priors["track_temp"]
    rain_expected = bool(rain_p is not None and rain_p >= RAIN_PROBABILITY_THRESHOLD)

    if track_temp is not None or rain is not None:
        source = "override"
        note = "Set by hand — " + note[0].lower() + note[1:]
        track = track_temp if track_temp is not None else track
        rain_expected = rain if rain is not None else rain_expected
    return Conditions(round(track, 1), round(air, 1), rain_expected, rain_p, source, note)


def _latest_race_before(con: duckdb.DuckDBPyConnection, season: int, date) -> str | None:
    row = con.execute(
        "SELECT race_id FROM silver.races WHERE season = ? AND date < ? ORDER BY date DESC LIMIT 1", [season, date]
    ).fetchone()
    return row[0] if row else None


def expected_grid(con: duckdb.DuckDBPyConnection, season: int, before_date) -> dict[str, int]:
    """Each driver's average starting slot this season, ranked 1..N — the
    grid before qualifying has been run."""
    rows = con.execute(
        """
        SELECT res.driver_id, AVG(CASE WHEN res.grid > 0 THEN res.grid ELSE 20 END) AS g
        FROM bronze.ergast_results res JOIN silver.races r ON r.season = res.season AND r.round = res.round
        WHERE res.season = ? AND r.date < ?
        GROUP BY res.driver_id ORDER BY g
        """,
        [season, before_date],
    ).fetchall()
    return {driver: i + 1 for i, (driver, _) in enumerate(rows)}


def entry_list(con: duckdb.DuckDBPyConnection, season: int, before_date) -> pd.DataFrame:
    latest = _latest_race_before(con, season, before_date)
    if latest is None:
        return pd.DataFrame(columns=["driver_id", "constructor_id"])
    s, rnd = (int(x) for x in latest.split("_"))
    return con.execute(
        "SELECT driver_id, constructor_id FROM bronze.ergast_results WHERE season = ? AND round = ?", [s, rnd]
    ).df()


def qualifying_order(con: duckdb.DuckDBPyConnection, season: int, rnd: int) -> dict[str, int]:
    """The qualifying classification, once qualifying has run — Saturday
    night's grid, before any penalties (which aren't in this data)."""
    rows = con.execute(
        "SELECT driver_id, position FROM bronze.ergast_qualifying WHERE season = ? AND round = ? ORDER BY position",
        [season, rnd],
    ).fetchall()
    return {driver: int(pos) for driver, pos in rows if pos is not None}


def _grid_slots(order: dict[str, int], drivers: list[str], ours: str, our_slot: int) -> dict[str, int]:
    """Place everyone in expected-grid order around our driver's slot."""
    others = sorted((d for d in drivers if d != ours), key=lambda d: order.get(d, 99))
    slots, slot = {ours: our_slot}, 1
    for d in others:
        if slot == our_slot:
            slot += 1
        slots[d] = slot
        slot += 1
    return slots


def future_setup(race_id: str, driver_id: str, grid: int | None, track_temp: float | None, rain: bool | None):
    """Everything build_race_plan needs for an unrun race: the starting
    state, the field, the conditions and the circuit priors."""
    con = get_connection()
    try:
        entry = calendar_entry(con, race_id)
        if entry is None:
            raise ValueError(f"{race_id!r} isn't on the calendar")
        season, circuit_id, date = int(entry["season"]), entry["circuit_id"], entry["date"]
        priors = circuit_priors(con, circuit_id, season)
        conditions = race_conditions(entry, priors, track_temp, rain)
        entries = entry_list(con, season, date)
        # Qualifying run: the grid is known. Otherwise the season-average one.
        qualified = qualifying_order(con, season, int(entry["round"]))
        order = qualified or expected_grid(con, season, date)
        latest = _latest_race_before(con, season, date)
    finally:
        con.close()

    if driver_id not in set(entries.driver_id):
        raise ValueError(f"{driver_id!r} isn't on the {season} entry list")
    our_slot = grid or order.get(driver_id, len(entries))
    slots = _grid_slots(order, list(entries.driver_id), driver_id, our_slot)

    # Template: this driver's state at the start of their latest race —
    # team, CarProfile and driver form as they stand now — with the
    # circuit, weather and start-of-race facts replaced.
    df = load_race_features()
    rows = df[(df.race_id == latest) & (df.driver_id == driver_id)].sort_values("lap_number")
    rows = rows.dropna(subset=["stint_number", "tyre_age"])
    if rows.empty:
        raise ValueError(f"No recent race state for {driver_id!r}")
    template = RaceState.from_gold_row(rows.iloc[0], race_total_laps=priors["total_laps"])
    baseline = priors["track_temp"]
    state = replace(
        template,
        circuit_id=circuit_id,
        current_lap=0,
        race_total_laps=priors["total_laps"],
        compound=DEFAULT_PIT_START_COMPOUND,
        tyre_age=0.0,
        stint_number=1,
        current_position=float(our_slot),
        gap_to_leader=starting_gap_for_position(our_slot),
        field_avg_lap_time_seconds=priors["pace_scale"] or template.field_avg_lap_time_seconds,
        track_temp=conditions.track_temp,
        air_temp=conditions.air_temp,
        rainfall_flag=conditions.rain_expected,
        condition_delta=(conditions.track_temp - baseline) if baseline is not None else None,
        historical_sc_rate=priors["sc_rate"],
        historical_dnf_rate=priors["dnf_rate"],
        historical_overtaking_rate=priors["overtaking_rate"],
        circuit_baseline_track_temp=baseline,
        compounds_used_this_race=set(),
    )

    # Season form, sharpened by qualifying once it has run
    # (race_plan/weekend_pace.py).
    form = _season_form(season, str(date))
    quali = qualifying_gaps(race_id)
    rivals = [
        RivalTrend(
            driver_id=r.driver_id,
            team_id=r.constructor_id,
            gap_to_leader=starting_gap_for_position(slots[r.driver_id]),
            recent_pace_delta=weekend_pace(_form_or_zero(form, r.driver_id), quali.get(r.driver_id)),
            compound=DEFAULT_PIT_START_COMPOUND,
            tyre_age=0.0,
        )
        for r in entries.itertuples()
        if r.driver_id != driver_id
    ]
    anchor = weekend_pace(_form_or_zero(form, driver_id), quali.get(driver_id))
    entry = {**entry, "qualifying_run": bool(qualified)}
    return state, rivals, conditions, priors, anchor, entry
