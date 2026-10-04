"""Practice long runs: how fast each car is on race-like runs, and how fast
its tyres go off — per driver, per team, per compound.

On Friday the teams run their race simulations: five, ten, fifteen laps
on one set of tyres, at race fuel. Those runs are the weekend's first look
at race pace and at how this track treats each car's tyres, before
qualifying says anything.

Push laps
---------
A long run is MIN_RUN_LAPS or more consecutive laps on one set of tyres
that are timed, accurately timed, not in or out of the pits, and set under
green flags. Its push laps are the ones within PUSH_LAP_BAND of the run's
median lap time, which drops a lap lost in traffic, a cool-down lap and
the odd quick first lap on fresh rubber. Checked against a published FP2
long-run analysis of the 2026 race at Sepang: it reproduces the lap counts
and averages (Antonelli 11 laps, 104.48 s; Leclerc 7, 104.800; Stroll 5,
106.953).

Degradation
-----------
Within a run tyre age and fuel load change together, one lap at a time,
so practice alone can't tell tyre wear from fuel burn. The season's
fuel-and-track effect, measured from races where pit stops separate the
two (car_profiles.degradation_curves.season_model), is taken out: a
car's degradation on a compound is the slope of its push laps against
tyre age, minus that effect. Per team, both drivers' push laps together.

What feeds the strategy is decided by evidence, not by how persuasive a
long-run chart looks — see USE_IN_STRATEGY below and
race_plan/long_runs_validation.py.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.db import get_connection

PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")
MIN_RUN_LAPS = 5
PUSH_LAP_BAND = 0.015  # within 1.5% of the run's median
MIN_DEGRADATION_LAPS = 6  # fewer push laps on a compound than this and no slope is fitted
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _ids(race_id: str) -> tuple[int, int]:
    season, rnd = race_id.split("_")
    return int(season), int(rnd)


@lru_cache(maxsize=1)
def _team_map() -> dict[str, str]:
    """FastF1 team names to constructor ids, through the same matching
    silver.laps uses — on a read-only connection: this runs inside the
    API, and a writable one would lock every other reader out."""
    from pipelines.silver.id_mappings import build_team_id_map

    con = get_connection()
    try:
        return build_team_id_map(con)
    finally:
        con.close()


@lru_cache(maxsize=16)
def team_names(season: int) -> dict[str, str]:
    """Constructor id to its name that season, without the "F1 Team" suffix."""
    con = get_connection()
    try:
        rows = con.execute(
            "SELECT DISTINCT constructor_id, constructor_name FROM bronze.ergast_constructor_standings WHERE season <= ? ORDER BY season",
            [season],
        ).fetchall()
    finally:
        con.close()
    return {cid: name.removesuffix(" F1 Team") for cid, name in rows}


@lru_cache(maxsize=64)
def practice_laps(race_id: str) -> pd.DataFrame:
    """Every practice lap of the weekend, with the run it belongs to and
    whether it's a push lap. Empty if no practice has been ingested."""
    season, rnd = _ids(race_id)
    con = get_connection()
    try:
        laps = con.execute(
            """
            SELECT session_type AS session, driver AS code, team, lap_number, stint_number, compound,
                   tyre_age_laps AS tyre_age, lap_time_seconds AS lap_time, is_pit_lap, is_accurate, track_status
            FROM bronze.fastf1_laps
            WHERE season = ? AND round = ? AND session_type IN ('FP1', 'FP2', 'FP3')
            ORDER BY session_type, driver, lap_number
            """,
            [season, rnd],
        ).df()
        names = con.execute(
            """
            SELECT DISTINCT driver_code AS code, driver_id FROM bronze.ergast_results WHERE season = ?
            """,
            [season],
        ).df()
    finally:
        con.close()
    if laps.empty:
        return laps

    team_map = _team_map()
    laps = laps.merge(names, on="code", how="left")
    laps["driver_id"] = laps["driver_id"].fillna(laps["code"].str.lower())
    laps["team_id"] = laps["team"].map(team_map).fillna(laps["team"])

    clean = (
        laps["lap_time"].notna()
        & ~laps["is_pit_lap"].astype(bool)
        & laps["is_accurate"].fillna(False).astype(bool)
        & (laps["track_status"].astype(str) == "1")
    )
    car = [laps["session"], laps["code"]]
    # A run breaks at any unclean lap, a new set of tyres, or a gap in lap numbers.
    breaks = (~clean) | (laps.groupby(car)["stint_number"].diff() != 0) | (laps.groupby(car)["lap_number"].diff() != 1)
    laps["run"] = breaks.groupby(car).cumsum()
    laps["clean"] = clean
    run_key = ["session", "code", "run"]
    in_clean = laps[clean]
    run_size = in_clean.groupby(run_key)["lap_time"].transform("size")
    run_median = in_clean.groupby(run_key)["lap_time"].transform("median")
    long = run_size >= MIN_RUN_LAPS
    push = long & ((in_clean["lap_time"] - run_median).abs() <= PUSH_LAP_BAND * run_median)
    laps["long_run"] = False
    laps["push"] = False
    laps.loc[in_clean.index, "long_run"] = long
    laps.loc[in_clean.index, "push"] = push
    return laps


def _fuel_effect(season: int) -> float:
    """Seconds per lap of running that fuel burn and track rubbering take
    off a lap time — negative. Nearest earlier season with data if this
    one has none."""
    from car_profiles.degradation_curves import season_model

    for s in range(season, 2017, -1):
        try:
            return float(season_model(s)["fuel_track_seconds_per_lap"])
        except ValueError:
            continue
    return -0.05


def _slope(age: pd.Series, lap_time: pd.Series, run: pd.Series) -> float | None:
    """Slope of lap time on tyre age within runs (each run centred on its
    own mean), so two runs at different fuel loads don't fake a slope."""
    if len(age) < MIN_DEGRADATION_LAPS:
        return None
    a = age - age.groupby(run).transform("mean")
    t = lap_time - lap_time.groupby(run).transform("mean")
    denominator = float((a * a).sum())
    if denominator <= 0:
        return None
    return float((a * t).sum() / denominator)


def summary(race_id: str, session: str | None = None) -> pd.DataFrame:
    """One row per driver: push-lap pace, spread and count, the gap to the
    quickest, compounds run, and degradation per compound (s/lap per lap
    of tyre age, fuel effect removed)."""
    laps = practice_laps(race_id)
    if laps.empty:
        return pd.DataFrame()
    if session is not None:
        laps = laps[laps["session"] == session]
    push = laps[laps["push"]].copy()
    if push.empty:
        return pd.DataFrame()
    push["run_id"] = push["session"] + "|" + push["run"].astype(str)
    fuel = _fuel_effect(_ids(race_id)[0])

    rows = []
    for (driver_id, code), g in push.groupby(["driver_id", "code"]):
        row = {
            "driver_id": driver_id,
            "code": code,
            "team_id": g["team_id"].mode().iloc[0],
            "mean_lap": float(g["lap_time"].mean()),
            "sd": float(g["lap_time"].std(ddof=1)) if len(g) > 1 else None,
            "push_laps": int(len(g)),
            "compounds": sorted(g["compound"].dropna().unique().tolist()),
        }
        for compound in DRY_COMPOUNDS:
            on = g[g["compound"] == compound]
            raw = _slope(on["tyre_age"], on["lap_time"], on["run_id"])
            row[f"deg_{compound.lower()}"] = None if raw is None else raw - fuel
            row[f"laps_{compound.lower()}"] = int(len(on))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_lap").reset_index(drop=True)
    out["gap"] = out["mean_lap"] - out["mean_lap"].iloc[0]
    return out


def team_summary(race_id: str, session: str | None = None) -> pd.DataFrame:
    """Per team: both drivers' push laps together — pace, and degradation
    per compound."""
    laps = practice_laps(race_id)
    if laps.empty:
        return pd.DataFrame()
    if session is not None:
        laps = laps[laps["session"] == session]
    push = laps[laps["push"]].copy()
    if push.empty:
        return pd.DataFrame()
    push["run_id"] = push["session"] + "|" + push["code"] + "|" + push["run"].astype(str)
    fuel = _fuel_effect(_ids(race_id)[0])
    rows = []
    for team_id, g in push.groupby("team_id"):
        row = {"team_id": team_id, "mean_lap": float(g["lap_time"].mean()), "push_laps": int(len(g)),
               "drivers": sorted(g["code"].unique().tolist())}
        for compound in DRY_COMPOUNDS:
            on = g[g["compound"] == compound]
            raw = _slope(on["tyre_age"], on["lap_time"], on["run_id"])
            row[f"deg_{compound.lower()}"] = None if raw is None else raw - fuel
            row[f"laps_{compound.lower()}"] = int(len(on))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_lap").reset_index(drop=True)
    out["gap"] = out["mean_lap"] - out["mean_lap"].iloc[0]
    return out


def field_degradation(race_id: str) -> dict[str, float | None]:
    """The whole field's degradation per compound this weekend, every
    car's push laps together (runs centred per car), fuel effect removed."""
    laps = practice_laps(race_id)
    if laps.empty:
        return {c: None for c in DRY_COMPOUNDS}
    push = laps[laps["push"]].copy()
    push["run_id"] = push["session"] + "|" + push["code"] + "|" + push["run"].astype(str)
    fuel = _fuel_effect(_ids(race_id)[0])
    out = {}
    for compound in DRY_COMPOUNDS:
        on = push[push["compound"] == compound]
        raw = _slope(on["tyre_age"], on["lap_time"], on["run_id"])
        out[compound] = None if raw is None else raw - fuel
    return out
