"""A driver's profile for the dashboard — PRD Section 13.2's Driver View:
per-driver performance profile, circuit history, wet/dry splits.

The pace measure throughout is the gap to the TEAMMATE on the same lap:
driver's lap time minus teammate's, on laps where both were on green-flag
racing laps. It's the one comparison that holds the car constant, which
matters more than usual here — across 2018-2026 a driver's raw pace mostly
tracks which car they were in and which regulation era it was.
Field-relative pace is shown beside it, labelled as car-and-driver.

Wet laps are laps on INTERMEDIATE or WET tyres, not laps where the weather
station reported rain: rainfall_flag is set on thousands of slick-tyre laps
(a shower at the station, a damp patch), and "wet-weather skill" means
driving on a wet track, which the tyre choice records directly.

Wet/dry splits are given per regulation era (2018-21, 2022-25, 2026+):
a wet-weather gap measured against a 2019 teammate in a 2019 car isn't
the same quantity as one measured in 2026.
"""

from __future__ import annotations

import math

import duckdb
import pandas as pd

from models.common.data import is_classified

WET_COMPOUNDS = ("INTERMEDIATE", "WET")

# Ergast's `position` is finishing ORDER, populated for retirements too
# (a lap-3 retirement reads P22); whether a driver was classified comes from
# `status` — models/common/data.py's is_classified. Finishing averages use
# classified results only; head-to-head uses order, where a retirement
# correctly counts as finishing behind.
OUTLIER_FACTOR = 1.07  # laps beyond 107% of the race's median green lap: traffic, damage, spins
MIN_WET_LAPS = 10  # below this a wet split is anecdote, reported as null


def era_of(season: int) -> str:
    if season >= 2026:
        return "2026+"
    if season >= 2022:
        return "2022-25"
    return "2018-21"


# One row per lap where the driver and a teammate both raced a clean green
# lap ON THE SAME KIND OF TYRE. `gap` < 0 means the driver was faster.
# Crossover laps — one car on slicks, the other on intermediates — are
# dropped: that gap measures a tyre call, not driving, and classifying the
# lap by the driver's own tyre put the same lap in one teammate's dry split
# and the other's wet split (tests/test_driver_profiles.py caught it).
_TEAMMATE_LAPS_SQL = f"""
WITH green AS (
    SELECT lf.race_id, r.season, lf.driver_id, lf.team_id, lf.lap_number, lf.compound,
           lf.lap_time_seconds, lf.pace_delta_this_lap,
           lf.compound IN {WET_COMPOUNDS} AS is_wet
    FROM gold.lap_features lf JOIN silver.races r ON r.race_id = lf.race_id
    WHERE NOT lf.is_pit_lap AND NOT lf.safety_car_active AND NOT lf.vsc_active
      AND NOT lf.red_flag_active AND NOT COALESCE(lf.yellow_active, FALSE)
      AND lf.lap_number > 1 AND lf.lap_time_seconds IS NOT NULL
),
clean AS (
    SELECT g.*
    FROM green g
    JOIN (SELECT race_id, MEDIAN(lap_time_seconds) AS med FROM green GROUP BY race_id) m USING (race_id)
    WHERE g.lap_time_seconds < m.med * {OUTLIER_FACTOR}
)
SELECT d.race_id, d.season, d.lap_number, d.is_wet, t.driver_id AS teammate_id,
       d.lap_time_seconds - t.lap_time_seconds AS gap,
       d.pace_delta_this_lap
FROM clean d
JOIN clean t
  ON t.race_id = d.race_id AND t.team_id = d.team_id AND t.lap_number = d.lap_number
 AND t.driver_id <> d.driver_id
 AND t.is_wet = d.is_wet
WHERE d.driver_id = ?
"""


def _f(x, digits: int = 3) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, digits)


def _i(x) -> int | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x)


def driver_profile(con: duckdb.DuckDBPyConnection, driver_id: str, season: int) -> dict:
    results = con.execute(
        """
        SELECT er.season, er.round, er.season || '_' || er.round AS race_id, r.name AS race_name,
               r.circuit_id, c.name AS circuit_name, er.constructor_id AS team_id,
               er.grid, er.position, er.points, er.status,
               er.driver_given_name, er.driver_family_name, er.driver_code, er.driver_nationality
        FROM bronze.ergast_results er
        LEFT JOIN silver.races r ON r.race_id = er.season || '_' || er.round
        LEFT JOIN silver.circuits c ON c.circuit_id = r.circuit_id
        WHERE er.driver_id = ?
        ORDER BY er.season, er.round
        """,
        [driver_id],
    ).df()
    if results.empty:
        raise LookupError(f"No results for driver {driver_id!r}")
    if season not in set(results["season"]):
        raise LookupError(f"{driver_id!r} has no results in {season}")
    results["classified"] = results["status"].map(is_classified)
    results["finish"] = results["position"].where(results["classified"])

    laps = con.execute(_TEAMMATE_LAPS_SQL, [driver_id]).df()

    # Teammate's finishing position per race, for head-to-head.
    teammates = con.execute(
        """
        SELECT a.season || '_' || a.round AS race_id, b.driver_id AS teammate_id, b.position AS teammate_position
        FROM bronze.ergast_results a
        JOIN bronze.ergast_results b
          ON b.season = a.season AND b.round = a.round AND b.constructor_id = a.constructor_id
         AND b.driver_id <> a.driver_id
        WHERE a.driver_id = ?
        """,
        [driver_id],
    ).df()

    dry = laps[~laps["is_wet"]]
    race_gap = dry.groupby("race_id")["gap"].median() if not dry.empty else pd.Series(dtype=float)
    race_gap_laps = dry.groupby("race_id").size() if not dry.empty else pd.Series(dtype=int)

    latest = results.iloc[-1]
    this = results[results["season"] == season].merge(teammates, on="race_id", how="left")
    this = this.drop_duplicates("race_id")

    race_rows = []
    for r in this.itertuples():
        race_rows.append(
            {
                "race_id": r.race_id,
                "race_name": r.race_name,
                "circuit_id": r.circuit_id,
                "team_id": r.team_id,
                "grid": _i(r.grid) or None,  # Ergast grid 0 = pit-lane start
                "position": _i(r.position),
                "classified": bool(r.classified),
                "status": r.status,
                "points": _f(r.points, 1),
                "teammate_id": r.teammate_id if isinstance(r.teammate_id, str) else None,
                "teammate_position": _i(r.teammate_position),
                "teammate_gap_s": _f(race_gap.get(r.race_id)),
                "teammate_gap_laps": _i(race_gap_laps.get(r.race_id)),
            }
        )

    both = this.dropna(subset=["position", "teammate_position"])
    season_gaps = race_gap[race_gap.index.isin(this["race_id"])]
    summary = {
        "races": int(len(this)),
        "points": _f(this["points"].sum(), 1),
        "wins": int((this["position"] == 1).sum()),
        "podiums": int((this["position"] <= 3).sum()),
        "not_classified": int((~this["classified"]).sum()),
        "avg_grid": _f(this.loc[this["grid"] > 0, "grid"].mean(), 1),
        "avg_finish": _f(this["finish"].mean(), 1),
        "ahead_of_teammate": int((both["position"] < both["teammate_position"]).sum()),
        "head_to_head_races": int(len(both)),
        "median_teammate_gap_s": _f(season_gaps.median()),
    }

    # Career, one row per season — form across teams and eras.
    career = []
    for s, grp in results.groupby("season"):
        gaps = race_gap[race_gap.index.isin(grp["race_id"])]
        career.append(
            {
                "season": int(s),
                "era": era_of(int(s)),
                "teams": sorted(grp["team_id"].dropna().unique().tolist()),
                "races": int(len(grp)),
                "points": _f(grp["points"].sum(), 1),
                "wins": int((grp["position"] == 1).sum()),
                "podiums": int((grp["position"] <= 3).sum()),
                "avg_finish": _f(grp["finish"].mean(), 1),
                "not_classified": int((~grp["classified"]).sum()),
                "median_teammate_gap_s": _f(gaps.median()),
            }
        )

    # Circuit history, every season.
    circuits = []
    for cid, grp in results.dropna(subset=["circuit_id"]).groupby("circuit_id"):
        gaps = race_gap[race_gap.index.isin(grp["race_id"])]
        last = grp.iloc[-1]
        circuits.append(
            {
                "circuit_id": cid,
                "circuit_name": grp["circuit_name"].iloc[0],
                "races": int(len(grp)),
                "best_finish": _i(grp["finish"].min()),
                "avg_finish": _f(grp["finish"].mean(), 1),
                "wins": int((grp["position"] == 1).sum()),
                "last_race_id": last["race_id"],
                "last_position": _i(last["finish"]),
                "last_status": last["status"],
                "median_teammate_gap_s": _f(gaps.median()),
            }
        )
    circuits.sort(key=lambda c: (-c["races"], c["circuit_name"] or ""))

    # Wet/dry, per regulation era.
    splits = []
    if not laps.empty:
        laps = laps.assign(era=laps["season"].map(era_of))
        for era, grp in laps.groupby("era"):
            wet, dry_e = grp[grp["is_wet"]], grp[~grp["is_wet"]]
            enough = len(wet) >= MIN_WET_LAPS
            splits.append(
                {
                    "era": era,
                    "dry_laps": int(len(dry_e)),
                    "wet_laps": int(len(wet)),
                    "wet_races": int(wet["race_id"].nunique()),
                    "dry_teammate_gap_s": _f(dry_e["gap"].median()),
                    "wet_teammate_gap_s": _f(wet["gap"].median()) if enough else None,
                    "dry_field_delta_s": _f(dry_e["pace_delta_this_lap"].median()),
                    "wet_field_delta_s": _f(wet["pace_delta_this_lap"].median()) if enough else None,
                }
            )

    return {
        "driver_id": driver_id,
        "code": latest["driver_code"] if isinstance(latest["driver_code"], str) else None,
        "given_name": latest["driver_given_name"],
        "family_name": latest["driver_family_name"],
        "nationality": latest["driver_nationality"],
        "season": season,
        "seasons": sorted(int(s) for s in results["season"].unique()),
        "summary": summary,
        "races": race_rows,
        "career": career,
        "circuits": circuits,
        "wet_dry": splits,
        "min_wet_laps": MIN_WET_LAPS,
    }
