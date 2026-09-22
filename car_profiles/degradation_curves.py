"""Team-specific tyre degradation curves — PRD Section 13.2's Tyre View.

Raw lap time against tyre age is the wrong curve: a car gets lighter by
~1.5 kg of fuel every lap and the track rubbers in, so lap times FALL over
a stint even as the tyre wears. gold.lap_features' `degradation_rate` is
that raw slope, which is why its per-compound averages come out strongly
negative — it measures fuel burn more than tyres.

The fuel and track-evolution effect is estimated from the data, per
season, with one regression:

    lap_time = a[race, driver] + b * lap_number + c[compound] * tyre_age + d[compound]

`a` absorbs everything fixed for a driver in a race (car, circuit,
weather). Within one race, lap_number and tyre_age move together during a
stint but come apart at every pit stop (the tyre resets, the lap count
doesn't), and that is what lets `b` (fuel + track) and `c` (tyre wear) be
told apart. Measured `b` sits at -0.049 to -0.056 s/lap for 2018-2025 —
the figure usually quoted for fuel effect — and -0.042 in 2026, whose
cars start with less fuel.

A team's curve is then its laps with `b * lap_number` removed, each stint
expressed relative to its own laps 2-4, and the median taken across
stints at each tyre age. Out-laps (tyre age 1), pit, safety-car, VSC,
yellow, red-flag and wet laps are excluded, as are laps over 107% of the
race's median green lap (traffic, damage, spins).

Regulation eras: everything is per season, never pooled across seasons.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.db import get_connection

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
BASELINE_TYRE_AGES = (2, 4)  # inclusive; a stint's reference pace
MAX_CURVE_TYRE_AGE = 40
# Fewer laps than this at a tyre age isn't a curve point. Deliberately
# high: only unusually long stints reach high tyre ages (survivorship), so
# the tail of a curve rests on a few atypical stints — Ferrari's 2026 hard
# curve dove to -1.5 s past 33 laps on five laps at a threshold of 5.
MIN_LAPS_PER_POINT = 15
OUTLIER_FACTOR = 1.07


@lru_cache(maxsize=1)
def _green_laps() -> pd.DataFrame:
    con = get_connection()
    try:
        df = con.execute(
            """
            SELECT lf.race_id, r.season, lf.driver_id, lf.team_id, lf.stint_number,
                   lf.lap_number, lf.tyre_age, lf.compound, lf.lap_time_seconds, lf.track_temp
            FROM gold.lap_features lf JOIN silver.races r ON r.race_id = lf.race_id
            WHERE NOT lf.is_pit_lap AND NOT lf.safety_car_active AND NOT lf.vsc_active
              AND NOT lf.red_flag_active AND NOT COALESCE(lf.yellow_active, FALSE)
              AND NOT COALESCE(lf.rainfall_flag, FALSE)
              AND lf.compound IN ('SOFT', 'MEDIUM', 'HARD')
              AND lf.lap_number > 1 AND lf.tyre_age >= 2
              AND lf.lap_time_seconds IS NOT NULL
            """
        ).df()
    finally:
        con.close()
    race_median = df.groupby("race_id")["lap_time_seconds"].transform("median")
    return df[df["lap_time_seconds"] < race_median * OUTLIER_FACTOR].reset_index(drop=True)


@lru_cache(maxsize=None)
def season_model(season: int) -> dict:
    """The regression in the module docstring, for one season: the fuel +
    track coefficient and the field-wide wear rate per compound."""
    df = _green_laps()
    g = df[df["season"] == season]
    if g.empty:
        raise ValueError(f"No green dry laps for season {season}")
    X = pd.DataFrame({"lap_number": g["lap_number"].astype(float)})
    for compound in DRY_COMPOUNDS:
        on = (g["compound"] == compound).astype(float)
        X[f"age_{compound}"] = g["tyre_age"] * on
        X[f"is_{compound}"] = on
    key = (g["race_id"] + "|" + g["driver_id"]).to_numpy()
    Xd = X - X.groupby(key).transform("mean")
    yd = g["lap_time_seconds"] - g["lap_time_seconds"].groupby(key).transform("mean")
    beta, *_ = np.linalg.lstsq(Xd.to_numpy(), yd.to_numpy(), rcond=None)
    coef = dict(zip(X.columns, beta))
    return {
        "fuel_track_seconds_per_lap": float(coef["lap_number"]),
        "field_wear_seconds_per_lap": {c: float(coef[f"age_{c}"]) for c in DRY_COMPOUNDS},
        "laps_used": int(len(g)),
    }


def _corrected(season: int) -> pd.DataFrame:
    df = _green_laps()
    g = df[df["season"] == season].copy()
    b = season_model(season)["fuel_track_seconds_per_lap"]
    g["corrected"] = g["lap_time_seconds"] - b * g["lap_number"]
    stint = ["race_id", "driver_id", "stint_number"]
    lo, hi = BASELINE_TYRE_AGES
    baseline = g[g["tyre_age"].between(lo, hi)].groupby(stint)["corrected"].mean().rename("baseline")
    g = g.join(baseline, on=stint).dropna(subset=["baseline"])
    g["delta"] = g["corrected"] - g["baseline"]
    return g


def _curve(frame: pd.DataFrame) -> list[dict]:
    points = []
    for age, grp in frame[frame["tyre_age"] <= MAX_CURVE_TYRE_AGE].groupby("tyre_age"):
        if len(grp) < MIN_LAPS_PER_POINT:
            continue
        points.append(
            {
                "tyre_age": int(age),
                "median_delta_s": round(float(grp["delta"].median()), 3),
                "p25_delta_s": round(float(grp["delta"].quantile(0.25)), 3),
                "p75_delta_s": round(float(grp["delta"].quantile(0.75)), 3),
                "laps": int(len(grp)),
            }
        )
    return points


def _slope(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """Wear rate in s/lap as a within-stint slope of the corrected lap time
    on tyre age, with a standard error, pooled over the given stints."""
    if len(frame) < 10:
        return None, None
    stint = frame["race_id"] + "|" + frame["driver_id"] + "|" + frame["stint_number"].astype(str)
    x = frame["tyre_age"] - frame["tyre_age"].groupby(stint).transform("mean")
    y = frame["corrected"] - frame["corrected"].groupby(stint).transform("mean")
    sxx = float((x * x).sum())
    if sxx == 0:
        return None, None
    slope = float((x * y).sum() / sxx)
    residual = y - slope * x
    dof = max(len(frame) - stint.nunique() - 1, 1)
    se = float(np.sqrt((residual * residual).sum() / dof / sxx))
    return slope, se


def team_tyre_report(team_id: str, season: int) -> dict:
    g = _corrected(season)
    team = g[g["team_id"] == team_id]
    if team.empty:
        raise LookupError(f"No green dry laps for {team_id!r} in {season}")

    model = season_model(season)
    compounds = []
    for compound in DRY_COMPOUNDS:
        t = team[team["compound"] == compound]
        f = g[g["compound"] == compound]
        slope, se = _slope(t)
        field_slope, _ = _slope(f)

        # Operating window: this team's wear rate in each race against that
        # race's track temperature. Per race, so a curve of points, not a
        # claimed optimum — the data is too thin for one.
        by_race = []
        for race_id, r in t.groupby("race_id"):
            s, _ = _slope(r)
            if s is not None:
                by_race.append(
                    {
                        "race_id": race_id,
                        "track_temp": round(float(r["track_temp"].mean()), 1),
                        "wear_s_per_lap": round(s, 4),
                        "laps": int(len(r)),
                    }
                )
        compounds.append(
            {
                "compound": compound,
                "stints": int(t.groupby(["race_id", "driver_id", "stint_number"]).ngroups),
                "wear_s_per_lap": None if slope is None else round(slope, 4),
                "wear_ci95": None if se is None else [round(slope - 1.96 * se, 4), round(slope + 1.96 * se, 4)],
                "field_wear_s_per_lap": None if field_slope is None else round(field_slope, 4),
                "curve": _curve(t),
                "field_curve": _curve(f),
                "by_race": by_race,
            }
        )
    return {
        "team_id": team_id,
        "season": season,
        "fuel_track_seconds_per_lap": round(model["fuel_track_seconds_per_lap"], 4),
        "compounds": compounds,
    }
