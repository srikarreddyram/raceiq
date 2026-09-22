"""Where each car is fast — straight-line speed against overall pace.

Nobody publishes power figures, and top speed alone isn't power: it's
power against drag. What the data can say honestly is where a car makes
and loses its time, by putting two measurements side by side:

  straight-line speed  the OpenF1 speed-trap reading (km/h) on green,
                       non-pit racing laps, relative to the field median
                       in the same race, so circuits cancel out
  overall race pace    the car's median green-lap pace against the field
                       (s/lap, negative is faster), from gold.lap_features

A car quick in a straight line but slow over a lap is losing its time in
the corners (short on downforce, or trimmed for low drag); one slow on
the straights but quick over a lap is winning the corners (a
high-downforce, draggier car). `reading` puts that into words from the
numbers, and names the circuit where the straight-line gap was largest.

Coverage is 2023 onwards — the seasons OpenF1 has speed-trap data for.
2026 cars run active aerodynamics and a near-50/50 combustion/electric
power split, with no DRS, so their speed-trap differences reflect energy
deployment as much as drag; the reading says so for that season.
"""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# Differences smaller than these are within race-to-race noise and are
# described as "level with the field".
SPEED_NOISE_KPH = 1.5
PACE_NOISE_S = 0.08

_SQL = """
WITH sess AS (
    SELECT session_key, CAST(MIN(date_start) AS DATE) AS d FROM bronze.openf1_laps GROUP BY session_key
),
race AS (
    SELECT s.session_key, r.race_id, r.season, r.round, r.circuit_id
    FROM sess s JOIN silver.races r ON ABS(DATE_DIFF('day', CAST(r.date AS DATE), s.d)) <= 1
    WHERE r.season = ?
),
entry AS (
    SELECT DISTINCT f.season, f.round, CAST(f.driver_number AS INT) AS num, e.driver_id, e.constructor_id
    FROM bronze.fastf1_laps f
    JOIN bronze.ergast_results e ON e.season = f.season AND e.round = f.round AND e.driver_code = f.driver
    WHERE f.session_type = 'R'
),
laps AS (
    SELECT race.race_id, race.circuit_id, en.constructor_id AS team_id, o.st_speed, lf.pace_delta_this_lap
    FROM bronze.openf1_laps o
    JOIN race ON race.session_key = o.session_key
    JOIN entry en ON en.season = race.season AND en.round = race.round AND en.num = o.driver_number
    JOIN gold.lap_features lf
      ON lf.race_id = race.race_id AND lf.driver_id = en.driver_id AND lf.lap_number = o.lap_number
    WHERE o.st_speed IS NOT NULL AND NOT lf.is_pit_lap AND NOT lf.safety_car_active
      AND NOT lf.vsc_active AND NOT lf.red_flag_active AND NOT COALESCE(lf.yellow_active, FALSE)
      AND lf.lap_number > 1
)
SELECT race_id, circuit_id, team_id,
       MEDIAN(st_speed) AS trap_kph,
       MEDIAN(pace_delta_this_lap) AS pace_s,
       COUNT(*) AS laps
FROM laps
GROUP BY race_id, circuit_id, team_id
HAVING COUNT(*) >= 10
"""


def _ci(values: pd.Series) -> list[float] | None:
    v = values.dropna()
    if len(v) < 2:
        return None
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))
    return [float(v.mean()) - half, float(v.mean()) + half]


def _reading(name: str, speed: float, pace: float, best_circuit: str | None, best_delta: float | None, season: int) -> str:
    fast_straight = speed > SPEED_NOISE_KPH
    slow_straight = speed < -SPEED_NOISE_KPH
    quick = pace < -PACE_NOISE_S
    slow = pace > PACE_NOISE_S
    sp = f"{abs(speed):.1f} km/h {'quicker' if speed > 0 else 'slower'} than the field through the speed trap"
    pc = f"{abs(pace):.2f} s/lap {'faster' if pace < 0 else 'slower'} over a lap"
    if fast_straight and quick:
        text = f"{name} is {sp} and {pc}: quick everywhere — strong power and an efficient car, not a trade-off."
    elif fast_straight and slow:
        text = f"{name} is {sp} but {pc}: the time is going in the corners — a car short on downforce, or trimmed for low drag."
    elif slow_straight and quick:
        text = f"{name} is {sp} yet {pc}: it's winning the corners — a high-downforce car paying for it in drag."
    elif slow_straight and slow:
        text = f"{name} is {sp} and {pc}: down on both, straights and corners alike."
    elif quick:
        text = f"{name} is level with the field on the straights and {pc}: the advantage is in the corners."
    elif slow:
        text = f"{name} is level with the field on the straights and {pc}: it's losing time in the corners."
    elif fast_straight:
        text = f"{name} is {sp} and level on overall pace: straight-line speed traded for cornering."
    elif slow_straight:
        text = f"{name} is {sp} and level on overall pace: cornering bought with drag."
    else:
        text = f"{name} is level with the field on the straights and over a lap."
    if best_circuit and best_delta is not None and abs(best_delta) >= SPEED_NOISE_KPH:
        text += f" Largest straight-line gap: {best_circuit} ({best_delta:+.1f} km/h)."
    if season >= 2026:
        text += (
            " 2026 cars have active aero, a near-50/50 electric power split and no DRS, so speed-trap"
            " gaps this season reflect energy deployment as much as drag."
        )
    return text


def speed_profile(con: duckdb.DuckDBPyConnection, season: int) -> dict:
    df = con.execute(_SQL, [season]).df()
    if df.empty:
        raise LookupError(f"No speed-trap data for {season} (OpenF1 covers 2023 onwards)")
    df["trap_delta"] = df["trap_kph"] - df.groupby("race_id")["trap_kph"].transform("median")
    df["pace_delta"] = df["pace_s"] - df.groupby("race_id")["pace_s"].transform("median")

    names = con.execute("SELECT constructor_id, name FROM silver.constructors WHERE season = ?", [season]).df()
    names = dict(zip(names.constructor_id, names.name))
    circuits = con.execute("SELECT circuit_id, name FROM silver.circuits").df()
    circuits = dict(zip(circuits.circuit_id, circuits.name))

    teams = []
    for team_id, g in df.groupby("team_id"):
        g = g.sort_values("race_id", key=lambda s: s.str.split("_").str[1].astype(int))
        speed, pace = float(g.trap_delta.mean()), float(g.pace_delta.mean())
        best = g.loc[g.trap_delta.abs().idxmax()]
        name = names.get(team_id, team_id)
        teams.append(
            {
                "team_id": team_id,
                "name": name,
                "races": int(len(g)),
                "top_speed_delta_kph": round(speed, 2),
                "top_speed_ci95": None if (ci := _ci(g.trap_delta)) is None else [round(c, 2) for c in ci],
                "median_trap_kph": round(float(g.trap_kph.median()), 1),
                "pace_delta_s": round(pace, 3),
                "pace_ci95": None if (ci := _ci(g.pace_delta)) is None else [round(c, 3) for c in ci],
                "reading": _reading(
                    name, speed, pace, circuits.get(best.circuit_id, best.circuit_id), float(best.trap_delta), season
                ),
                "by_race": [
                    {
                        "race_id": r.race_id,
                        "circuit_name": circuits.get(r.circuit_id, r.circuit_id),
                        "trap_kph": round(float(r.trap_kph), 1),
                        "trap_delta_kph": round(float(r.trap_delta), 2),
                        "pace_delta_s": round(float(r.pace_delta), 3),
                    }
                    for r in g.itertuples()
                ],
            }
        )
    teams.sort(key=lambda t: t["pace_delta_s"])
    return {"season": season, "races": int(df.race_id.nunique()), "teams": teams}
