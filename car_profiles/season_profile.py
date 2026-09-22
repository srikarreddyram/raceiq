"""A team's CarProfile for the dashboard — PRD Section 13.2's Car Profile
View: inferred characteristics with confidence intervals, how they
settled over the season, and the tyre-temperature picture.

Built from gold.car_race_observations (one raw reading per race), not from
gold.car_profiles. The two answer different questions:

- gold.car_profiles holds what was knowable BEFORE each race, which is
  what a model predicting that race may use.
- This module describes the car as of now — every completed race of the
  season included — which is what an engineer looking at the car wants.

Confidence intervals: a profile value is the mean of a handful of per-race
readings, so its interval is a t-interval on that mean (few races, so t
rather than normal). The two correlation characteristics get a Fisher-z
interval instead. Both are wide early in a season, which is the point:
round 3's profile is a guess and should look like one.

The three raw per-compound degradation slopes that gold.car_profiles also
carries are not shown here. They're lap-time slopes that include fuel burn
(see car_profiles/degradation_curves.py) and would read as tyres getting
faster with age. The Tyre View shows fuel-corrected wear instead.
`degradation_vs_field` is kept: it's a team-minus-field difference within
each race, so the fuel effect common to every car cancels.

Layer 2 of PRD 8.1 (manually seeded engineering priors) doesn't exist in
this project — see car_profiles/inference/characteristics.py — and is
reported as absent rather than filled in.
"""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from car_profiles.inference.characteristics import MIN_RACES_FOR_CONFIDENCE

CHARACTERISTICS = [
    {
        "key": "tyre_warmup_rate",
        "label": "Tyre warm-up",
        "unit": "s",
        "kind": "mean",
        "better": "lower",
        "description": "Pace on laps 1-3 of a stint minus laps 4-10, relative to the field. Lower means the car gets tyres working sooner.",
    },
    {
        "key": "cold_tyre_pace_loss",
        "label": "Out-lap loss",
        "unit": "s",
        "kind": "mean",
        "better": "lower",
        "description": "The out-lap against the lap after it, relative to the field. Includes pit-exit time, so compare teams, not absolutes.",
    },
    {
        "key": "downforce_proxy",
        "label": "Downforce bias",
        "unit": "s",
        "kind": "mean",
        "better": None,
        "description": "Sector 2 pace against sectors 1+3, relative to the field. Negative means comparatively stronger in S2, assumed the high-downforce sector — an approximation, not true everywhere.",
    },
    {
        "key": "degradation_vs_field",
        "label": "Degradation vs field",
        "unit": "s/lap",
        "kind": "mean",
        "better": "lower",
        "description": "This car's in-stint lap-time slope minus the field's in the same race. Negative means kinder to its tyres than the average car.",
    },
    {
        "key": "safety_car_restart_pace",
        "label": "Restart pace",
        "unit": "s",
        "kind": "mean",
        "better": "lower",
        "description": "Pace on the first lap after a safety car, against the car's own green-flag average. Lower means quicker to switch tyres back on.",
    },
    {
        "key": "undercut_vulnerability",
        "label": "Pit-cycle loss",
        "unit": "places",
        "kind": "mean",
        "better": "lower",
        "description": "Positions lost from the lap of a stop to three laps later. A stand-in for PRD 8.1's undercut vulnerability, which needs per-rival pit timing this data lacks.",
    },
    {
        "key": "tyre_temp_sensitivity",
        "label": "Heat sensitivity",
        "unit": "r",
        "kind": "correlation",
        "better": None,
        "description": "Correlation across races between degradation vs field and track temperature. Positive means the car suffers more than rivals when it's hot.",
    },
    {
        "key": "aero_sensitivity",
        "label": "Wind sensitivity",
        "unit": "r",
        "kind": "correlation",
        "better": "lower",
        "description": "Correlation across races between lap-to-lap pace variability and wind speed. Positive means the car gets more erratic in wind.",
    },
]

_CORRELATION_PAIRS = {
    "tyre_temp_sensitivity": ("degradation_vs_field", "race_track_temp"),
    "aero_sensitivity": ("pace_variability", "race_wind_speed"),
}


def _finite(x) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _mean_estimate(values: pd.Series) -> dict:
    v = values.dropna().astype(float)
    n = len(v)
    if n == 0:
        return {"value": None, "ci95": None, "n": 0}
    mean = float(v.mean())
    if n < 2:
        return {"value": mean, "ci95": None, "n": n}
    half = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))
    return {"value": mean, "ci95": [mean - half, mean + half], "n": n}


def _correlation_estimate(x: pd.Series, y: pd.Series) -> dict:
    pair = pd.concat([x, y], axis=1).dropna().astype(float)
    n = len(pair)
    if n < 3 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
        return {"value": None, "ci95": None, "n": n}
    r = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])
    if n < 4 or abs(r) >= 1:
        return {"value": r, "ci95": None, "n": n}
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return {"value": r, "ci95": [math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)], "n": n}


def _estimate(obs: pd.DataFrame, key: str) -> dict:
    if key in _CORRELATION_PAIRS:
        a, b = _CORRELATION_PAIRS[key]
        return _correlation_estimate(obs[a], obs[b])
    return _mean_estimate(obs[key])


def _rounded(est: dict) -> dict:
    return {
        "value": None if est["value"] is None else round(est["value"], 4),
        "ci95": None if est["ci95"] is None else [round(c, 4) for c in est["ci95"]],
        "n": est["n"],
    }


def seasons_for(con: duckdb.DuckDBPyConnection, team_id: str) -> list[int]:
    return [
        int(r[0])
        for r in con.execute(
            "SELECT DISTINCT season FROM gold.car_race_observations WHERE team_id = ? ORDER BY season", [team_id]
        ).fetchall()
    ]


def season_profile(con: duckdb.DuckDBPyConnection, team_id: str, season: int) -> dict:
    obs_all = con.execute(
        """
        SELECT o.*, r.circuit_id, r.name AS race_name, ch.circuit_baseline_track_temp
        FROM gold.car_race_observations o
        JOIN silver.races r ON r.race_id = o.race_id
        LEFT JOIN gold.circuit_history ch ON ch.race_id = o.race_id
        WHERE o.season = ?
        ORDER BY o.date
        """,
        [season],
    ).df()
    obs = obs_all[obs_all["team_id"] == team_id].reset_index(drop=True)
    if obs.empty:
        raise LookupError(f"No races for {team_id!r} in {season}")

    teams = sorted(obs_all["team_id"].unique())
    characteristics = []
    for meta in CHARACTERISTICS:
        key = meta["key"]
        current = _estimate(obs, key)

        # Where this car sits against the rest of the grid, same season.
        by_team = {t: _estimate(obs_all[obs_all["team_id"] == t], key)["value"] for t in teams}
        valued = {t: v for t, v in by_team.items() if v is not None}
        rank = None
        if current["value"] is not None and meta["better"] is not None:
            ordered = sorted(valued, key=valued.get, reverse=meta["better"] == "higher")
            rank = ordered.index(team_id) + 1 if team_id in ordered else None

        # How the estimate settled race by race — the "confidence over the
        # season" PRD 13.2 asks for.
        trajectory = []
        for k in range(1, len(obs) + 1):
            est = _rounded(_estimate(obs.iloc[:k], key))
            trajectory.append({"race_id": obs.loc[k - 1, "race_id"], **est})

        characteristics.append(
            {
                **meta,
                **_rounded(current),
                "field_mean": None if not valued else round(float(np.mean(list(valued.values()))), 4),
                "field_min": None if not valued else round(min(valued.values()), 4),
                "field_max": None if not valued else round(max(valued.values()), 4),
                "rank": rank,
                "teams_ranked": len(valued),
                "trajectory": trajectory,
            }
        )

    tyre_window = [
        {
            "race_id": row.race_id,
            "race_name": row.race_name,
            "track_temp": _finite(row.race_track_temp),
            "circuit_baseline_track_temp": _finite(row.circuit_baseline_track_temp),
            "degradation_vs_field": _finite(row.degradation_vs_field),
        }
        for row in obs.itertuples()
    ]

    return {
        "team_id": team_id,
        "season": season,
        "races_observed": int(len(obs)),
        "last_race_id": obs["race_id"].iloc[-1],
        "min_races_for_confidence": MIN_RACES_FOR_CONFIDENCE,
        "characteristics": characteristics,
        "tyre_window": tyre_window,
        "seeded_priors": None,
        "seeded_priors_note": (
            "PRD 8.1's manually seeded priors (downforce philosophy, tyre operating window, setup "
            "sensitivity, known weaknesses) come from technical journalism this project doesn't "
            "ingest. They're absent rather than invented."
        ),
    }


def confidence_over_season(con: duckdb.DuckDBPyConnection, season: int) -> list[dict]:
    """How sure the CarProfiles are, race by race, across the whole grid —
    PRD 13.2's "car profile inference confidence over the season".

    For every team and every mean-type characteristic, the 95% interval's
    half-width after k races, divided by how far apart the grid's teams
    ended up on that characteristic — so 1.0 means "the uncertainty is as
    wide as the whole grid's spread" and the number is comparable across
    characteristics with different units. Summarised per k as a median
    and interquartile band, plus the share of (team, characteristic)
    pairs whose interval already excludes the grid mean.
    """
    obs = con.execute(
        "SELECT * FROM gold.car_race_observations WHERE season = ? ORDER BY date", [season]
    ).df()
    if obs.empty:
        return []
    keys = [c["key"] for c in CHARACTERISTICS if c["kind"] == "mean"]
    spread = {}
    for key in keys:
        team_means = obs.groupby("team_id")[key].mean().dropna()
        spread[key] = float(team_means.max() - team_means.min()) if len(team_means) > 1 else float("nan")

    obs = obs.assign(k=obs.groupby("team_id").cumcount() + 1)
    rows = []
    for k in range(1, int(obs["k"].max()) + 1):
        relative, distinct, total = [], 0, 0
        for key in keys:
            estimates = {}
            for team, grp in obs[obs["k"] <= k].groupby("team_id"):
                if grp["k"].max() < k:
                    continue  # a team with fewer than k races isn't at step k yet
                estimates[team] = _mean_estimate(grp[key])
            values = [e["value"] for e in estimates.values() if e["value"] is not None]
            if not values or not math.isfinite(spread[key]) or spread[key] == 0:
                continue
            grid_mean = float(np.mean(values))
            for e in estimates.values():
                if e["ci95"] is None:
                    continue
                total += 1
                relative.append((e["ci95"][1] - e["ci95"][0]) / 2 / spread[key])
                distinct += e["ci95"][0] > grid_mean or e["ci95"][1] < grid_mean
        if relative:
            rows.append(
                {
                    "races": k,
                    "median_relative_halfwidth": round(float(np.median(relative)), 4),
                    "p25_relative_halfwidth": round(float(np.quantile(relative, 0.25)), 4),
                    "p75_relative_halfwidth": round(float(np.quantile(relative, 0.75)), 4),
                    "distinct_share": round(distinct / total, 4),
                    "pairs": total,
                }
            )
    return rows
