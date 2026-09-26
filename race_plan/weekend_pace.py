"""How fast each car will be in THIS race — season form, sharpened by
qualifying once it has run.

The pre-race field used to be projected from season form alone: each
driver's average green-flag pace delta over the season's earlier races.
Season form is the stable thing, but it knows nothing about this weekend.
A car that qualifies well off what its form predicts usually races off it
too — the track doesn't suit it, an upgrade didn't work, a setup went the
wrong way. Williams at Hungary 2025: form +0.05 s/lap (a top-half car on
the season), qualifying P19 and 0.63 s off the median, race pace +0.42.
Projected from form, Albon cut through the eight slower-on-paper cars
ahead of him to P11.4 in the simulation; he finished P15.

Measured (`fit()`), race pace per driver-race against season form and the
qualifying gap (best Q time minus the session median, clipped at
+/-QUALI_CLIP_SECONDS — crashes and wet laps put some "gaps" at 40 s):

                           2025 test: MAE   median |err|   corr
    form alone (x0.93)               0.322      0.270      0.787
    0.65 form + 0.44 quali           0.293      0.224      0.829

Fitted on 2018-2024 (no intercept: both sides are deltas to the field).
The same fit from 2018-20 scored on 2021 has the same shape (0.62 /
0.44), so it doesn't lean on one regulation era. Form alone is shrunk
too: a season average overstates how far apart two cars are on one
Sunday. Before qualifying — an unrun race planned early in the week —
only that term exists.

The simulation's pre-race pace-error table
(monte_carlo.PACE_ERROR_PERCENTILES_PRE_RACE) was measured against form
alone and is kept: this estimator's errors over the same seasons span
the same range (1st-99th percentile -1.13 to 1.27 s/lap, against -1.15
to 1.31). Its gain is in the middle of the distribution, on 2025.

Usage (re-fit and print the coefficients and error spread):
    uv run python -m race_plan.weekend_pace
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.db import get_connection

FORM_WEIGHT = 0.651
QUALI_WEIGHT = 0.437
QUALI_CLIP_SECONDS = 1.0
FORM_WEIGHT_WITHOUT_QUALI = 0.934


def _lap_seconds(value) -> float:
    if not isinstance(value, str) or ":" not in value:
        return np.nan
    minutes, seconds = value.split(":")
    try:
        return 60 * float(minutes) + float(seconds)
    except ValueError:
        return np.nan


@lru_cache(maxsize=None)
def qualifying_gaps(race_id: str) -> dict[str, float]:
    """Each driver's best qualifying lap minus the session median, in
    seconds, clipped. Empty when qualifying hasn't run (or isn't ingested)."""
    season, rnd = (int(part) for part in race_id.split("_"))
    con = get_connection()
    try:
        q = con.execute(
            "SELECT driver_id, q1, q2, q3 FROM bronze.ergast_qualifying WHERE season = ? AND round = ?",
            [season, rnd],
        ).df()
    finally:
        con.close()
    if q.empty:
        return {}
    best = q[["q1", "q2", "q3"]].map(_lap_seconds).min(axis=1)
    if best.notna().sum() < 5:
        return {}
    gap = (best - best.median()).clip(-QUALI_CLIP_SECONDS, QUALI_CLIP_SECONDS)
    # No time at all (didn't set one): as slow as the clip allows, which is
    # where a car with no lap starts and what it most likely has.
    gap = gap.fillna(QUALI_CLIP_SECONDS)
    return dict(zip(q["driver_id"], gap.astype(float)))


def weekend_pace(form: float, quali_gap: float | None) -> float:
    if quali_gap is None or not np.isfinite(quali_gap):
        return FORM_WEIGHT_WITHOUT_QUALI * form
    return FORM_WEIGHT * form + QUALI_WEIGHT * quali_gap


# --------------------------------------------------------------------------
# Fitting — reproduces the constants above, and the error spread that
# the docstring compares with the simulation's pre-race table.
# --------------------------------------------------------------------------

TRAIN_SEASONS = range(2018, 2025)
TEST_SEASON = 2025


def _dataset() -> pd.DataFrame:
    from race_plan.field import _season_form

    con = get_connection()
    try:
        race = con.execute(
            """
            SELECT rf.race_id, r.season, r.date, rf.driver_id, AVG(rf.pace_delta_this_lap) AS race_pace
            FROM gold.race_features rf JOIN silver.races r ON r.race_id = rf.race_id
            WHERE NOT rf.is_pit_lap AND NOT rf.safety_car_active AND NOT rf.vsc_active
                AND NOT rf.red_flag_active AND NOT COALESCE(rf.rainfall_flag, FALSE)
            GROUP BY 1, 2, 3, 4
            HAVING COUNT(*) >= 15
            """
        ).df()
    finally:
        con.close()
    race["form"] = [_season_form(int(s), str(d)).get(drv, np.nan) for s, d, drv in zip(race.season, race.date, race.driver_id)]
    race["quali"] = [qualifying_gaps(r).get(d, np.nan) for r, d in zip(race.race_id, race.driver_id)]
    # |race pace| > 5 s/lap is a car that limped round, not a pace.
    return race.dropna(subset=["form", "quali", "race_pace"]).query("abs(race_pace) < 5")


def fit() -> None:
    df = _dataset()
    train, test = df[df.season.isin(TRAIN_SEASONS)], df[df.season == TEST_SEASON]

    def lstsq(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(frame[cols].to_numpy(), frame["race_pace"].to_numpy(), rcond=None)
        return beta

    alone = lstsq(train, ["form"])
    both = lstsq(train, ["form", "quali"])
    for name, cols, beta in (("form alone", ["form"], alone), ("form + quali", ["form", "quali"], both)):
        err = test[cols].to_numpy() @ beta - test["race_pace"]
        print(
            f"{name:13s} coef {np.round(beta, 3)}   {TEST_SEASON}: MAE {err.abs().mean():.3f}  "
            f"median {err.abs().median():.3f}  corr {np.corrcoef(test[cols].to_numpy() @ beta, test['race_pace'])[0, 1]:.3f}"
        )
    # Percentiles 1-99 of actual minus estimate, over the seasons
    # monte_carlo's pre-race table was measured on, to compare with it.
    fit_rows = df[df.season.between(2018, 2023)]
    resid = fit_rows["race_pace"] - fit_rows[["form", "quali"]].to_numpy() @ both
    print(f"\npre-race error with qualifying, 2018-2023, {len(resid)} driver-races, percentiles 1-99:")
    print(", ".join(f"{v:.3f}" for v in np.percentile(resid, range(1, 100))))


if __name__ == "__main__":
    fit()
