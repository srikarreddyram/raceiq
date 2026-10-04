"""The wet-race plan: what to do if the track is wet.

The dry planner (race_plan/plan.py) plans on slicks. In the rain that is
the wrong answer — the field goes onto intermediates or full wets — so
when rain is forecast, or fell in a race already run, the plan also
carries this: the tyre to start on, when to come back to slicks, what to
do if rain arrives mid-race, and how much more often a wet race is
neutralised.

Why a set of measured calls and not a simulated wet race
--------------------------------------------------------
The dry simulation is fitted on hundreds of races. Only about twenty
since 2018 were run mostly on wet-weather tyres, too few to fit wet pace,
wet tyre wear and crossover timing the same way. And a forecast can't say
which laps will be wet: predicting, lap by lap, whether the field would be
on wet-weather tyres from hourly rainfall (the archive a forecast
resembles), on races held out of the fit, caught 25% of the laps it was,
with 46% of its calls false alarms. It missed Turkey 2021 and Japan 2022
entirely. Rain at a circuit is too local for an hourly grid.

So the plan gives the chance of a wet race from the forecast, and what
teams have actually done once the track was wet — measured from the
track's own rain sensor and every car's tyres in every wet race in the
warehouse (wet_race_facts). The call between this plan and the dry one is
made on the day, looking at the track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.data import is_classified
from models.common.db import get_connection

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
SLICKS = ("SOFT", "MEDIUM", "HARD")
WET_PLAN_PROBABILITY = 0.3  # show the wet plan from this chance of rain at the start
WET_PLAN_LEADS = 0.5  # and lead with it, ahead of the dry plan, from this one


@dataclass
class WetRaceFacts:
    wet_races: int
    wet_starts: int
    start_on_inters: float  # share of cars at a wet start
    start_on_wets: float
    wet_start_races_on_wets: int  # races where most of the field started on full wets
    wet_starts_behind_sc: int
    to_slicks_median_laps: float | None  # laps from the last rain on track to most of the field on slicks
    to_slicks_range: tuple[float, float] | None
    to_slicks_races: int
    stayed_wet_races: int  # never went back to slicks
    early_switch_gain: float | None  # places gained, switching with or just before the field
    late_switch_gain: float | None  # 1-3 laps after the field
    switch_cars: tuple[int, int]
    onto_wet_tyres_median_laps: float | None  # rain arriving mid-race: laps until most of the field is on wet-weather tyres
    onto_wet_tyres_races: int
    slick_after_wet: dict[str, float]  # compound fitted when switching back
    sc_rate: tuple[float, float]  # wet, dry
    red_flag_rate: tuple[float, float]
    dnf_rate: tuple[float, float]
    stops: tuple[float, float]  # per classified finisher, wet, dry


@lru_cache(maxsize=1)
def wet_race_facts() -> WetRaceFacts:
    """Every race in the warehouse where most of the field ran wet-weather
    tyres at some point, measured. Rain is the track's own sensor."""
    con = get_connection()
    try:
        laps = con.execute(
            """
            SELECT lf.race_id, lf.driver_id, lf.lap_number AS lap, lf.compound, lf.stint_number,
                   lf.current_position AS pos, lf.safety_car_active AS sc, lf.red_flag_active AS red, w.rainfall AS rain
            FROM gold.lap_features lf
            LEFT JOIN silver.weather w ON w.race_id = lf.race_id AND w.lap_number = lf.lap_number
            """
        ).df()
        results = con.execute("SELECT season || '_' || round AS race_id, driver_id, status FROM bronze.ergast_results").df()
    finally:
        con.close()

    laps["cls"] = np.select(
        [laps.compound == "WET", laps.compound == "INTERMEDIATE", laps.compound.isin(SLICKS)], ["wet", "inter", "slick"], "other"
    )
    known = laps[laps.cls != "other"]
    share = known.groupby(["race_id", "lap"])["cls"].value_counts(normalize=True).unstack(fill_value=0)
    for c in ("wet", "inter", "slick"):
        if c not in share:
            share[c] = 0.0
    share = share.reset_index()
    share["wetlike"] = share["wet"] + share["inter"]
    wet_races = share.groupby("race_id")["wetlike"].max()
    wet_races = sorted(wet_races[wet_races >= 0.5].index)
    rain_by_lap = laps.groupby(["race_id", "lap"])["rain"].max().fillna(False).astype(bool)

    # Wet starts.
    lap1 = known[known.lap == 1]
    start_share = lap1.groupby("race_id")["cls"].value_counts(normalize=True).unstack(fill_value=0)
    wetlike_start = start_share.get("wet", 0) + start_share.get("inter", 0)
    wet_start_ids = start_share[wetlike_start >= 0.5].index
    start_cars = lap1[lap1.race_id.isin(wet_start_ids)]
    early_sc = laps[laps.lap <= 3].groupby("race_id")["sc"].max()

    # Back to slicks, and who gained by the timing of their switch.
    to_slicks, stayed, gains, slick_fitted = [], 0, [], []
    for race_id in wet_races:
        s = share[share.race_id == race_id].sort_values("lap")
        wet_from = int(s[s.wetlike >= 0.5].lap.min())
        back = s[(s.lap > wet_from) & (s.slick >= 0.5)]
        if back.empty:
            stayed += 1
            continue
        switch = int(back.lap.min())
        rain = rain_by_lap.loc[race_id]
        rained = rain[(rain.index < switch) & rain]
        if len(rained):
            to_slicks.append(switch - int(rained.index.max()))
        car_laps = laps[laps.race_id == race_id].sort_values("lap")
        first = car_laps[(car_laps.cls == "slick") & (car_laps.lap > wet_from)].groupby("driver_id").agg(lap=("lap", "min"), compound=("compound", "first"))
        slick_fitted += first["compound"].tolist()
        for driver_id, row in first.iterrows():
            before = car_laps[(car_laps.driver_id == driver_id) & (car_laps.lap == switch - 5)].pos
            after = car_laps[(car_laps.driver_id == driver_id) & (car_laps.lap == switch + 10)].pos
            if len(before) and len(after):
                gains.append((row["lap"] - switch, float(before.iloc[0] - after.iloc[0])))
    g = pd.DataFrame(gains, columns=["rel", "gained"])
    early = g[g.rel.between(-2, 0)]
    late = g[g.rel.between(1, 3)]

    # Rain arriving mid-race.
    onto = []
    for race_id in wet_races:
        s = share[share.race_id == race_id].sort_values("lap")
        if s.wetlike.iloc[:3].mean() >= 0.5:
            continue
        on = int(s[s.wetlike >= 0.5].lap.min())
        rain = rain_by_lap.loc[race_id]
        first_rain = rain[rain & (rain.index <= on)]
        if len(first_rain):
            onto.append(on - int(first_rain.index.min()))

    # Cautions, retirements and stops: wet races against the rest.
    flags = laps.groupby("race_id").agg(sc=("sc", "max"), red=("red", "max"))
    flags["wet"] = flags.index.isin(wet_races)
    results["dnf"] = ~results["status"].map(is_classified)
    flags = flags.join(results.groupby("race_id")["dnf"].mean())
    stops = (laps.groupby(["race_id", "driver_id"])["stint_number"].max() - 1).rename("stops").reset_index()
    finishers = results[~results["dnf"]][["race_id", "driver_id"]]
    stops = stops.merge(finishers, on=["race_id", "driver_id"])
    stops["wet"] = stops.race_id.isin(wet_races)

    def pair(frame: pd.DataFrame, column: str) -> tuple[float, float]:
        by = frame.groupby("wet")[column].mean()
        return float(by.get(True, np.nan)), float(by.get(False, np.nan))

    fitted = pd.Series(slick_fitted).value_counts(normalize=True) if slick_fitted else pd.Series(dtype=float)
    return WetRaceFacts(
        wet_races=len(wet_races),
        wet_starts=len(wet_start_ids),
        start_on_inters=float((start_cars.cls == "inter").mean()) if len(start_cars) else float("nan"),
        start_on_wets=float((start_cars.cls == "wet").mean()) if len(start_cars) else float("nan"),
        wet_start_races_on_wets=int((start_share.loc[wet_start_ids].get("wet", 0) >= 0.5).sum()) if len(wet_start_ids) else 0,
        wet_starts_behind_sc=int(early_sc.reindex(wet_start_ids).fillna(False).sum()),
        to_slicks_median_laps=float(np.median(to_slicks)) if to_slicks else None,
        to_slicks_range=(float(np.percentile(to_slicks, 25)), float(np.percentile(to_slicks, 75))) if to_slicks else None,
        to_slicks_races=len(to_slicks),
        stayed_wet_races=stayed,
        early_switch_gain=float(early.gained.mean()) if len(early) else None,
        late_switch_gain=float(late.gained.mean()) if len(late) else None,
        switch_cars=(len(early), len(late)),
        onto_wet_tyres_median_laps=float(np.median(onto)) if onto else None,
        onto_wet_tyres_races=len(onto),
        slick_after_wet={k: float(v) for k, v in fitted.items()},
        sc_rate=pair(flags, "sc"),
        red_flag_rate=pair(flags, "red"),
        dnf_rate=pair(flags, "dnf"),
        stops=pair(stops, "stops"),
    )


@dataclass
class WetPlan:
    chance_at_start: float | None  # chance of rain around the start
    chance_during: float | None  # highest chance in any hour of the race
    hourly: list[dict]  # [{"hour_utc", "rain_probability", "rain_mm"}]
    leads: bool  # rain likely enough that this plan leads the page
    start_tyre: str  # "INTERMEDIATE" or "WET"
    calls: list[dict] = field(default_factory=list)  # [{"title", "text"}]
    source: str = "forecast"  # or "measured", for a race already run


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def build_wet_plan(
    chance_at_start: float | None,
    hourly: list[dict] | None,
    source: str = "forecast",
    chance_during: float | None = None,
) -> WetPlan | None:
    """The wet-race plan for a race with this chance of rain, or None if
    rain is unlikely enough that the dry plan is the plan. `chance_during`
    overrides the hourly figures (a race already run: it rained or it
    didn't)."""
    hourly = hourly or []
    chances = [h["rain_probability"] for h in hourly if h.get("rain_probability") is not None]
    during = chance_during if chance_during is not None else (max(chances) if chances else chance_at_start)
    worst = max(x for x in (chance_at_start, during) if x is not None) if (chance_at_start is not None or during is not None) else None
    if worst is None or worst < WET_PLAN_PROBABILITY:
        return None
    f = wet_race_facts()
    # Intermediates by default: most cars at most wet starts. Whether the
    # grid is under standing water is a call made by looking at it — the
    # forecast can't tell a shower from a flood at a single circuit.
    start_tyre = "INTERMEDIATE"
    first_mm = hourly[0].get("rain_mm") if hourly else None
    calls = [
        {
            "title": "Start on intermediates",
            "text": (
                f"At the {f.wet_starts} wet starts since 2018, {_pct(f.start_on_inters)} of cars started on intermediates and "
                f"{_pct(f.start_on_wets)} on full wets; full wets were most of the field's start tyre in "
                f"{f.wet_start_races_on_wets} of them. Switch to full wets if there's standing water on the grid on the day."
                + (f" The forecast has {first_mm:.1f} mm of rain in the first hour." if first_mm is not None else "")
                + f" {f.wet_starts_behind_sc} of the {f.wet_starts} wet starts had the safety car out in the first three laps."
            ),
        },
    ]
    if f.to_slicks_median_laps is not None:
        early = f"+{f.early_switch_gain:.1f}" if f.early_switch_gain is not None else "—"
        late = f"{f.late_switch_gain:+.1f}" if f.late_switch_gain is not None else "—"
        lo, hi = f.to_slicks_range
        mix = ", ".join(f"{c.lower()} {_pct(v)}" for c, v in sorted(f.slick_after_wet.items(), key=lambda kv: -kv[1])[:3])
        calls.append(
            {
                "title": f"Back to slicks about {f.to_slicks_median_laps:.0f} laps after the rain stops",
                "text": (
                    f"That's when most of the field has switched (usually {lo:.0f}-{hi:.0f} laps, over {f.to_slicks_races} races). "
                    f"Go with the first wave: cars that switched with the field or up to two laps ahead gained {early} places "
                    f"on average ({f.switch_cars[0]} cars); a lap to three late, {late} ({f.switch_cars[1]} cars). "
                    f"Slicks fitted at the switch: {mix}. In {f.stayed_wet_races} of {f.wet_races} wet races the track never dried "
                    f"enough and the field finished on wet-weather tyres."
                ),
            }
        )
    if f.onto_wet_tyres_median_laps is not None:
        calls.append(
            {
                "title": "If the rain arrives during the race",
                "text": (
                    f"Most of the field was on intermediates {f.onto_wet_tyres_median_laps:.0f} laps after rain started on track "
                    f"({f.onto_wet_tyres_races} races). Stopping a lap or two before the field does is the same call as the "
                    f"switch back to slicks, in reverse."
                ),
            }
        )
    sc_wet, sc_dry = f.sc_rate
    red_wet, red_dry = f.red_flag_rate
    stops_wet, stops_dry = f.stops
    calls.append(
        {
            "title": "Expect it to be neutralised",
            "text": (
                f"Wet races had a safety car {_pct(sc_wet)} of the time ({_pct(sc_dry)} dry) and a red flag {_pct(red_wet)} "
                f"({_pct(red_dry)} dry) — and a red flag hands everyone a free tyre change. Finishers made {stops_wet:.1f} stops "
                f"on average, against {stops_dry:.1f} in the dry: plan the tyre sets for three."
            ),
        }
    )
    return WetPlan(
        chance_at_start=chance_at_start,
        chance_during=during,
        hourly=hourly,
        leads=bool(worst >= WET_PLAN_LEADS),
        start_tyre=start_tyre,
        calls=calls,
        source=source,
    )
