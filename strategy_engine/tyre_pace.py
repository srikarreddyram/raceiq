"""How much a tyre plan costs in pace — measured tyre physics, applied the
same way to every car in the simulation.

Replaces the LSTM as the source of the pace DIFFERENCE between candidate
strategies, for two measured reasons:

- Across the 110 candidates for Hamilton at Bahrain 2025 lap 20, almost
  all two-stoppers, the LSTM's predicted pace had a standard deviation of
  0.53 s/lap and a 2 s/lap range. That is model noise, not strategy: the
  engine ranks every candidate and keeps the best, so it reliably picked
  the most optimistic outlier (-1.37 s/lap against his measured -0.63),
  and a P10 car won a third of simulations (the Win Probability model:
  2.5%).
- Only our car had a strategy-aware pace. Every rival kept its recent pace
  for the rest of the race — no tyre wear, no fresh-tyre gain after its
  stop — so our plans were scored against a field that never aged.

Here a car's pace over the remaining race is its recent pace with its
CURRENT tyre age's cost taken out (an age-neutral level), plus the mean
tyre cost of its plan over the remaining laps. Our car's plan is each
candidate strategy; a rival's is "run the current tyres until the stint
runs out, then stop as often as the circuit's typical stint length
demands, on hards" — the same stops the simulation charges it
(tyre_baselines.rival_stop_laps).

Cost per lap is the season's measured, fuel-corrected wear rate for the
compound (car_profiles/degradation_curves.season_model) times tyre age,
capped at 35 laps. Per season, so regulation eras keep their own tyres.
Compound pace OFFSETS from the same regression are deliberately not used:
they came out physically backwards in some seasons (softs slower than
mediums by 0.29 s/lap in 2021), because which compound gets used is
confounded with when and where. Wear is stable (0.02-0.06 s/lap per lap
in every season). Wet and unknown compounds carry no wear model and cost
nothing here — their level comes from the car's recent pace alone.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from car_profiles.degradation_curves import DRY_COMPOUNDS, season_model
from strategy_engine.field import RivalTrend
from strategy_engine.state import RaceState
from strategy_engine.tyre_baselines import RIVAL_FINAL_COMPOUND, rival_stop_laps

MAX_WEAR_AGE = 35  # matches monte_carlo.MAX_TYRE_AGE_FOR_SIMULATION
TREND_WINDOW_MIDPOINT = 2  # recent pace is a 5-lap mean; its tyres were ~2 laps younger than now


@lru_cache(maxsize=None)
def wear_rates(season: int | None) -> dict[str, float]:
    """Measured wear per compound for `season`, falling back to the nearest
    earlier season with data (a season's first race has none of its own)."""
    candidates = [season] if season is not None else []
    candidates += list(range((season or 2026) - 1, 2017, -1))
    for s in candidates:
        try:
            wear = season_model(s)["field_wear_seconds_per_lap"]
        except ValueError:
            continue
        return {c: max(0.0, float(w)) for c, w in wear.items()}
    return {c: 0.0 for c in DRY_COMPOUNDS}


def tyre_cost(compound: str | None, tyre_age: float, wear: dict[str, float]) -> float:
    rate = wear.get(str(compound).upper(), 0.0) if compound else 0.0
    return rate * float(np.clip(tyre_age, 0, MAX_WEAR_AGE))


def plan_cost(tyre_plan: list[dict], wear: dict[str, float]) -> float:
    """Mean per-lap tyre cost over a plan from monte_carlo._deterministic_tyre_plan."""
    if not tyre_plan:
        return 0.0
    return float(np.mean([tyre_cost(lap["compound"], lap["tyre_age"], wear) for lap in tyre_plan]))


def age_neutral(recent_pace: float, compound: str | None, tyre_age: float, wear: dict[str, float]) -> float:
    return recent_pace - tyre_cost(compound, max(tyre_age - TREND_WINDOW_MIDPOINT, 0.0), wear)


def rival_plan_cost(rival: RivalTrend, circuit_id: str, n_laps: int, wear: dict[str, float]) -> float:
    stops = set(rival_stop_laps(circuit_id, rival.compound, rival.tyre_age, n_laps))
    costs = []
    compound, age = rival.compound, rival.tyre_age
    for lap in range(n_laps):
        if lap in stops:
            compound, age = RIVAL_FINAL_COMPOUND, 0.0
        else:
            age += 1
        costs.append(tyre_cost(compound, age, wear))
    return float(np.mean(costs)) if costs else 0.0


def tyre_adjusted_rivals(rivals: list[RivalTrend], state: RaceState, season: int | None) -> list[RivalTrend]:
    """Each rival's pace over the remaining race: age-neutral recent pace
    plus the tyre cost of its assumed stops (tyre_baselines.rival_stop_laps)."""
    wear = wear_rates(season)
    n_laps = state.race_total_laps - state.current_lap
    adjusted = []
    for rival in rivals:
        recent = rival.recent_pace_delta if np.isfinite(rival.recent_pace_delta) else 0.0
        pace = age_neutral(recent, rival.compound, rival.tyre_age, wear) + rival_plan_cost(rival, state.circuit_id, n_laps, wear)
        adjusted.append(replace(rival, recent_pace_delta=pace))
    return adjusted
