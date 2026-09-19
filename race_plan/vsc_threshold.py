"""When to stop waiting for a safety car and just pit.

A stop under a safety car or VSC is cheap: the whole field is slowed, so
the time lost in the pit lane costs far less track position than it does
under green. That makes "wait a bit longer and hope for a caution" a real
strategy — and a real trap, because every lap spent waiting is a lap on a
tyre that keeps getting slower.

The trade has three quantities this project already measures, so the
threshold doesn't need a new model:

  h(lap)  per-lap probability a safety car appears, from the Safety Car
          model (models/safety_car/), converted from its N-lap window
          output to a per-lap hazard exactly as the Monte Carlo simulation
          does.
  S       what a caution saves on the stop itself: the circuit's own
          measured pit loss (strategy_engine/pit_loss.py) times the
          fraction of it a caution removes.
  d(age)  what one more lap on the current tyre costs, relative to having
          already switched to a fresh one — the fitted degradation slope
          multiplied by how old the tyre already is.

Waiting one more lap is worth it while

    h(lap) * S  >  d(age)

The left side is roughly flat across a race; the right side grows every
lap as the tyre ages. So they cross exactly once, and that crossing is
the "pull the plug" lap: past it, the expected saving from a caution no
longer covers what the worn tyre is costing you to keep waiting. Before
it, there's a genuine window where holding on is the better bet.

This is deliberately an expected-value rule and not a simulation. It's
meant to be explainable on the pit wall in one sentence — "we hold until
lap 34, after that the tyre costs more than the safety car saves" — which
a 5,000-run Monte Carlo distribution is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.common.registry import load_latest_model
from models.safety_car.train import FEATURE_COLUMNS as SAFETY_CAR_FEATURES
from models.safety_car.train import WINDOW_LAPS as SAFETY_CAR_WINDOW_LAPS
from strategy_engine.model_features import apply_reference_categoricals
from strategy_engine.pit_loss import typical_pit_loss_seconds
from strategy_engine.state import RaceState
from strategy_engine.tyre_baselines import typical_degradation_rate

# A caution doesn't make a stop free — the pit lane still has to be
# driven. This is the complement of the simulation's SC_PIT_DISCOUNT (the
# fraction of pit loss still paid under a caution), kept consistent with
# it deliberately: the two numbers describe the same physical effect and
# must not drift apart.
from strategy_engine.simulation.monte_carlo import SC_PIT_DISCOUNT

CAUTION_SAVING_FRACTION = 1.0 - SC_PIT_DISCOUNT


@dataclass
class WaitWindow:
    """The answer, per stop: hold until `pull_the_plug_lap`, then stop."""

    open_lap: int
    pull_the_plug_lap: int
    latest_safe_lap: int
    caution_saving_seconds: float
    per_lap_caution_probability: float
    reason: str

    @property
    def has_window(self) -> bool:
        return self.pull_the_plug_lap > self.open_lap


def per_lap_caution_probability(state: RaceState, lap_number: int) -> float:
    """The Safety Car model's P(caution within the next N laps), converted
    to a single-lap hazard via the standard "at least one event" identity —
    the same conversion build_shared_context uses, so the planner and the
    simulator can't disagree about how likely a caution is.
    """
    model = load_latest_model("safety_car_probability")
    values = {
        "lap_number": lap_number,
        "laps_remaining": state.race_total_laps - lap_number,
        "closest_gap_on_track": state.gap_to_car_ahead,
        "condition_delta": state.condition_delta,
        "historical_sc_rate": state.historical_sc_rate,
        "safety_car_active": 0,
        "yellow_active": 0,
        "vsc_active": 0,
        "rainfall_flag": int(state.rainfall_flag),
        "circuit_id": state.circuit_id,
    }
    row = {col: values.get(col, 0) for col in SAFETY_CAR_FEATURES}
    p_window = float(model.predict_proba(apply_reference_categoricals(pd.DataFrame([row])))[0, 1])
    return 1 - (1 - p_window) ** (1 / SAFETY_CAR_WINDOW_LAPS)


def compute_wait_window(
    state: RaceState,
    *,
    compound: str,
    stint_start_lap: int,
    earliest_lap: int,
    latest_safe_lap: int,
    exclude_race_id: str | None = None,
) -> WaitWindow:
    """Find the lap at which waiting for a caution stops paying.

    `latest_safe_lap` is the hard tyre-life bound from the planner — past
    it the stint isn't viable regardless of what the expected value says,
    so the window is clipped there rather than allowed to recommend
    running a tyre into the ground on the chance of a caution.
    """
    # exclude_race_id keeps a pre-race plan from reading the race it is
    # planning — critical at a first-time circuit, where that race is the
    # only data there is.
    pit_loss = typical_pit_loss_seconds(state.circuit_id, exclude_race_id)
    saving = pit_loss * CAUTION_SAVING_FRACTION

    degradation = state.degradation_rate
    if degradation is None or pd.isna(degradation) or degradation <= 0:
        # A non-positive fitted slope means this stint hasn't shown
        # measurable wear yet (or the fit is noise). Fall back to what this
        # compound typically does at this circuit rather than concluding
        # the tyre never degrades, which would make waiting look free
        # forever.
        degradation = typical_degradation_rate(state.circuit_id, compound, exclude_race_id)

    pull_the_plug = earliest_lap
    for lap in range(earliest_lap, latest_safe_lap + 1):
        tyre_age = lap - stint_start_lap
        cost_of_one_more_lap = degradation * max(tyre_age, 1)
        expected_saving = per_lap_caution_probability(state, lap) * saving
        if expected_saving < cost_of_one_more_lap:
            break
        pull_the_plug = lap

    hazard_at_open = per_lap_caution_probability(state, earliest_lap)
    if pull_the_plug >= latest_safe_lap:
        reason = (
            f"tyre life runs out first — a caution is still worth waiting for at lap {latest_safe_lap}, "
            f"but the stint can't safely go further"
        )
    elif pull_the_plug <= earliest_lap:
        reason = (
            f"no waiting window — at {hazard_at_open * 100:.1f}%/lap a caution saves "
            f"{hazard_at_open * saving:.2f}s in expectation, already less than the "
            f"{degradation * max(earliest_lap - stint_start_lap, 1):.2f}s/lap the worn tyre is costing"
        )
    else:
        reason = (
            f"hold to lap {pull_the_plug}: a caution saves {saving:.1f}s and is "
            f"{hazard_at_open * 100:.1f}% likely per lap, which beats the worn tyre's cost until then"
        )

    return WaitWindow(
        open_lap=earliest_lap,
        pull_the_plug_lap=min(pull_the_plug, latest_safe_lap),
        latest_safe_lap=latest_safe_lap,
        caution_saving_seconds=saving,
        per_lap_caution_probability=hazard_at_open,
        reason=reason,
    )
