"""Monte Carlo simulation — PRD Section 12.4.

A candidate strategy fixes its own pit laps and compounds by definition,
so which laps are pit stops is deterministic — identical across all 5,000
simulations. What's actually stochastic is just (a) whether a safety car
occurs on a given lap, and (b) lap-to-lap pace noise. That means the
expensive part (calling the Safety Car model) only needs to happen once
per *remaining lap*, not once per lap per simulation — the difference
between meeting PRD Section 19's "5,000 scenarios in under 10 seconds"
and not.

**Getting the comparison across strategies right took four attempts —
worth knowing in full, because each failure mode is non-obvious and the
next fix undid the previous one's approach:**

1. Ranking each strategy's absolute simulated lap time against a
   *constant* field-average baseline broke as soon as a stochastic event
   was involved: a safety car correctly inflated the driver's own
   predicted pace, but the constant baseline it was measured against did
   not — so every SC-affected simulation showed the driver losing
   enormous ground to a "field" that never experienced the same slowdown.
   With SC probability compounding across 30+ remaining laps into
   near-certainty, every strategy predicted the leader finishing last
   almost every time.
2. Differencing against a shared "never pit again" reference trajectory
   (a "common random numbers" fix) solved that, but introduced a worse
   problem: the reference itself often required tyre ages no real stint
   reaches, and even after capping tyre age, small per-lap biases in an
   autoregressive prediction chain (feeding each lap's predicted time
   into the next lap's input) compounded across 30+ steps into a
   multi-second gap between two strategies that looked nearly identical
   lap by lap — enough to make every real candidate register a false
   100% win probability against that one broken baseline.
3. Removing the reference trajectory and comparing each strategy only to
   its own no-SC counterfactual fixed that specific problem, but a
   subtler asymmetry remained: an SC's *own* pace penalty was still being
   added to this driver's gap even though a safety car slows the whole
   field roughly equally — the leader is under the same caution. Adding
   that shared, track-wide effect to only one side of the comparison
   still produced a near-certain, one-sided penalty over a long-enough
   remaining distance.
4. Re-deriving tyre/compound pace differences from the same autoregressive
   lap-time chain (to let compound choice still influence the ranking)
   surfaced the deepest issue: that chain is genuinely chaotic with a
   tree-based model over 30+ steps, not just noisy. For some race states
   it collapsed to an unrealistic flat prediction within two or three
   iterations — a leaf-boundary artifact, not a real pace trend — enough
   to swing an actual last-place driver to a false 100% win probability.
   Damping the recurrence toward a stable anchor reduced drift but didn't
   prevent this; a tree model's prediction surface isn't smooth enough
   for self-referential chaining to be reliable over a full-race horizon.

What ships here reflects all four lessons: safety-car draws and lap-time
noise are shared across every candidate (`build_shared_context`, true
common random numbers), a safety car's pace effect never touches the
gap-to-leader directly, and — the one real, acknowledged gap against the
PRD's vision — candidate strategies are differentiated by tyre-life
feasibility (filtered at search time via the Tyre Degradation model, see
search/candidates.py), pit count and timing, and safety-car luck, not by
fine-grained predicted pace differences between compounds. Closing that
gap needs either a smoother model for multi-lap-ahead prediction or a
non-autoregressive way to query the Lap Time model that doesn't compound
its own output — noted as follow-up work, not silently worked around.

Two more simplifications, documented where they matter:
- Rivals' future pace projects their *current* trend forward over a
  bounded horizon (see field.py and RIVAL_TREND_HORIZON_LAPS below)
  rather than simulating their own strategic decisions — in particular,
  rivals are never modeled as pitting, so any strategy that does pit
  carries a real, bounded, and explicable handicap in this comparison
  roughly equal to one pit stop's time loss.
- "Pitting under a safety car is free track position" (PRD Section 12.6's
  own example reasoning) is modeled as a discount on this driver's own
  pit loss when it lands on a simulated SC lap — rivals bunching up under
  that same SC isn't modeled, so this understates the real effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.common.registry import load_latest_model
from models.safety_car.train import FEATURE_COLUMNS as SAFETY_CAR_FEATURES
from models.safety_car.train import WINDOW_LAPS as SAFETY_CAR_WINDOW_LAPS
from strategy_engine.field import RivalTrend
from strategy_engine.model_features import apply_reference_categoricals
from strategy_engine.pit_loss import typical_pit_loss_seconds
from strategy_engine.search.candidates import Strategy
from strategy_engine.state import RaceState

LAP_TIME_NOISE_SECONDS = 1.2  # grounded in the Lap Time model's own measured stable-regime RMSE (~1.07s)
RIVAL_PACE_NOISE_PER_LAP = 0.5
RIVAL_TREND_HORIZON_LAPS = 10  # how far a rival's short-term (5-lap) pace trend is projected before reverting to neutral
SC_PIT_DISCOUNT = 0.3  # fraction of pit loss still paid if the stop lands under a simulated safety car


@dataclass
class SimulationResult:
    strategy: Strategy
    final_positions: np.ndarray  # shape (n_simulations,), 1-indexed
    safety_car_occurred: np.ndarray  # shape (n_simulations,), bool


@dataclass
class SharedContext:
    """Randomness computed once per race state and reused across every
    candidate strategy so they're all compared under identical safety-car
    and noise conditions (see module docstring on why sharing this
    matters, and why it deliberately carries no reference trajectory).
    """

    n_laps: int
    sc_occurs: np.ndarray  # shape (n_simulations, n_laps)
    noise: np.ndarray  # shape (n_simulations, n_laps)


def _safety_car_row(state: RaceState, lap_number: int) -> pd.DataFrame:
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
    return apply_reference_categoricals(pd.DataFrame([row]))


def _deterministic_pit_plan(state: RaceState, strategy: Strategy) -> list[bool]:
    """One entry per remaining lap: whether it's a scheduled pit lap —
    fixed by the strategy, identical for every simulation.
    """
    pit_laps = {lap for lap, _ in strategy.pit_plan}
    return [lap in pit_laps for lap in range(state.current_lap + 1, state.race_total_laps + 1)]


def build_shared_context(state: RaceState, n_simulations: int, rng: np.random.Generator) -> SharedContext:
    """Computed once per race state, reused for every candidate strategy —
    the safety-car draws and lap-time noise, shared so every strategy is
    evaluated under identical random conditions (common random numbers).
    """
    n_laps = state.race_total_laps - state.current_lap
    safety_car_model = load_latest_model("safety_car_probability")
    sc_probs = np.empty(n_laps)
    for i, lap_number in enumerate(range(state.current_lap + 1, state.race_total_laps + 1)):
        # SC probability depends only on lap number / circuit / conditions —
        # not on tyre strategy — so it's identical for every candidate and
        # only needs computing once here.
        p_window = safety_car_model.predict_proba(_safety_car_row(state, lap_number))[0, 1]
        # The model predicts P(SC somewhere in the next WINDOW_LAPS laps),
        # not P(SC on this exact lap) — treating it as a per-lap hazard
        # directly would massively over-count exposure once compounded
        # across dozens of remaining laps. Converting via the standard "at
        # least one event" formula recovers an approximate independent
        # per-lap hazard: if 1-(1-h)^W = p_window, then h = 1-(1-p_window)^(1/W).
        sc_probs[i] = 1 - (1 - p_window) ** (1 / SAFETY_CAR_WINDOW_LAPS)

    sc_occurs = rng.random((n_simulations, n_laps)) < sc_probs[None, :]
    noise = rng.normal(0, LAP_TIME_NOISE_SECONDS, size=(n_simulations, n_laps))

    return SharedContext(n_laps=n_laps, sc_occurs=sc_occurs, noise=noise)


def simulate_strategy(
    state: RaceState,
    strategy: Strategy,
    rivals: list[RivalTrend],
    shared: SharedContext,
) -> SimulationResult:
    n_simulations = shared.sc_occurs.shape[0]
    is_pit_lap = _deterministic_pit_plan(state, strategy)
    pit_loss = typical_pit_loss_seconds(state.circuit_id)

    # A safety car slows the whole field by roughly the same amount — the
    # leader is under the same caution as everyone else — so it doesn't
    # move this driver's *gap to the leader* on its own. Only two things
    # legitimately do that here: this strategy's own pit stop cost (only
    # this driver stops, discounted if it lands under a simulated SC — PRD
    # 12.6's "pitting under a safety car is free track position"), and
    # idiosyncratic lap-to-lap noise. See the module docstring for what
    # else was tried here and why it didn't hold up.
    pit_time_loss = np.zeros((n_simulations, shared.n_laps))
    for i, scheduled in enumerate(is_pit_lap):
        if scheduled:
            discount = np.where(shared.sc_occurs[:, i], SC_PIT_DISCOUNT, 1.0)
            pit_time_loss[:, i] = pit_loss * discount

    cumulative_deviation = (shared.noise + pit_time_loss).sum(axis=1)
    our_final_gap = state.gap_to_leader + cumulative_deviation

    # A rival's recent pace trend is measured over a short (5-lap) window and
    # is itself noisy — extrapolating it linearly across the *entire*
    # remaining race compounds that noise into an unrealistic sustained
    # advantage or deficit. Projecting it only across a bounded horizon and
    # assuming field-average pace beyond that is a mean-reversion
    # assumption: a short-term pace difference (fuel phase, tyre phase)
    # doesn't linearly persist for 30+ laps.
    effective_laps = min(shared.n_laps, RIVAL_TREND_HORIZON_LAPS)

    rng = np.random.default_rng()  # rival noise doesn't need to be paired across strategies
    rival_final_gaps = np.stack(
        [
            rival.gap_to_leader
            + effective_laps * rival.recent_pace_delta
            + rng.normal(0, RIVAL_PACE_NOISE_PER_LAP * np.sqrt(shared.n_laps), size=n_simulations)
            for rival in rivals
        ],
        axis=0,
    )  # shape (n_rivals, n_simulations)

    final_position = 1 + (rival_final_gaps < our_final_gap[None, :]).sum(axis=0)

    return SimulationResult(
        strategy=strategy,
        final_positions=final_position,
        safety_car_occurred=shared.sc_occurs.any(axis=1),
    )
