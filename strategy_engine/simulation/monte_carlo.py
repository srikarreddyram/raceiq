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

**Round 5 closed that gap.** models/lap_time_sequence/ trains an LSTM that
takes a candidate's entire deterministic tyre/pit covariate sequence and
predicts every lap's time in one forward pass — no autoregressive
feedback loop, so nothing to chain into chaos. `lstm_oracle.py` wraps it;
`pace_deviation` below uses it to compare a strategy's actual predicted
green-flag pace against the field average, restoring the tyre/compound
differentiation round 4 had to give up. It isn't free of failure modes of
its own, though: a wet-race snapshot (a long INTERMEDIATE stint on a
drying track — rare enough in training data to be near-unseen) produced
an implied ~1.7s/lap sustained improvement, physically not impossible on
a drying track, but confident enough at N=5,000 to make an actual leader
show a flat, noise-proof 100% win probability. `PACE_DEVIATION_CAP_PER_LAP`
bounds that risk pragmatically (clip to a generous but finite per-lap
effect) rather than resolving the deeper question of whether that specific
prediction was correct or an extrapolation artifact — a proper fix would
have the model report its own uncertainty (e.g. quantile regression) so a
rare, low-confidence scenario widens this simulation's noise instead of
narrowing it to a point estimate.

**Round 6** is documented inline in `simulate_strategy`: this driver's
pace deviation was projected across the whole remaining race while rivals'
trends were capped at a ten-lap horizon, so whoever was fastest relative
to the field banked an advantage nobody could answer. Retraining the LSTM
made it *more* accurate and thereby made this *worse* — an honest -1.1s/lap
estimate for a race leader turned into a 42-second cushion and a 96% win
probability. Both sides now use the same horizon.

**Round 7 — the race is now run lap by lap, with track position.**
Everything above projected one number per car — its gap to the leader at
the flag — and ranked those, with a fixed "passing margin" bolted on at
the end to stop grid order from evaporating. That can't represent the
thing that makes grid position worth something: a faster car stuck
behind a slower one, losing time in its dirty air. Each car's race time
now advances a lap at a time, and on every lap:

- a car within DIRTY_AIR_GAP_SECONDS of the car ahead loses
  DIRTY_AIR_LOSS_SECONDS (measured: 0.142 s/lap, the same driver's pace
  within 1 s of a car against in clean air);
- it only gets past if it's quicker by the circuit's passing delta this
  lap; otherwise it's held FOLLOW_GAP_SECONDS behind;
- a car that pits drops back through the order by its pit loss;
- a safety car can bunch the field behind the leader (switched off —
  it validated worse; see SC_BUNCHES_FIELD);
- a retiring rival leaves on its drawn lap.

Every car gets the same lap-to-lap variation (LAP_NOISE_SECONDS, measured
at 0.405 s in clean air); our car previously had 1.2 s and rivals 0.5 s,
an asymmetry that would decide random passes. Pace-estimate errors are
sampled from their measured distributions instead of a bell curve whose
width outliers had inflated (see PACE_ERROR_PERCENTILES_*), and safety
cars start at a calibrated rate and last their measured five laps (see
SC_ONSET_SCALE) instead of firing as independent single laps in 78% of
races.

Validated against real 2025 finishes, not against the classifiers. In-race
(strategy_engine/validate_in_race.py — every finisher of nine dry 2025
races, from 40% distance, on the strategy they actually ran) the old
aggregate projection scored expected-finish MAE 4.78, Spearman 0.859,
P(win) Brier 0.144; this scores 2.01, 0.874 and 0.037, beating both the
Final Race Position classifier (3.16, 0.774) and "finish where you are
now" (2.30, 0.830). Pre-race (race_plan/grid_sensitivity.py, every
starter from their real grid slot) front-row and midfield starters now
land where real ones do — P1-5 average 3.67 vs 3.52 real (the aggregate
projection said 5.10), P6-10 8.38 vs 8.38 — but back-half starters are
still 1-1.5 places too optimistic, and per-race rank correlation (0.70)
is below the aggregate projection's 0.77. That residual is open.

Three more simplifications, documented where they matter:
- Rivals' future pace projects their *current* trend forward (see
  field.py) rather than simulating their own strategic decisions. Rivals are now
  charged one pit stop's time if their current tyre age would exceed a
  realistic stint length before the race ends (see
  tyre_baselines.typical_max_stint_length) — found necessary via the Win
  Probability model cross-check (recommendation/reasoning.py) on a real
  scenario where every rival, including the leader, was already on their
  second stint; without it, any strategy for OUR driver that pits was
  penalized against a field modeled as never stopping again.
- "Pitting under a safety car is free track position" (PRD Section 12.6's
  own example reasoning) is modeled as a discount on this driver's own
  pit loss when it lands on a simulated SC lap — rivals bunching up under
  that same SC isn't modeled, so this understates the real effect.
- Rival attrition/DNF is now modeled too, added for the same reason as the
  pit-stop fix above: the Final Race Position model cross-check
  (oracles.predict_expected_finish_now) found that for a real last-placed
  driver (Bahrain 2025, lap 20, bortoleto — genuinely P20), the
  classifier's historically-grounded estimate was P12.5 while the
  simulation's best strategy still showed a flat P20.0 — because rivals
  were never modeled as retiring, so a driver already last had zero
  chance of inheriting a place from someone else's mechanical failure or
  crash. `gold.circuit_history.historical_dnf_rate` (built the same way
  as `historical_sc_rate`, from `bronze.ergast_results.status`) now grounds
  a per-rival retirement draw in `build_shared_context`, converted from a
  full-race rate to a remaining-laps probability the same way the safety
  car model's window probability is converted to a per-lap hazard. This
  only models RIVALS retiring — our own driver's retirement risk isn't
  modeled, since the engine's job is recommending a strategy assuming they
  finish, not risk-adjusting for their own mechanical failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.common.registry import load_latest_model
from models.safety_car.train import FEATURE_COLUMNS as SAFETY_CAR_FEATURES
from models.safety_car.train import WINDOW_LAPS as SAFETY_CAR_WINDOW_LAPS
from strategy_engine.field import RivalTrend
from strategy_engine.lstm_oracle import predict_trajectory
from strategy_engine.model_features import apply_reference_categoricals
from strategy_engine.pit_loss import typical_pit_loss_seconds
from strategy_engine.search.candidates import Strategy
from strategy_engine.state import RaceState
from strategy_engine.tyre_baselines import rival_stop_laps, typical_degradation_rate

# Lap-to-lap variation in a car's lap time around its own trend, the same
# for every car: std of clean-air green-lap residuals from a quadratic
# trend within each stint, 2022-2025, 41,916 laps. (Replaced 1.2 s for our
# car — the Lap Time model's RMSE, which is model error, not lap-to-lap
# variation — and 0.5 s for rivals.)
LAP_NOISE_SECONDS = 0.405
# Retired in round 6 (see simulate_strategy): rivals' pace trends used to
# be projected across only this many laps while this driver's ran the full
# remaining race, and that asymmetry was the bug. Kept as a named record of
# the assumption rather than silently deleted, since "mean-revert pace to
# the field average after N laps" is a reasonable idea that simply has to
# be applied to both sides at once if it's applied at all.
RETIRED_RIVAL_TREND_HORIZON_LAPS = 10
SC_PIT_DISCOUNT = 0.3  # fraction of pit loss still paid if the stop lands under a simulated safety car
MAX_TYRE_AGE_FOR_SIMULATION = 35  # near the 99th percentile observed stint length across compounds in training data
PACE_DEVIATION_CAP_PER_LAP = 2.0  # seconds/lap; see simulate_strategy's use for why this exists
DEFAULT_DNF_RATE = 0.15  # this project's own dataset-wide average, for circuits with no prior-race history yet

# How wrong a pace estimate is, in seconds per lap — sampled from the
# MEASURED distribution, not a bell curve. Percentiles 1-99, 2018-2023
# (training seasons only):
#   PRE_RACE  (actual race pace) minus (season-to-date form before the race),
#             2,063 driver-races.
#   IN_RACE   (pace over laps 21+) minus (pace over laps 4-20), 1,865
#             driver-races with at least 8 laps in each.
#
# History, because both earlier versions were wrong in instructive ways.
# First these were normal draws with a std of 0.737 (pre-race) / 0.634
# (in-race). The std is dominated by a few huge outliers — damage, a
# disaster stint — while the middle of the distribution is much tighter
# (robust spread ~0.46 pre-race). A bell curve that wide hands every car
# a large chance of a miracle pace day that doesn't exist, which scattered
# front-runners backwards in every plan. Splitting the std by grid band
# (0.47 at the front to 1.16 at the back) looked like a finding and was
# the same artefact: the back bands' extra "spread" was outliers (skew 16
# for P16-20), and their 5th-95th percentile range matches the front's.
# The tails here stop at the 1st/99th percentile; real disasters are the
# retirement draws' job.
PACE_ERROR_PERCENTILES_PRE_RACE = (
    -1.149, -1.025, -0.873, -0.800, -0.739, -0.698, -0.671, -0.640, -0.601, -0.561, -0.534, -0.513,
    -0.500, -0.482, -0.466, -0.447, -0.438, -0.421, -0.408, -0.393, -0.383, -0.367, -0.356, -0.343,
    -0.332, -0.315, -0.303, -0.293, -0.280, -0.266, -0.257, -0.243, -0.229, -0.219, -0.205, -0.193,
    -0.184, -0.170, -0.158, -0.146, -0.133, -0.124, -0.115, -0.098, -0.087, -0.077, -0.068, -0.055,
    -0.050, -0.040, -0.031, -0.021, -0.009, -0.001, 0.015, 0.024, 0.034, 0.047, 0.056, 0.067,
    0.077, 0.086, 0.098, 0.111, 0.125, 0.133, 0.143, 0.152, 0.167, 0.179, 0.187, 0.194, 0.208,
    0.220, 0.234, 0.248, 0.263, 0.281, 0.300, 0.319, 0.335, 0.354, 0.379, 0.395, 0.417, 0.435,
    0.470, 0.489, 0.509, 0.536, 0.572, 0.606, 0.663, 0.711, 0.780, 0.848, 0.938, 1.103, 1.313,
)
PACE_ERROR_PERCENTILES_IN_RACE = (
    -1.158, -0.951, -0.843, -0.790, -0.746, -0.698, -0.674, -0.629, -0.589, -0.556, -0.534, -0.506,
    -0.474, -0.454, -0.430, -0.419, -0.395, -0.376, -0.355, -0.342, -0.330, -0.314, -0.304, -0.284,
    -0.272, -0.261, -0.251, -0.245, -0.228, -0.220, -0.202, -0.190, -0.179, -0.171, -0.157, -0.145,
    -0.133, -0.124, -0.117, -0.105, -0.092, -0.083, -0.075, -0.064, -0.058, -0.049, -0.041, -0.033,
    -0.023, -0.015, -0.002, 0.005, 0.014, 0.025, 0.029, 0.038, 0.046, 0.054, 0.062, 0.074, 0.083,
    0.096, 0.107, 0.117, 0.130, 0.142, 0.151, 0.164, 0.173, 0.187, 0.198, 0.211, 0.226, 0.241,
    0.248, 0.269, 0.277, 0.292, 0.303, 0.313, 0.326, 0.340, 0.353, 0.369, 0.387, 0.408, 0.432,
    0.471, 0.491, 0.528, 0.560, 0.590, 0.630, 0.680, 0.719, 0.795, 0.868, 0.962, 1.161,
)


def sample_pace_errors(rng: np.random.Generator, size: tuple[int, ...], pre_race: bool) -> np.ndarray:
    table = np.array(PACE_ERROR_PERCENTILES_PRE_RACE if pre_race else PACE_ERROR_PERCENTILES_IN_RACE)
    u = rng.uniform(0.01, 0.99, size=size)
    return np.interp(u, np.arange(1, 100) / 100, table)


# Safety cars. The Safety Car model predicts P(a safety car is active in the
# next WINDOW_LAPS laps); converting that straight to an independent per-lap
# hazard counts every lap of a multi-lap caution as a fresh chance to start
# one. It put a safety car in 78% of 2024-25 races against 46% actual —
# a Brier score (0.334) worse than always guessing the base rate. Now: a
# caution STARTS with the converted hazard times SC_ONSET_SCALE — fitted so
# the simulated share of races with a safety car matches 2018-2023's 60.5%
# — and then runs SC_PERIOD_LAPS laps (the measured median, 135 periods).
# On 2024-25 that scores Brier 0.251, better than the base rate's 0.270;
# it still over-predicts those two seasons (59% vs 46%), which simply had
# fewer cautions than the years it was fitted on.
SC_ONSET_SCALE = 0.60
SC_PERIOD_LAPS = 5


# Track position (round 7). A car within DIRTY_AIR_GAP_SECONDS of the car
# ahead at the start of a lap loses DIRTY_AIR_LOSS_SECONDS on it — measured,
# the same driver's median pace within 1.0 s of a car ahead minus in clean
# air (> 2 s), 1,401 driver-races, 2022-2025. It passes only if it's
# quicker than that car by the circuit's passing delta on the lap;
# otherwise it's held FOLLOW_GAP_SECONDS behind. The delta scales with how
# hard the circuit is to pass at, via
# gold.circuit_history.historical_overtaking_rate (on-track position
# changes per green lap: Monaco 0.073, Las Vegas 0.377).
# PASSING_DELTA_BASE_SECONDS is calibrated against real 2025 finishes, both
# pre-race (race_plan/grid_sensitivity.py, every starter from their real
# grid slot) and in-race (strategy_engine/validate_in_race.py, every
# finisher from 40% distance on their actual strategy). In-race, dry races,
# expected-finish MAE / Spearman: 0.35 s -> 2.05 / 0.863, 0.6 s -> 2.01 /
# 0.874, 1.0 s -> 1.97 / 0.872. Flat between 0.6 and 1.0 on both checks;
# 0.6 has the best rank correlation.
DIRTY_AIR_GAP_SECONDS = 1.0
DIRTY_AIR_LOSS_SECONDS = 0.142
FOLLOW_GAP_SECONDS = 0.5
# Safety-car bunching: tried and switched OFF on evidence. Compressing
# everyone behind the leader under a caution is what happens on track, but
# against real 2025 finishes (strategy_engine/validate_in_race.py, dry
# races) it made in-race predictions worse — expected-finish MAE 2.16 vs
# 2.01, Spearman 0.857 vs 0.874, P(win) Brier 0.042 vs 0.037. Kept behind
# this switch so the result can be re-checked, not silently deleted.
SC_BUNCHES_FIELD = False
SC_BUNCH_GAP_SECONDS = 0.8  # spacing behind the leader when bunching is on
MEDIAN_OVERTAKING_RATE = 0.17
PASSING_DELTA_BASE_SECONDS = 0.6
PASSING_DELTA_BOUNDS = (0.15, 4.0)


@dataclass
class SimulationResult:
    strategy: Strategy
    final_positions: np.ndarray  # shape (n_simulations,), 1-indexed
    safety_car_occurred: np.ndarray  # shape (n_simulations,), bool


@dataclass
class SharedContext:
    """Randomness computed once per race state and reused across every
    candidate strategy so they're all compared under identical safety-car,
    noise, and rival-retirement conditions (see module docstring on why
    sharing this matters, and why it deliberately carries no reference
    trajectory).
    """

    n_laps: int
    sc_occurs: np.ndarray  # shape (n_simulations, n_laps)
    noise: np.ndarray  # shape (n_simulations, n_laps, 1 + n_rivals); column 0 is our car
    rival_retire_lap: np.ndarray  # shape (n_simulations, n_rivals), lap index they retire on; n_laps = never
    pace_error: np.ndarray  # shape (n_simulations,), seconds per lap — see build_shared_context
    rival_pace_error: np.ndarray  # shape (n_simulations, n_rivals), same units

    @property
    def rival_retires(self) -> np.ndarray:
        return self.rival_retire_lap < self.n_laps


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


def _deterministic_tyre_plan(state: RaceState, strategy: Strategy) -> list[dict]:
    """One entry per remaining lap: the compound/tyre_age/stint_number/
    degradation trend that lap runs under — fixed by the strategy, and the
    input the LSTM oracle (lstm_oracle.py) needs to score the whole
    remaining race in one forward pass.

    Note the `pd.isna` check rather than `x or default`: `float('nan') or
    default` evaluates to `nan`, not `default` — NaN is truthy in Python —
    which silently let a real NaN (e.g. the race leader's gap_to_car_ahead,
    or a driver debuting on a fresh compound with no degradation_rate yet)
    slip through a naive fallback and propagate NaN through every later
    lap once accumulated. Found via this exact symptom: predict_trajectory
    returning an all-NaN array end to end.
    """
    pit_map = dict(strategy.pit_plan)
    compound = state.compound
    tyre_age = state.tyre_age
    stint_number = state.stint_number
    degradation_rate = (
        state.degradation_rate
        if state.degradation_rate is not None and not pd.isna(state.degradation_rate)
        else typical_degradation_rate(state.circuit_id, compound)
    )
    grip_estimate = state.grip_estimate if state.grip_estimate is not None and not pd.isna(state.grip_estimate) else 0.0

    plan = []
    for lap in range(state.current_lap + 1, state.race_total_laps + 1):
        is_pit_lap = lap in pit_map
        if is_pit_lap:
            compound = pit_map[lap]
            tyre_age = 0.0
            stint_number += 1
            degradation_rate = typical_degradation_rate(state.circuit_id, compound)
            grip_estimate = 0.0
        elif tyre_age < MAX_TYRE_AGE_FOR_SIMULATION:
            tyre_age += 1
            grip_estimate += degradation_rate

        plan.append(
            {
                "lap_number": lap,
                "compound": compound,
                "tyre_age": tyre_age,
                "stint_number": stint_number,
                "degradation_rate": degradation_rate,
                "grip_estimate": grip_estimate,
                "is_pit_lap": is_pit_lap,
            }
        )
    return plan


def build_shared_context(
    state: RaceState,
    rivals: list[RivalTrend],
    n_simulations: int,
    rng: np.random.Generator,
    pre_race: bool = False,
) -> SharedContext:
    """Computed once per race state, reused for every candidate strategy —
    the safety-car draws, lap-time noise, retirement draws and pace-estimate
    errors, shared so every strategy is evaluated under identical random
    conditions (common random numbers).

    `pre_race` (race_plan/) draws pace errors from the pre-race
    distribution: nothing about any car's pace today has been seen yet.
    """
    n_laps = state.race_total_laps - state.current_lap
    n_rivals = len(rivals)
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

    # A caution starts (see SC_ONSET_SCALE) and then runs SC_PERIOD_LAPS laps.
    onsets = rng.random((n_simulations, n_laps)) < (SC_ONSET_SCALE * sc_probs)[None, :]
    running = np.cumsum(onsets, axis=1)
    running[:, SC_PERIOD_LAPS:] -= running[:, :-SC_PERIOD_LAPS].copy()
    sc_occurs = running > 0
    noise = rng.normal(0, LAP_NOISE_SECONDS, size=(n_simulations, n_laps, 1 + n_rivals)).astype(np.float32)

    # A rival's chance of retiring somewhere in the *remaining* laps, from
    # this circuit's historical full-race DNF rate via the same "at least
    # one event over a window" conversion used for the safety-car model
    # above. The retirement LAP is drawn uniformly over the remaining
    # distance: in a lap-by-lap race a retirement matters from the lap it
    # happens, not just at the flag. Our own retirement isn't modelled —
    # the engine recommends a strategy assuming this driver finishes.
    p_dnf_full_race = (
        state.historical_dnf_rate
        if state.historical_dnf_rate is not None and not pd.isna(state.historical_dnf_rate)
        else DEFAULT_DNF_RATE
    )
    p_dnf_remaining = 1 - (1 - p_dnf_full_race) ** (n_laps / state.race_total_laps)
    retires = rng.random((n_simulations, n_rivals)) < p_dnf_remaining
    retire_lap = np.where(retires, rng.integers(0, max(n_laps, 1), size=(n_simulations, n_rivals)), n_laps)

    # One pace-estimate error per simulation per car, drawn here so every
    # candidate strategy is judged under the SAME draw. It multiplies the
    # whole remaining distance and so dominates the spread — correctly: you
    # don't know a car's pace to better than a few tenths a lap. Rivals
    # carry the same uncertainty as our car (round 6's lesson: an estimate
    # this driver is uncertain about is equally uncertain for everyone).
    errors = sample_pace_errors(rng, (n_simulations, 1 + n_rivals), pre_race)

    return SharedContext(
        n_laps=n_laps,
        sc_occurs=sc_occurs,
        noise=noise,
        rival_retire_lap=retire_lap,
        pace_error=errors[:, 0],
        rival_pace_error=errors[:, 1:],
    )


def passing_delta_seconds(overtaking_rate: float | None) -> float:
    """How much quicker than the car ahead a car must be on a lap to get
    past — larger where passing is historically rare."""
    if overtaking_rate is None or not np.isfinite(overtaking_rate) or overtaking_rate <= 0:
        overtaking_rate = MEDIAN_OVERTAKING_RATE
    delta = PASSING_DELTA_BASE_SECONDS * (MEDIAN_OVERTAKING_RATE / overtaking_rate)
    return float(np.clip(delta, *PASSING_DELTA_BOUNDS))


def run_race(
    start_times: np.ndarray,
    pace_per_lap: np.ndarray,
    noise: np.ndarray,
    pit_loss: np.ndarray,
    retire_lap: np.ndarray,
    sc_occurs: np.ndarray,
    passing_delta: float,
) -> np.ndarray:
    """Run the remaining race lap by lap for every simulation at once, and
    return each car's final race time (np.inf for a retirement).

    start_times   (n_cars,)                   gap to the leader now
    pace_per_lap  (n_sims, n_cars)            seconds/lap vs the field average
    noise         (n_sims, n_laps, n_cars)    lap-to-lap variation
    pit_loss      (n_sims, n_laps, n_cars)    time lost on each car's pit laps
    retire_lap    (n_sims, n_cars)            lap index a car retires on; n_laps = never
    sc_occurs     (n_sims, n_laps)            safety car on that lap
    """
    n_sims, n_laps, n_cars = noise.shape
    times = np.broadcast_to(start_times.astype(np.float64), (n_sims, n_cars)).copy()
    rows = np.arange(n_sims)[:, None]

    for lap in range(n_laps):
        active = retire_lap > lap
        # Road order at the start of the lap; retired cars sort to the back.
        order = np.argsort(np.where(active, times, np.inf), axis=1, kind="stable")
        start = times[rows, order]
        proposed = start + (pace_per_lap + noise[:, lap, :] + pit_loss[:, lap, :])[rows, order]
        live = active[rows, order]

        # Dirty air: following closely at the start of the lap costs time.
        close = np.zeros_like(live)
        # Retired cars sit at +inf; inf - inf is NaN, so compare only
        # between cars that are both still running.
        both_live = live[:, 1:] & live[:, :-1]
        gaps = np.where(both_live, start[:, 1:] - np.where(both_live, start[:, :-1], 0.0), np.inf)
        close[:, 1:] = gaps < DIRTY_AIR_GAP_SECONDS
        proposed = proposed + np.where(close & live, DIRTY_AIR_LOSS_SECONDS, 0.0)

        # Held up: working front to back, a car that would end the lap
        # within the follow gap of the rearmost car ahead of it — without
        # being quicker by the passing delta — is held behind it. A car in
        # the pits this lap has a huge proposed time and drops back freely.
        rear = proposed[:, 0].copy()
        for k in range(1, n_cars):
            mine = proposed[:, k]
            held = live[:, k] & (mine >= rear - passing_delta) & (mine < rear + FOLLOW_GAP_SECONDS)
            mine = np.where(held, rear + FOLLOW_GAP_SECONDS, mine)
            proposed[:, k] = mine
            rear = np.where(live[:, k], np.maximum(rear, mine), rear)

        # A safety car bunches everyone still running behind the leader,
        # keeping the order. (Its slowing of every car equally doesn't move
        # anyone relative to anyone else, so it isn't added as time.)
        sc = sc_occurs[:, lap]
        if SC_BUNCHES_FIELD and sc.any():
            leader = proposed[:, :1]
            bunched = leader + np.arange(n_cars)[None, :] * SC_BUNCH_GAP_SECONDS
            proposed = np.where(sc[:, None] & live, np.minimum(proposed, bunched), proposed)

        times[rows, order] = np.where(live, proposed, np.inf)

    return times


def simulate_strategy(
    state: RaceState,
    strategy: Strategy,
    rivals: list[RivalTrend],
    shared: SharedContext,
    pace_deviation_override: float | None = None,
) -> SimulationResult:
    """`pace_deviation_override` is our car's per-lap pace edge. Every
    production caller passes it: in-race, engine.candidate_pace_overrides
    (recent pace made tyre-age-neutral plus the strategy's measured tyre
    cost, the same treatment rivals get in tyre_pace.tyre_adjusted_rivals);
    pre-race, race_plan's season-form anchor. Left None, it falls back to
    the LSTM's absolute pace — kept for direct experiments only, since that
    is the two-estimator asymmetry strategy_engine/tyre_pace.py explains.
    """
    n_simulations = shared.sc_occurs.shape[0]
    n_laps = shared.n_laps
    n_cars = 1 + len(rivals)
    is_pit_lap = _deterministic_pit_plan(state, strategy)
    pit_loss_s = typical_pit_loss_seconds(state.circuit_id)

    # This strategy's own stops, discounted when one lands under a simulated
    # safety car — PRD 12.6's "pitting under a safety car is free track
    # position".
    pit_loss = np.zeros((n_simulations, n_laps, n_cars), dtype=np.float32)
    for i, scheduled in enumerate(is_pit_lap):
        if scheduled:
            pit_loss[:, i, 0] = pit_loss_s * np.where(shared.sc_occurs[:, i], SC_PIT_DISCOUNT, 1.0)

    # A rival's own strategy isn't simulated (see field.py), but it's
    # charged the stops it would need for no stint to outrun the circuit's
    # typical stint length (tyre_baselines.rival_stop_laps) — first a field
    # that never stopped, then one that stopped at most once, both
    # penalised every strategy of ours that pits.
    for j, rival in enumerate(rivals, start=1):
        for stop in rival_stop_laps(state.circuit_id, rival.compound, rival.tyre_age, n_laps):
            pit_loss[:, stop, j] = pit_loss_s

    if pace_deviation_override is not None:
        pace_deviation_per_lap = float(pace_deviation_override)
    else:
        tyre_plan = _deterministic_tyre_plan(state, strategy)
        green_times, _sc_times = predict_trajectory(state, tyre_plan)
        pace_deviation_per_lap = float(green_times.mean() - state.field_avg_lap_time_seconds)
    # A NaN pace edge is not survivable: `field_avg_lap_time_seconds` is
    # genuinely NaN on lap 1 (FastF1 records no standing-start lap time), and
    # a NaN anywhere here used to compare False against every rival and
    # score this driver P1 in every simulation. Unknown edge = no edge.
    if not np.isfinite(pace_deviation_per_lap):
        pace_deviation_per_lap = 0.0
    # A wet-race snapshot once produced a ~1.7 s/lap sustained edge from a
    # covariate mix sparse in training; clipped to a generous but finite
    # per-lap bound (see the module docstring, round 5).
    pace_deviation_per_lap = float(np.clip(pace_deviation_per_lap, -PACE_DEVIATION_CAP_PER_LAP, PACE_DEVIATION_CAP_PER_LAP))

    # Round 6: both sides project their pace across the full remaining race,
    # each with its share of the same pace-estimate uncertainty.
    pace_per_lap = np.empty((n_simulations, n_cars))
    pace_per_lap[:, 0] = pace_deviation_per_lap + shared.pace_error
    for j, rival in enumerate(rivals, start=1):
        trend = rival.recent_pace_delta if np.isfinite(rival.recent_pace_delta) else 0.0
        pace_per_lap[:, j] = trend + shared.rival_pace_error[:, j - 1]

    start_times = np.array([state.gap_to_leader] + [r.gap_to_leader for r in rivals], dtype=float)
    start_times = np.where(np.isfinite(start_times), start_times, np.nanmax(start_times[np.isfinite(start_times)], initial=0.0))
    retire_lap = np.concatenate([np.full((n_simulations, 1), n_laps), shared.rival_retire_lap], axis=1)

    final_times = run_race(
        start_times,
        pace_per_lap,
        shared.noise,
        pit_loss,
        retire_lap,
        shared.sc_occurs,
        passing_delta_seconds(state.historical_overtaking_rate),
    )
    ours = final_times[:, :1]
    final_position = 1 + (final_times[:, 1:] < ours).sum(axis=1)

    return SimulationResult(
        strategy=strategy,
        final_positions=final_position,
        safety_car_occurred=shared.sc_occurs.any(axis=1),
    )
