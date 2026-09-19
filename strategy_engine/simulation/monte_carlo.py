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
from strategy_engine.tyre_baselines import typical_degradation_rate, typical_max_stint_length

LAP_TIME_NOISE_SECONDS = 1.2  # grounded in the Lap Time model's own measured stable-regime RMSE (~1.07s)
RIVAL_PACE_NOISE_PER_LAP = 0.5
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

# How wrong a pace estimate typically is, in seconds per lap — the piece
# this simulation was missing entirely, and the reason PACE_DEVIATION_CAP_PER_LAP
# had to exist as a band-aid.
#
# Both figures are measured on this project's own data, not assumed:
#   IN_RACE   std of (a driver's actual pace over the remaining laps) minus
#             (their observed pace over the first 20), across 3,379
#             driver-races. Essentially unbiased (mean -0.010).
#   PRE_RACE  std of (their actual race pace) minus (their season-to-date
#             average before that race), across 3,060 driver-races. Also
#             unbiased (mean +0.008).
#
# Observing twenty laps barely narrows it, which is itself the finding:
# race pace genuinely moves around with fuel, tyres, traffic and track
# evolution. Treating the estimate as exact is what let a 0.5s/lap edge
# compound into a guaranteed win over a full race distance.
PACE_ESTIMATE_UNCERTAINTY_IN_RACE = 0.634
PACE_ESTIMATE_UNCERTAINTY_PRE_RACE = 0.737

# Track position is sticky: being quicker is not the same as getting past.
# Ranking purely on projected time gaps assumes free overtaking, which made
# a grid slot worth almost nothing — pole and P5 planned to the same
# finishing position, because 5.8s of starting advantage is noise against a
# race-long pace spread.
#
# A rival is therefore only passed if the time advantage exceeds a margin
# that scales with how hard this circuit actually is to overtake at, using
# gold.circuit_history.historical_overtaking_rate (measured on-track
# position changes per green lap: Monaco 0.073, Las Vegas 0.377).
# OVERTAKE_MARGIN_BASE_SECONDS is calibrated so simulated grid-to-flag
# movement matches the 2.85 positions actually observed across this
# project's data — see car_profiles/... no: see the calibration in
# race_plan/calibrate_overtaking.py.
MEDIAN_OVERTAKING_RATE = 0.17
OVERTAKE_MARGIN_BASE_SECONDS = 20.0


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
    noise: np.ndarray  # shape (n_simulations, n_laps)
    rival_retires: np.ndarray  # shape (n_simulations, n_rivals), bool — see build_shared_context
    pace_error: np.ndarray  # shape (n_simulations,), seconds per lap — see build_shared_context
    rival_pace_error: np.ndarray  # shape (n_simulations, n_rivals), same units


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
    pace_uncertainty: float = PACE_ESTIMATE_UNCERTAINTY_IN_RACE,
) -> SharedContext:
    """Computed once per race state, reused for every candidate strategy —
    the safety-car draws, lap-time noise, and rival-retirement draws,
    shared so every strategy is evaluated under identical random
    conditions (common random numbers).
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

    # A rival's chance of retiring somewhere in the *remaining* laps, from
    # this circuit's historical full-race DNF rate via the same "at least
    # one event over a window" conversion used for the safety-car model
    # above (p_full_race is the "window", n_laps/race_total_laps of it is
    # the fraction still ahead of us). One draw per (simulation, rival),
    # identical across every candidate strategy — a rival's real-world
    # retirement doesn't depend on what tyre strategy WE choose. See
    # module docstring for why this exists: without it, a driver who is
    # already last has no modeled chance of inheriting a place.
    p_dnf_full_race = (
        state.historical_dnf_rate
        if state.historical_dnf_rate is not None and not pd.isna(state.historical_dnf_rate)
        else DEFAULT_DNF_RATE
    )
    p_dnf_remaining = 1 - (1 - p_dnf_full_race) ** (n_laps / state.race_total_laps)
    rival_retires = rng.random((n_simulations, len(rivals))) < p_dnf_remaining

    # One pace-estimate error per simulation, drawn here rather than inside
    # simulate_strategy so every candidate strategy is judged under the SAME
    # draw — the same common-random-numbers discipline the safety car and
    # retirement draws already follow. Without that, two candidates would be
    # compared partly on which got the luckier view of the car's pace.
    #
    # It multiplies the whole remaining distance, so it dominates: at the
    # pre-race figure over 57 laps it contributes ~42s of spread against
    # ~9s from lap-to-lap noise. That is the correct order of magnitude —
    # before a race you genuinely do not know a car's pace to better than
    # a few tenths a lap, and the simulation now says so instead of
    # reporting 100% win probabilities.
    pace_error = rng.normal(0.0, pace_uncertainty, size=n_simulations)

    # Rivals get the same per-lap pace uncertainty, for the same reason
    # round 6 made the projection horizons match: an estimate this driver
    # is uncertain about is equally uncertain for everyone else. Previously
    # only RIVAL_PACE_NOISE_PER_LAP applied to them, which over a race is
    # ~3.8s against this driver's ~42s — an 11x asymmetry that made a
    # finishing position mostly a lottery on our own pace draw and left the
    # grid, worth ~16s from pole to P16, unable to compete with it.
    rival_pace_error = rng.normal(0.0, pace_uncertainty, size=(n_simulations, max(len(rivals), 1)))

    return SharedContext(
        n_laps=n_laps,
        sc_occurs=sc_occurs,
        noise=noise,
        rival_retires=rival_retires,
        pace_error=pace_error,
        rival_pace_error=rival_pace_error,
    )


def simulate_strategy(
    state: RaceState,
    strategy: Strategy,
    rivals: list[RivalTrend],
    shared: SharedContext,
    pace_deviation_override: float | None = None,
) -> SimulationResult:
    """`pace_deviation_override` replaces this driver's per-lap pace edge
    instead of deriving it from the LSTM. race_plan/ uses it so that both
    sides of the comparison are estimated the same way: rivals there are
    projected from season form, and scoring our driver from the LSTM while
    scoring rivals from season form is exactly the two-estimator asymmetry
    that round 6 (above) was about. Left None in-race, where the LSTM and
    the rivals' live pace trends are both measured off the same session.
    """
    n_simulations = shared.sc_occurs.shape[0]
    is_pit_lap = _deterministic_pit_plan(state, strategy)
    pit_loss = typical_pit_loss_seconds(state.circuit_id)

    # A safety car slows the whole field by roughly the same amount — the
    # leader is under the same caution as everyone else — so it doesn't
    # move this driver's *gap to the leader* on its own. Only three things
    # legitimately do that here: this strategy's own pit stop cost (only
    # this driver stops, discounted if it lands under a simulated SC — PRD
    # 12.6's "pitting under a safety car is free track position"),
    # idiosyncratic lap-to-lap noise, and — unlike every earlier attempt in
    # this module's history — this strategy's actual predicted green-flag
    # pace vs. the field average, now that the LSTM oracle
    # (lstm_oracle.py) can produce that trajectory in one forward pass
    # with no autoregressive chaining to go unstable. See the module
    # docstring for the four rounds of bugs that came from trying this
    # with a chained single-step model instead.
    pit_time_loss = np.zeros((n_simulations, shared.n_laps))
    for i, scheduled in enumerate(is_pit_lap):
        if scheduled:
            discount = np.where(shared.sc_occurs[:, i], SC_PIT_DISCOUNT, 1.0)
            pit_time_loss[:, i] = pit_loss * discount

    if pace_deviation_override is not None:
        pace_deviation_per_lap = float(pace_deviation_override)
    else:
        tyre_plan = _deterministic_tyre_plan(state, strategy)
        green_times, _sc_times = predict_trajectory(state, tyre_plan)
        pace_deviation_per_lap = float(green_times.mean() - state.field_avg_lap_time_seconds)
    # A wet-race snapshot (long INTERMEDIATE stint, track drying) surfaced a
    # prediction implying ~1.7s/lap of sustained improvement — physically
    # not impossible on a drying track, but a large enough claim, for a
    # combination of covariates sparse enough in training, that it
    # deserves a skeptical bound rather than blind trust: at N=5,000 sims
    # it made an actual race leader's win probability a flat, noise-proof
    # 100%. Clipped to a generous per-lap bound that's still well above the
    # ~0.5-1s/lap differences seen between reasonable candidate strategies
    # in normal (dry, in-distribution) scenarios.
    # A NaN here is not survivable and must never reach the ranking below.
    # `field_avg_lap_time_seconds` is genuinely NaN on lap 1 of a race —
    # FastF1 records no lap time for the standing-start lap, so the whole
    # field average for that lap is undefined — and any arithmetic on it
    # stays NaN all the way to `our_final_gap`. Every NaN comparison
    # evaluates False, so `(rival_gaps < our_gap).sum()` counts zero rivals
    # ahead and the driver is scored P1 in *every* simulation: a 100% win
    # probability generated purely by missing data. Treat an unknown pace
    # edge as no edge, which is the neutral assumption, not the flattering
    # one.
    if not np.isfinite(pace_deviation_per_lap):
        pace_deviation_per_lap = 0.0
    pace_deviation_per_lap = float(np.clip(pace_deviation_per_lap, -PACE_DEVIATION_CAP_PER_LAP, PACE_DEVIATION_CAP_PER_LAP))

    # **Round 6 — the horizons on the two sides of the comparison have to
    # match.** They didn't: this driver's pace deviation was projected
    # across every remaining lap, while each rival's was capped at
    # RIVAL_TREND_HORIZON_LAPS. Finishing position depends only on the
    # *relative* gap, so an asymmetric horizon hands whichever car is
    # fastest relative to the field an advantage no rival is ever allowed
    # to answer.
    #
    # It stayed hidden while the LSTM under-predicted pace deltas, and
    # surfaced when retraining made it MORE accurate: at Bahrain 2025 lap
    # 20 the model put the leader at -1.145s/lap versus the field (his
    # actual lap that moment was -1.007, so the estimate was sound), which
    # over 37 laps compounded into a 42-second gain — larger than the
    # entire P1-to-P20 spread of ~39s. Every candidate therefore "won"
    # ~96% of the time against a Win Probability classifier saying 24%.
    # The cross-checks in recommendation/reasoning.py are what caught it.
    #
    # Both sides now project across the full remaining race. The
    # alternative — capping both at RIVAL_TREND_HORIZON_LAPS — is equally
    # symmetric but measurably worse: it truncates the pace signal at 10
    # laps while noise keeps accumulating over all 37 (variance grows with
    # every lap run, so that part can't be capped), which buried the same
    # leader at 6.8% win and an expected P6.5. Checked against the two
    # independently-trained classifiers on three real drivers spanning the
    # front, midfield and back of the grid, full-race projection agreed
    # far better (mean win-probability disagreement 3.1pp vs 6.5pp), and
    # it's the more defensible assumption physically: a genuinely faster
    # car stays faster, where mean-reverting pace to the field average
    # after ten laps claims the field converges mid-race, which it doesn't.
    # Per simulation, not a single number: the point estimate plus that
    # simulation's share of how wrong such estimates typically are.
    pace_deviation = (pace_deviation_per_lap + shared.pace_error) * shared.n_laps

    cumulative_deviation = pace_deviation + (shared.noise + pit_time_loss).sum(axis=1)
    our_final_gap = state.gap_to_leader + cumulative_deviation

    rng = np.random.default_rng()  # rival noise doesn't need to be paired across strategies

    def _rival_owed_pit_loss(rival: RivalTrend) -> float:
        # A rival's own future strategy isn't simulated (see field.py), which
        # previously meant every rival was implicitly modeled as "never pits
        # again" for the rest of the race — a real bias found via the Win
        # Probability model cross-check (recommendation/reasoning.py): a
        # race leader whose recommended strategy involves a stop was
        # charged the full pit-loss cost while rivals on comparably worn
        # tyres were charged nothing for the stop they'd also need. If this
        # rival's current tyre age would exceed a realistic stint length
        # for their compound at this circuit before the race ends, charge
        # them one pit stop's worth of time too — undiscounted for a
        # simulated safety car, since we have no way of knowing when in
        # their own (unmodeled) strategy that stop would land relative to
        # this simulation's SC draws.
        laps_until_needed = typical_max_stint_length(state.circuit_id, rival.compound) - rival.tyre_age
        return pit_loss if laps_until_needed < shared.n_laps else 0.0

    rival_final_gaps = np.stack(
        [
            rival.gap_to_leader
            # Same horizon as this driver's own pace_deviation above — see
            # round 6 there for why the two must match.
            + shared.n_laps * rival.recent_pace_delta
            + _rival_owed_pit_loss(rival)
            + rng.normal(0, RIVAL_PACE_NOISE_PER_LAP * np.sqrt(shared.n_laps), size=n_simulations)
            for rival in rivals
        ],
        axis=0,
    )  # shape (n_rivals, n_simulations)

    # Their share of the same pace-estimate uncertainty this driver carries.
    if rivals:
        rival_final_gaps = rival_final_gaps + shared.rival_pace_error[:, : len(rivals)].T * shared.n_laps

    # A retired rival (shared.rival_retires, drawn once per race state and
    # reused for every candidate — see build_shared_context) no longer
    # finishes ahead of anyone, regardless of the gap they were projected
    # to hold. shape (n_simulations, n_rivals) -> (n_rivals, n_simulations)
    # to match rival_final_gaps.
    rival_final_gaps = np.where(shared.rival_retires.T, np.inf, rival_final_gaps)

    # Belt and braces after the NaN guard above: a non-finite gap on either
    # side would otherwise compare False and silently flatter this driver.
    rival_final_gaps = np.where(np.isfinite(rival_final_gaps), rival_final_gaps, np.inf)
    our_final_gap = np.where(np.isfinite(our_final_gap), our_final_gap, np.inf)

    # Track position is sticky (see OVERTAKE_MARGIN_BASE_SECONDS). A rival
    # who is ahead keeps the place unless this driver beats them by the
    # circuit's passing margin; one who is behind only gets by if THEY
    # clear it. At Monaco that margin is large enough that the grid order
    # largely survives, which is the whole point.
    overtaking_rate = state.historical_overtaking_rate
    if overtaking_rate is None or not np.isfinite(overtaking_rate) or overtaking_rate <= 0:
        overtaking_rate = MEDIAN_OVERTAKING_RATE
    passing_margin = OVERTAKE_MARGIN_BASE_SECONDS * (MEDIAN_OVERTAKING_RATE / overtaking_rate)

    started_ahead = np.array(
        [rival.gap_to_leader < state.gap_to_leader for rival in rivals], dtype=bool
    )[:, None]
    threshold = np.where(started_ahead, our_final_gap[None, :] + passing_margin, our_final_gap[None, :] - passing_margin)

    final_position = 1 + (rival_final_gaps < threshold).sum(axis=0)

    return SimulationResult(
        strategy=strategy,
        final_positions=final_position,
        safety_car_occurred=shared.sc_occurs.any(axis=1),
    )
