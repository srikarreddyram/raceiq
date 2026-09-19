"""Human-readable reasoning and the final recommendation payload — PRD
Section 12.6/12.7.

Every bullet here is generated from a value the engine actually computed
— nothing is templated filler. A statistic that isn't available (no
circuit history yet, no rival ahead) simply produces no bullet about it,
rather than a generic placeholder sentence.
"""

from __future__ import annotations

from strategy_engine.oracles import (
    predict_expected_finish_now,
    predict_remaining_tyre_life,
    predict_win_probability_now,
)
from strategy_engine.scoring.score import StrategyScore
from strategy_engine.state import RaceState

# Large gaps between a classifier's static estimate and the simulation's
# own output for the recommended strategy are worth flagging rather than
# silently ignoring — see predict_win_probability_now/predict_expected_finish_now's
# docstrings. Position threshold is in finishing places, not a fraction.
_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD = 0.15
_EXPECTED_FINISH_DISAGREEMENT_THRESHOLD = 3.0


def build_reasoning(
    state: RaceState,
    top: StrategyScore,
    alternative: StrategyScore | None,
    static_win_probability: float,
    static_expected_finish: float,
) -> list[str]:
    bullets: list[str] = []

    remaining_life = predict_remaining_tyre_life(state)
    bullets.append(
        f"Current {state.compound} tyres (age {state.tyre_age:.0f} laps) predicted to have "
        f"~{remaining_life:.0f} laps of life left."
    )

    if state.historical_sc_rate is not None:
        bullets.append(
            f"Safety car has appeared in {state.historical_sc_rate * 100:.0f}% of prior races at this circuit."
        )

    if state.condition_delta is not None:
        direction = "above" if state.condition_delta > 0 else "below"
        bullets.append(
            f"Track temperature is {abs(state.condition_delta):.1f}C {direction} this circuit's historical average."
        )

    if state.rival_ahead is not None and state.gap_to_car_ahead is not None:
        bullets.append(
            f"Gap to car ahead ({state.rival_ahead.driver_id}, {state.rival_ahead.compound} "
            f"tyres, age {state.rival_ahead.tyre_age:.0f}): {state.gap_to_car_ahead:.1f}s."
        )

    bullets.append(
        f"{top.label}: {top.win_probability * 100:.0f}% win, {top.podium_probability * 100:.0f}% podium, "
        f"expected P{top.expected_finish:.1f} across simulations "
        f"({top.safety_car_encounter_rate * 100:.0f}% of runs encountered a safety car)."
    )

    win_disagreement = abs(static_win_probability - top.win_probability)
    if win_disagreement >= _WIN_PROBABILITY_DISAGREEMENT_THRESHOLD:
        bullets.append(
            f"Note: the Win Probability model's independent estimate for this driver's current position "
            f"is {static_win_probability * 100:.0f}%, notably different from the simulation's "
            f"{top.win_probability * 100:.0f}% for the recommended strategy — worth a sanity check before acting."
        )

    finish_disagreement = abs(static_expected_finish - top.expected_finish)
    if finish_disagreement >= _EXPECTED_FINISH_DISAGREEMENT_THRESHOLD:
        bullets.append(
            f"Note: the Final Race Position model's independent estimate for this driver's current position "
            f"is P{static_expected_finish:.1f}, notably different from the simulation's P{top.expected_finish:.1f} "
            f"for the recommended strategy — worth a sanity check before acting."
        )

    if alternative is not None:
        bullets.append(
            f"Alternative — {alternative.label}: lower risk (variance {alternative.risk_score:.1f} vs "
            f"{top.risk_score:.1f}) but {alternative.expected_points:.1f} expected points vs {top.expected_points:.1f}."
        )

    return bullets


def build_recommendation(
    state: RaceState, ranked: list[StrategyScore], n_simulations: int
) -> dict:
    top = ranked[0]
    alternatives = ranked[1:4]
    static_win_probability = predict_win_probability_now(state)
    static_expected_finish = predict_expected_finish_now(state)

    return {
        "circuit_id": state.circuit_id,
        "driver_id": state.driver_id,
        "team_id": state.team_id,
        "current_lap": state.current_lap,
        "laps_remaining": state.laps_remaining,
        "n_simulations": n_simulations,
        "circuit_context": {
            "historical_sc_rate": state.historical_sc_rate,
            "condition_delta": state.condition_delta,
        },
        "model_cross_checks": {
            # The Win Probability and Final Race Position classifiers' own
            # estimates for the driver's actual current state, independent
            # of the simulation below — see oracles.py's docstrings.
            "win_probability_model_estimate": static_win_probability,
            "expected_finish_model_estimate": static_expected_finish,
        },
        "recommended_strategy": {
            "action": top.label,
            "pit_plan": [list(stop) for stop in top.pit_plan],
            "win_probability": top.win_probability,
            "podium_probability": top.podium_probability,
            "points_probability": top.points_probability,
            "expected_finish": top.expected_finish,
            "expected_points": top.expected_points,
            "risk_score": top.risk_score,
            "strategy_score": top.strategy_score,
            "safety_car_encounter_rate": top.safety_car_encounter_rate,
            "finish_distribution": top.finish_distribution,
        },
        "reasoning": build_reasoning(
            state, top, alternatives[0] if alternatives else None, static_win_probability, static_expected_finish
        ),
        "alternatives": [
            {
                "action": alt.label,
                "pit_plan": [list(stop) for stop in alt.pit_plan],
                "win_probability": alt.win_probability,
                "podium_probability": alt.podium_probability,
                "expected_finish": alt.expected_finish,
                "expected_points": alt.expected_points,
                "risk_score": alt.risk_score,
                "strategy_score": alt.strategy_score,
            }
            for alt in alternatives
        ],
    }
