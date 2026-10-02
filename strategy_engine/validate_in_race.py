"""Does the in-race simulation predict how races actually finish?

For a sample of 2025 races (the test season), every classified driver is
snapshotted at 40% race distance and the simulation is run with the
strategy they ACTUALLY went on to use — so this measures the simulation,
not the strategy search. Their simulated finish is compared with where
they really finished, alongside two independent references:

  stay put      finish = position at the snapshot (the naive baseline)
  classifiers   the Final Race Position model's expected finish and the
                Win Probability model's estimate, from the same snapshot

Reported: MAE of expected finish, per-race Spearman correlation, and the
Brier score of P(win) against who actually won.

The classifier cross-checks in recommendation/reasoning.py are useful
smoke alarms, but they are models too; this is ground truth.

`--oracle-rivals` gives every rival the stops it really went on to make,
instead of the simulation's guess at them. It isn't a fair test — nobody
knows a rival's stops in advance — it's a ceiling: the gap between it and
the normal run is what guessing rivals' strategy costs.

`--season` and `--rounds` pick the races (default: ten rounds of 2025).
Races with rain or a red flag are reported separately, found from the lap
data: both rewrite everyone's strategy in ways no snapshot predicts.

Where it stands (expected-finish MAE in places; "clean" leaves out races
with rain or a red flag):

                         simulation   stay put   oracle rivals
    2025, 10 rounds         1.84        2.27         1.71
    2025, clean (8)         1.79        2.27         1.67
    2026, 15 rounds         2.39        1.88         2.10
    2026, clean (12)        1.97        1.87         1.80

On 2026 — new regulations, and cars that hold position far more after
40% distance than in 2025 — the simulation does not beat "finish where
you are now". Knowing rivals' stops would put it just ahead.

Usage:
    uv run python -m strategy_engine.validate_in_race [--delta 0.6] [--bunch] [--oracle-rivals]
                                                      [--season 2026] [--rounds 1-15]
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.common.data import is_classified, load_race_features
from models.common.db import get_connection
from strategy_engine.search.candidates import Strategy

SEASON = 2025
ROUNDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
SNAPSHOT_FRACTION = 0.4
# Races with rain or a red flag are reported separately (disrupted_races):
# every car's stops follow the weather or the stoppage, which no strategy
# model predicts from a mid-race snapshot. (This used to be a hand-kept
# list holding one race, Canada 2025 — which was dry. Its huge errors came
# from phantom stops in the timing data, since fixed in
# pipelines/silver/laps.py.)
N_SIMULATIONS = 400


def _actual_plan(race_rows: pd.DataFrame, driver_id: str, snapshot_lap: int) -> tuple:
    """The stops a driver actually made after the snapshot, one entry per
    stop: the in-lap, and the compound fitted.

    `is_pit_lap` is set on BOTH the in-lap and the out-lap of every stop
    (13,028 flagged laps for ~6,500 stops), so counting flagged laps
    charges every stop twice — this validation's first run did exactly
    that. A stop is the lap after which the stint number goes up.
    """
    laps = race_rows[race_rows.driver_id == driver_id].sort_values("lap_number")
    nxt_stint = laps.stint_number.shift(-1)
    nxt_compound = laps.compound.shift(-1)
    plan = []
    for row, stint_after, compound in zip(laps.itertuples(), nxt_stint, nxt_compound):
        if row.lap_number > snapshot_lap and pd.notna(stint_after) and stint_after > row.stint_number:
            plan.append((int(row.lap_number), compound if isinstance(compound, str) else "HARD"))
    return tuple(plan)


def disrupted_races(season: int) -> set[str]:
    """Races with rain or a red flag at any point."""
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT race_id FROM gold.lap_features WHERE race_id LIKE ?
            GROUP BY race_id
            HAVING BOOL_OR(COALESCE(rainfall_flag, FALSE)) OR BOOL_OR(COALESCE(red_flag_active, FALSE))
            """,
            [f"{season}_%"],
        ).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def _oracle_rivals(field: list, race_rows: pd.DataFrame, lap: int, total: int, wear: dict, n_simulations: int):
    """Each rival on the stops it really made: its pace over the remaining
    race (age-neutral recent pace plus its real plan's tyre cost) and its
    pit laps, in the shape SharedContext.rival_pit takes."""
    import strategy_engine.simulation.monte_carlo as mc
    from strategy_engine.tyre_pace import age_neutral, tyre_cost

    n_laps = total - lap
    pit = np.zeros((n_simulations, n_laps, len(field)), dtype=bool)
    compounds = np.zeros((n_simulations, len(field), mc.MAX_STINTS), dtype=np.int8)
    rivals = []
    for j, rival in enumerate(field):
        plan = dict(_actual_plan(race_rows, rival.driver_id, lap))
        compound, age, costs = rival.compound, rival.tyre_age, []
        for race_lap in range(lap + 1, total + 1):
            if race_lap in plan:
                compound, age = plan[race_lap], 0.0
                pit[:, race_lap - lap - 1, j] = True
            else:
                age += 1
            costs.append(tyre_cost(compound, age, wear))
        recent = rival.recent_pace_delta if np.isfinite(rival.recent_pace_delta) else 0.0
        pace = age_neutral(recent, rival.compound, rival.tyre_age, wear) + (float(np.mean(costs)) if costs else 0.0)
        rivals.append(replace(rival, recent_pace_delta=pace))
        compounds[:, j, :] = mc._stint_row([rival.compound] + [plan[k] for k in sorted(plan)])
    return rivals, pit, compounds


def run(delta: float | None = None, bunch: bool = False, oracle_rivals: bool = False) -> pd.DataFrame:
    import strategy_engine.simulation.monte_carlo as mc
    from strategy_engine.engine import candidate_pace_overrides
    from strategy_engine.field import build_field_snapshot_from_gold, driver_recent_pace
    from strategy_engine.oracles import predict_expected_finish_now, predict_win_probability_now
    from strategy_engine.scoring.score import score_strategy
    from strategy_engine.state import RaceState
    from strategy_engine.tyre_pace import race_wear, tyre_adjusted_rivals

    if delta is not None:
        mc.PASSING_DELTA_BASE_SECONDS = delta
    mc.SC_BUNCHES_FIELD = bunch

    df = load_race_features()
    con = get_connection()
    try:
        results = con.execute(
            f"SELECT season || '_' || round AS race_id, driver_id, position, status FROM bronze.ergast_results "
            f"WHERE season = {SEASON} AND round IN ({', '.join(map(str, ROUNDS))})"
        ).df()
    finally:
        con.close()
    results["classified"] = results["status"].map(is_classified)
    winners = results[results.position == 1].set_index("race_id").driver_id.to_dict()

    rows = []
    for race_id in sorted(results.race_id.unique(), key=lambda r: int(r.split("_")[1])):
        race_rows = df[df.race_id == race_id]
        total = int(race_rows.lap_number.max())
        lap = int(round(total * SNAPSHOT_FRACTION))
        at_lap = race_rows[race_rows.lap_number == lap].dropna(subset=["current_position", "stint_number", "tyre_age"])
        finishers = results[(results.race_id == race_id) & results.classified].set_index("driver_id").position
        for snap in at_lap.itertuples():
            if snap.driver_id not in finishers.index:
                continue
            try:
                state = RaceState.from_gold_row(race_rows.loc[snap.Index], race_total_laps=total)
                season = int(race_id.split("_")[0])
                wear = race_wear(race_id)
                field = build_field_snapshot_from_gold(race_id, lap, snap.driver_id)
                strategy = Strategy(pit_plan=_actual_plan(race_rows, snap.driver_id, lap), label="actual")
                if oracle_rivals:
                    rivals, rival_pit, rival_compounds = _oracle_rivals(field, race_rows, lap, total, wear, N_SIMULATIONS)
                else:
                    rivals = tyre_adjusted_rivals(field, state, season, wear)
                shared = mc.build_shared_context(state, rivals, N_SIMULATIONS, np.random.default_rng(3), **mc.race_model(season, wear))
                if oracle_rivals:
                    shared = replace(shared, rival_pit=rival_pit, rival_compounds=rival_compounds)
                (pace,) = candidate_pace_overrides(state, [strategy], driver_recent_pace(race_id, lap, snap.driver_id), season, wear)
                scored = score_strategy(mc.simulate_strategy(state, strategy, rivals, shared, pace_deviation_override=pace))
                rows.append(
                    {
                        "race_id": race_id,
                        "driver_id": snap.driver_id,
                        "now": float(state.current_position),
                        "actual": int(finishers[snap.driver_id]),
                        "won": winners.get(race_id) == snap.driver_id,
                        "sim_finish": scored.expected_finish,
                        "sim_win": scored.win_probability,
                        "clf_finish": predict_expected_finish_now(state),
                        "clf_win": predict_win_probability_now(state),
                    }
                )
            except Exception as exc:
                print(f"  skipped {race_id} {snap.driver_id}: {exc}")
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> dict:
    def rho(col: str) -> float:
        return float(np.nanmean([spearmanr(g[col], g["actual"]).statistic for _, g in df.groupby("race_id") if len(g) > 3]))

    out = {
        "n": len(df),
        "mae_sim": float((df.sim_finish - df.actual).abs().mean()),
        "mae_clf": float((df.clf_finish - df.actual).abs().mean()),
        "mae_now": float((df.now - df.actual).abs().mean()),
        "rho_sim": rho("sim_finish"),
        "rho_clf": rho("clf_finish"),
        "rho_now": rho("now"),
        "brier_sim": float(((df.sim_win - df.won) ** 2).mean()),
        "brier_clf": float(((df.clf_win - df.won) ** 2).mean()),
    }
    print(f"driver-snapshots: {out['n']}")
    print(f"expected finish MAE    simulation {out['mae_sim']:.2f}   classifier {out['mae_clf']:.2f}   stay-put {out['mae_now']:.2f}")
    print(f"per-race Spearman      simulation {out['rho_sim']:.3f}  classifier {out['rho_clf']:.3f}  stay-put {out['rho_now']:.3f}")
    print(f"P(win) Brier           simulation {out['brier_sim']:.4f} classifier {out['brier_clf']:.4f}")
    return out


def main() -> None:
    global SEASON, ROUNDS
    delta = float(sys.argv[sys.argv.index("--delta") + 1]) if "--delta" in sys.argv else None
    if "--season" in sys.argv:
        SEASON = int(sys.argv[sys.argv.index("--season") + 1])
    if "--rounds" in sys.argv:
        first, _, last = sys.argv[sys.argv.index("--rounds") + 1].partition("-")
        ROUNDS = list(range(int(first), int(last or first) + 1))
    df = run(delta, bunch="--bunch" in sys.argv, oracle_rivals="--oracle-rivals" in sys.argv)
    print(f"{SEASON}, rounds {ROUNDS[0]}-{ROUNDS[-1]}" + ("  (ORACLE: rivals on their real stops)" if "--oracle-rivals" in sys.argv else ""))
    report(df)
    disrupted = df.race_id.isin(disrupted_races(SEASON))
    if disrupted.any() and (~disrupted).any():
        print(f"\nexcluding races with rain or a red flag ({df[disrupted].race_id.nunique()} of {df.race_id.nunique()}):")
        report(df[~disrupted])


if __name__ == "__main__":
    main()
