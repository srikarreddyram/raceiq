"""Typical degradation rate per (circuit, compound), queried from history.

The simulation needs a starting degradation-rate assumption for a stint on
a compound the driver hasn't been running yet (e.g. simulating a switch to
Hard from a current Medium stint). Rather than guessing or carrying the
current compound's fitted rate forward unchanged, this looks up what that
compound has actually done at this circuit historically — falling back to
a global compound average, then a flat default, as the circuit-specific
sample thins out.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from models.common.db import get_connection

_DEFAULT_DEGRADATION_RATE = 0.03  # seconds/lap — a mild, conservative fallback


@lru_cache(maxsize=None)
def _circuit_compound_rates(exclude_race_id: str | None = None) -> dict[tuple[str, str], float]:
    con = get_connection()
    try:
        rows = con.execute(
            f"""
            SELECT r.circuit_id, lf.compound, AVG(lf.degradation_rate) AS avg_rate
            FROM gold.lap_features lf
            JOIN silver.races r ON r.race_id = lf.race_id
            WHERE lf.degradation_rate IS NOT NULL AND NOT lf.is_pit_lap
                {"AND lf.race_id != ?" if exclude_race_id else ""}
            GROUP BY r.circuit_id, lf.compound
            """,
            [exclude_race_id] if exclude_race_id else [],
        ).df()
    finally:
        con.close()
    return {(row.circuit_id, row.compound): row.avg_rate for row in rows.itertuples()}


@lru_cache(maxsize=None)
def _compound_rates() -> dict[str, float]:
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT compound, AVG(degradation_rate) AS avg_rate
            FROM gold.lap_features
            WHERE degradation_rate IS NOT NULL AND NOT is_pit_lap
            GROUP BY compound
            """
        ).df()
    finally:
        con.close()
    return {row.compound: row.avg_rate for row in rows.itertuples()}


@lru_cache(maxsize=None)
def _circuit_compound_max_stint(exclude_race_id: str | None = None) -> dict[tuple[str, str], float]:
    con = get_connection()
    try:
        rows = con.execute(
            f"""
            SELECT r.circuit_id, lf.compound,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY stint_max_age) AS typical_max_age
            FROM (
                SELECT race_id, driver_id, stint_number, compound, MAX(tyre_age) AS stint_max_age
                FROM gold.lap_features
                {"WHERE race_id != ?" if exclude_race_id else ""}
                GROUP BY race_id, driver_id, stint_number, compound
            ) lf
            JOIN silver.races r ON r.race_id = lf.race_id
            GROUP BY r.circuit_id, lf.compound
            """,
            [exclude_race_id] if exclude_race_id else [],
        ).df()
    finally:
        con.close()
    return {(row.circuit_id, row.compound): row.typical_max_age for row in rows.itertuples()}


_DEFAULT_MAX_STINT_LAPS = 25.0  # a mild, conservative fallback


def typical_max_stint_length(circuit_id: str, compound: str, exclude_race_id: str | None = None) -> float:
    """The 75th percentile of observed stint length (by final tyre age) for
    this (circuit, compound) — a generous "how long can this compound
    realistically run here before a team pits" estimate, used by the
    Monte Carlo simulation to decide whether a *rival* (whose own future
    strategy isn't otherwise modeled — see field.py) is carrying tyres old
    enough that they'd need another stop before the race ends. The 75th
    percentile (not the median) is deliberate: it's better to under-charge
    a rival who's genuinely nursing a long stint than to over-charge one
    who isn't, since the whole point is correcting a bias that previously
    always favored rivals.
    """
    rates = _circuit_compound_max_stint(exclude_race_id)
    if (circuit_id, compound) in rates:
        return float(rates[(circuit_id, compound)])
    return _DEFAULT_MAX_STINT_LAPS


def typical_degradation_rate(circuit_id: str, compound: str, exclude_race_id: str | None = None) -> float:
    circuit_rates = _circuit_compound_rates(exclude_race_id)
    if (circuit_id, compound) in circuit_rates:
        return float(circuit_rates[(circuit_id, compound)])

    compound_rates = _compound_rates()
    if compound in compound_rates:
        return float(compound_rates[compound])

    return _DEFAULT_DEGRADATION_RATE


RIVAL_FINAL_COMPOUND = "HARD"


def rival_stop_laps(circuit_id: str, compound: str, tyre_age: float, n_laps: int) -> list[int]:
    """Lap indices (0-based, within the remaining n_laps) at which a rival
    is assumed to stop: as many stops as it takes for no stint to outrun
    typical_max_stint_length — the same feasibility rule our own candidate
    strategies obey. The first stop comes when the current tyres run out;
    each later stint is on RIVAL_FINAL_COMPOUND.

    This used to be at most ONE stop per rival. In a multi-stop race
    (Bahrain) or a wet-to-dry one (Silverstone 2025) our car was charged
    every real stop it made while each rival paid for one, and the
    simulation predicted our car finishing 2.2 places worse than it did, on
    average (strategy_engine/validate_in_race.py).
    """
    stops = []
    next_stop = typical_max_stint_length(circuit_id, compound) - tyre_age
    stint = max(typical_max_stint_length(circuit_id, RIVAL_FINAL_COMPOUND), 1.0)
    while next_stop < n_laps:
        stops.append(int(max(next_stop, 0)))
        next_stop = max(next_stop, 0) + stint
    return stops


# ---------------------------------------------------------------------------
# What the field actually does: real stop patterns, for the pre-race field.
#
# rival_stop_laps gives every rival on the same compound the same stops, on
# the same laps. Before the race that is every rival: the pre-race field
# starts them all on the field's most common compound (race_plan/field.py),
# so the whole grid pitted together — at Hungary 2025, on lap 31 and again
# on lap 67 of 70, a stop no team would make three laps from the flag. Our
# car, on any other plan, jumped the entire field at once and skipped a
# 23 s stop everyone else paid for. Measured against real 2025 finishes
# (race_plan/grid_sensitivity.py) that made back-half starters finish
# 1.1-1.5 places better in the simulation than they really did, and made
# passing HARDER look like it helped them: a car that has jumped a
# one-lap-late field keeps what it jumped.
#
# So before the race each rival is given a stop pattern a real car ran at
# this circuit: how many stops, on what fraction of race distance. That
# brings the real mix of one- and two-stoppers and the real spread of stop
# laps, both measured.
# ---------------------------------------------------------------------------

STOP_PATTERN_MIN_SAMPLES = 20  # fewer and a pattern sample is a handful of cars


@dataclass(frozen=True)
class StopPatterns:
    fractions: tuple[tuple[float, ...], ...]  # each car's in-laps as a fraction of race distance
    compounds: tuple[tuple[str, ...], ...]  # each car's compound per stint, same order
    source: str  # which races they came from, for the plan to say


def _regulation_era(season: int) -> int:
    return 0 if season <= 2021 else 1 if season <= 2025 else 2


@lru_cache(maxsize=None)
def historical_stop_patterns(circuit_id: str, before_date: str) -> StopPatterns | None:
    """Every classified finisher's stops at this circuit, from dry races run
    before `before_date` (so a plan never reads the race it's planning).

    The planned race's regulation era first — the 2022 cars and tyres
    changed how long a stint lasts — then the nearest earlier era, and
    every era only when none has run STOP_PATTERN_MIN_SAMPLES cars here.
    Red-flag races are left out: the stoppage hands everyone a free tyre
    change, so their stints say nothing about how long tyres last (Albert
    Park 2023 and Monaco 2024 read as races nobody pitted in). None when the circuit has too little history at all (a
    new venue): the caller falls back to rival_stop_laps.
    """
    from models.common.data import is_classified

    con = get_connection()
    try:
        laps = con.execute(
            """
            WITH races AS (
                SELECT r.race_id, r.season, MAX(lf.lap_number) AS total_laps
                FROM silver.races r
                JOIN gold.lap_features lf ON lf.race_id = r.race_id
                WHERE r.circuit_id = ? AND r.date < ?
                GROUP BY r.race_id, r.season
                HAVING NOT BOOL_OR(COALESCE(lf.rainfall_flag, FALSE))
                    AND NOT BOOL_OR(COALESCE(lf.red_flag_active, FALSE))
            ),
            stints AS (
                SELECT lf.race_id, lf.driver_id, lf.stint_number,
                       MAX(lf.lap_number) AS in_lap,
                       MODE(lf.compound) AS compound
                FROM gold.lap_features lf JOIN races USING (race_id)
                GROUP BY 1, 2, 3
            )
            SELECT s.race_id, races.season, races.total_laps, s.driver_id, s.stint_number,
                   s.in_lap, s.compound, e.status
            FROM stints s
            JOIN races USING (race_id)
            JOIN bronze.ergast_results e ON e.season || '_' || e.round = s.race_id AND e.driver_id = s.driver_id
            ORDER BY s.race_id, s.driver_id, s.stint_number
            """,
            [circuit_id, before_date],
        ).df()
    finally:
        con.close()
    if laps.empty:
        return None
    laps = laps[laps["status"].map(is_classified)]

    cars = []
    for (race_id, _driver), car in laps.groupby(["race_id", "driver_id"], sort=False):
        car = car.sort_values("stint_number")
        total = float(car["total_laps"].iloc[0])
        stops = car.iloc[:-1]  # the last stint ends at the flag, not in the pits
        cars.append(
            (
                int(car["season"].iloc[0]),
                tuple(float(lap) / total for lap in stops["in_lap"]),
                tuple(str(c).upper() for c in car["compound"]),
            )
        )
    if not cars:
        return None

    # The era of the race being planned, not of the newest data: the first
    # 2026 race at a circuit has only 2022-25 history, and that's an
    # earlier era's cars, which the plan should say.
    target_era = _regulation_era(int(str(before_date)[:4]))
    same_era = [c for c in cars if _regulation_era(c[0]) == target_era]
    if len(same_era) >= STOP_PATTERN_MIN_SAMPLES:
        chosen, source = same_era, "this regulation era"
    else:
        # The nearest earlier era with enough cars; every era only if none has.
        earlier = sorted({_regulation_era(c[0]) for c in cars if _regulation_era(c[0]) < target_era}, reverse=True)
        chosen, source = cars, "every era"
        for era in earlier:
            pool = [c for c in cars if _regulation_era(c[0]) == era]
            if len(pool) >= STOP_PATTERN_MIN_SAMPLES:
                chosen, source = pool, "the previous regulations — too few dry, uninterrupted races here under the current ones"
                break
    if len(chosen) < STOP_PATTERN_MIN_SAMPLES:
        return None
    seasons = sorted({c[0] for c in chosen})
    return StopPatterns(
        fractions=tuple(c[1] for c in chosen),
        compounds=tuple(c[2] for c in chosen),
        source=f"{len(chosen)} cars at this circuit, {seasons[0]}-{seasons[-1]} ({source})",
    )
