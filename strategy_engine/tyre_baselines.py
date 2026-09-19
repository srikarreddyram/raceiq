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
