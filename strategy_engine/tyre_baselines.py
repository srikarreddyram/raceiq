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
def _circuit_compound_rates() -> dict[tuple[str, str], float]:
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT r.circuit_id, lf.compound, AVG(lf.degradation_rate) AS avg_rate
            FROM gold.lap_features lf
            JOIN silver.races r ON r.race_id = lf.race_id
            WHERE lf.degradation_rate IS NOT NULL AND NOT lf.is_pit_lap
            GROUP BY r.circuit_id, lf.compound
            """
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


def typical_degradation_rate(circuit_id: str, compound: str) -> float:
    circuit_rates = _circuit_compound_rates()
    if (circuit_id, compound) in circuit_rates:
        return float(circuit_rates[(circuit_id, compound)])

    compound_rates = _compound_rates()
    if compound in compound_rates:
        return float(compound_rates[compound])

    return _DEFAULT_DEGRADATION_RATE
