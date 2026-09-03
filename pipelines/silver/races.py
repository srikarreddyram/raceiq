"""Silver Races dimension (PRD Section 9.1).

`race_id` is minted here as `{season}_{round}` and used as the join key by
every other Silver fact table (laps, weather, track status, pit stops).
`circuit_id` is resolved from FastF1's raw location label to Ergast's
canonical id via `id_mappings.build_circuit_id_map` — see that module's
docstring for why this is a fuzzy match rather than a fixed table.
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table
from pipelines.silver.id_mappings import build_circuit_id_map


def build(con: duckdb.DuckDBPyConnection) -> None:
    races = con.execute(
        """
        SELECT season, round, circuit_id AS fastf1_circuit_id, name, date
        FROM bronze.fastf1_race_meta
        WHERE session_type = 'R'
        """
    ).df()
    if races.empty:
        return

    circuit_map = build_circuit_id_map(con)
    races["race_id"] = races["season"].astype(str) + "_" + races["round"].astype(str)
    races["circuit_id"] = races["fastf1_circuit_id"].map(circuit_map)

    write_silver_table(
        con, "races", races[["race_id", "season", "round", "circuit_id", "name", "date"]]
    )
