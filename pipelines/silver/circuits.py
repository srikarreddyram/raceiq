"""Silver Circuits dimension (PRD Section 9.1).

Ergast's circuit_id is canonical here — it's the id every other Silver
table resolves FastF1's `Location` label against (see id_mappings.py).
Geometry-derived fields (`lap_length_km`, `corner_count`, `drs_zones`,
`svg_path`, etc.) are not populated yet — those come from `track_maps/`,
not yet built.
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table


def build(con: duckdb.DuckDBPyConnection) -> None:
    circuits = con.execute(
        """
        SELECT DISTINCT
            circuit_id,
            name,
            country,
            locality,
            latitude,
            longitude
        FROM bronze.ergast_circuits
        """
    ).df()
    write_silver_table(con, "circuits", circuits)
