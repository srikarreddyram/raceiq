"""Silver Drivers dimension (PRD Section 9.1).

Ergast's `driverId` is canonical (e.g. "max_verstappen"); `driver_code`
(e.g. "VER") is what FastF1-sourced tables join against, scoped by season
since a 3-letter code isn't guaranteed unique across F1's entire history.
One row per driver per season — where a driver changed teams mid-season,
`constructor_id` reflects their most recent round that season.
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table


def build(con: duckdb.DuckDBPyConnection) -> None:
    drivers = con.execute(
        """
        SELECT
            driver_id,
            driver_code,
            driver_given_name AS given_name,
            driver_family_name AS family_name,
            driver_nationality AS nationality,
            driver_date_of_birth AS date_of_birth,
            season,
            constructor_id
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY season, driver_id ORDER BY round DESC
                ) AS rn
            FROM bronze.ergast_results
        )
        WHERE rn = 1
        """
    ).df()
    write_silver_table(con, "drivers", drivers)
