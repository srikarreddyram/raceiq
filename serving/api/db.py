"""Per-request DuckDB connection — PRD Section 14's serving layer reads
Gold/Silver, never Bronze or Raw.

A fresh read-only connection per request rather than one shared,
long-lived connection: DuckDB connections aren't safe for concurrent
queries from multiple threads, and a local file is cheap enough to open
that pooling isn't worth the complexity yet. If this ever becomes a
throughput bottleneck, that's the first thing to revisit.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb

from pipelines.bronze.db import bronze_db_path


def get_db() -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(bronze_db_path()), read_only=True)
    try:
        yield con
    finally:
        con.close()
