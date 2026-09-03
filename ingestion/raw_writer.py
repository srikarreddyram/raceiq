"""Persistence helpers for the Raw Layer (PRD Section 7).

The Raw Layer's contract is simple: store exactly what the source returned,
plus enough metadata to know when and how it was fetched, so any later
pipeline stage can be replayed from scratch. Nothing here parses, joins, or
validates — that starts in `pipelines/bronze`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _ingested_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(root: Path, source: str, relative_path: str, payload: Any) -> Path:
    """Write a raw JSON API response, wrapped with ingestion metadata.

    Used by the REST-based sources (Ergast, OpenF1, weather) where the
    source's native format already is JSON — we store it close to verbatim.
    """
    destination = root / source / f"{relative_path}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "source": source,
        "ingested_at": _ingested_at(),
        "payload": payload,
    }
    destination.write_text(json.dumps(envelope, indent=2, default=str))
    return destination


def write_table(root: Path, source: str, relative_path: str, frame: pd.DataFrame) -> Path:
    """Write a raw tabular extract (Parquet) with an ingestion timestamp column.

    Used for FastF1, whose Python API already returns parsed DataFrames —
    there is no more "native" a form to store than the table it hands us.
    Parquet (not CSV) preserves dtypes so Bronze doesn't have to re-infer
    them.
    """
    destination = root / source / f"{relative_path}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)

    frame = frame.copy()
    frame["_ingested_at"] = _ingested_at()
    frame.to_parquet(destination, index=False)
    return destination
