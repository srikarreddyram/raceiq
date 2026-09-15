"""Cross-source identifier normalisation (PRD Section 9 — "normalises
identifiers across data sources ... driver IDs between FastF1 and Ergast
differ").

Three identifiers disagree between FastF1 and Ergast, and none of them
share a common key the way drivers do:

- **Drivers** are the easy case: Ergast's `Driver.code` (e.g. "VER") is
  identical to FastF1's `Driver`/`Abbreviation` field, so that's a direct
  join, scoped by season since codes aren't guaranteed unique all-time.
- **Circuits** have no shared key at all. FastF1 labels a race by its
  event `Location` (e.g. "Sakhir"), Ergast labels the same circuit
  "bahrain". There's no code in common.
- **Teams** are similar: FastF1 uses a display name ("Red Bull Racing"),
  Ergast uses a slug ("red_bull").

For circuits and teams this module resolves FastF1's label to Ergast's
canonical id via fuzzy string matching (`rapidfuzz`, already a transitive
dependency of `fastf1`) against Ergast's own name/locality fields, rather
than a hand-typed lookup table spanning every circuit/team across 2018-2026.
Every match below `MATCH_THRESHOLD` is logged instead of silently applied,
so any bad match surfaces immediately rather than silently corrupting a
join.

`STATIC_OVERRIDES` exists for the day fuzzy matching gets something wrong
in practice — it takes precedence over the fuzzy match and should be
extended as those cases are found, not guessed at up front.
"""

from __future__ import annotations

import logging

import duckdb
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 70

CIRCUIT_STATIC_OVERRIDES: dict[str, str] = {
    # FastF1's Location is the city/country the circuit is in; Ergast's
    # locality is the specific district — for these three they share no
    # words at all, so no fuzzy match can bridge them.
    "Singapore": "marina_bay",  # Ergast locality: "Marina Bay"
    "Yas Island": "yas_marina",  # Ergast locality: "Abu Dhabi"
    # Hyphenation splits "Spa-Francorchamps" into one token, so it never
    # matches the single word "Spa" via whitespace-tokenized scoring.
    "Spa-Francorchamps": "spa",
    # Two confirmed silent false positives, found by manually auditing every
    # resolved mapping rather than only the ones that logged as unmapped —
    # a *wrong* match above MATCH_THRESHOLD raises no warning on its own.
    # "Monaco" (the country, FastF1's label for some seasons) scores higher
    # against "Monza" (72.7, short shared prefix) than against the correct
    # locality "Monte Carlo" (35.3) — every Monaco race using this label
    # was silently attributed to Monza until this was added. Likewise
    # "Yas Marina" scored higher against Singapore's "Marina Bay" (shared
    # word) than against its own locality "Abu Dhabi".
    "Monaco": "monaco",
    "Yas Marina": "yas_marina",
}
TEAM_STATIC_OVERRIDES: dict[str, str] = {
    # The team competed as "Racing Bulls" in 2025 after "RB" in 2024 and
    # "AlphaTauri" before that; Ergast has not (yet, as of this data) split
    # out a separate constructor_id for the newest name.
    "Racing Bulls": "rb",
}


def _fuzzy_map(labels: list[str], candidates: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Match each `label` to the best-scoring key in `candidates` (id -> match text).

    Returns {label: candidate_id}. Anything below MATCH_THRESHOLD is logged
    and excluded rather than guessed.
    """
    result: dict[str, str] = {}
    choices = list(candidates.items())  # [(id, match_text), ...]

    for label in labels:
        if label in overrides:
            result[label] = overrides[label]
            continue

        match = process.extractOne(
            label, [text for _, text in choices], scorer=fuzz.token_set_ratio
        )
        if match is None or match[1] < MATCH_THRESHOLD:
            logger.warning("No confident match for %r (best score: %s) — left unmapped", label, match)
            continue

        matched_text, score, index = match
        candidate_id = choices[index][0]
        result[label] = candidate_id
        logger.info("Mapped %r -> %r (score=%s)", label, candidate_id, score)

    return result


def build_circuit_id_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """{fastf1_location: ergast_circuit_id} for every distinct location we've ingested."""
    locations = con.execute("SELECT DISTINCT circuit_id FROM bronze.fastf1_race_meta").df()[
        "circuit_id"
    ].tolist()
    circuits = con.execute("SELECT circuit_id, name, locality FROM bronze.ergast_circuits").df()

    # Match against locality — FastF1's Location is a city/place name, same as this field,
    # not the full circuit name.
    candidates = {row.circuit_id: row.locality for row in circuits.itertuples()}
    return _fuzzy_map(locations, candidates, CIRCUIT_STATIC_OVERRIDES)


def build_team_id_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """{fastf1_team_name: ergast_constructor_id} for every distinct team we've ingested."""
    teams = con.execute("SELECT DISTINCT team FROM bronze.fastf1_laps").df()["team"].tolist()
    constructors = con.execute(
        "SELECT DISTINCT constructor_id, constructor_name FROM bronze.ergast_constructor_standings"
    ).df()

    candidates = {
        row.constructor_id: row.constructor_name
        for row in constructors.itertuples()
    }
    return _fuzzy_map(teams, candidates, TEAM_STATIC_OVERRIDES)
