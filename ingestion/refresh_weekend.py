"""Bring the warehouse up to date for the race weekend in front of you.

The planner is only as current as its data, and the historical backfill
(ingestion/backfill.py) can't keep it current through a weekend:

- it skips every race dated in the future, so on Saturday night the
  qualifying that just finished is never fetched — and the planner keeps
  using a season-average grid and form-only pace when the real grid and
  qualifying gaps (race_plan/weekend_pace.py) are sitting at the source;
- it fetches a round's results, qualifying and standings together and
  skips the round once a results file exists, so it has no way to pick up
  one without the others;
- by default it ingests the race session only, so the tyre-set advice
  (race_plan/tyre_allocation.py) has no practice laps to check against.

This fetches, for every round that needs it, whatever exists so far —
practice, qualifying, the race, results — and rebuilds Bronze, Silver and
Gold. "Needs it" is every round already run that's still missing its
results or race laps (so a few weeks without a refresh are caught up in
one go) plus a weekend that's underway. A session that hasn't happened
yet isn't an error: it's reported as not available and picked up on the
next run.

Models are not retrained here; the planner's inputs (grid, qualifying
pace, form, stop patterns, wear) all come straight from the warehouse.
Restart the API afterwards — it caches plans and lookups in memory.

Usage:
    uv run python -m ingestion.refresh_weekend                 # whatever needs it
    uv run python -m ingestion.refresh_weekend --round 16      # one round, whatever its date
    uv run python -m ingestion.refresh_weekend --no-rebuild    # fetch only
    uv run python -m ingestion.refresh_weekend --rebuild-only  # rebuild from what's on disk
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from datetime import date

from ingestion.backfill import REQUEST_SPACING_SECONDS, _backfill_fastf1, _backfill_openf1, _backfill_weather, _exists
from ingestion.config import IngestionConfig, load_config
from ingestion.ergast import client as ergast_client
from ingestion.ergast.ingest import ingest_circuits, ingest_schedule
from ingestion.raw_writer import write_json

logger = logging.getLogger(__name__)

WEEKEND_SESSIONS = ("FP1", "FP2", "FP3", "Q", "R")
# First practice is two days before the race; one more for time zones.
WEEKEND_LEAD_DAYS = 3


@dataclass
class RoundReport:
    season: int
    round_number: int
    name: str
    race_date: str
    fetched: list[str] = field(default_factory=list)
    not_available: list[str] = field(default_factory=list)

    def line(self) -> str:
        got = ", ".join(self.fetched) or "nothing new"
        missing = f"   not available yet: {', '.join(self.not_available)}" if self.not_available else ""
        return f"R{self.round_number:02d} {self.name} ({self.race_date}): {got}{missing}"


def rounds_to_refresh(races: list[dict], season: int, today: date, config: IngestionConfig) -> list[dict]:
    """Rounds already run that are still missing results or race laps, plus
    a weekend that's underway."""
    chosen = []
    for race in races:
        round_number = str(int(race["round"]))
        race_date = date.fromisoformat(race["date"])
        if race_date <= today:
            complete = _exists(config, "ergast", str(season), round_number, "results.json") and _exists(
                config, "fastf1", str(season), round_number, "R", "results.parquet"
            )
            if not complete:
                chosen.append(race)
        elif (race_date - today).days <= WEEKEND_LEAD_DAYS:
            chosen.append(race)
    return chosen


def _has_races(payload: dict) -> bool:
    return bool(payload.get("MRData", {}).get("RaceTable", {}).get("Races"))


def _refresh_ergast(season: int, round_number: int, config: IngestionConfig, report: RoundReport) -> None:
    """Results, qualifying and standings, each on its own and each written
    only once it has content — an empty results file written before the
    race would look like "already ingested" to the backfill afterwards."""
    sources = {
        "results": ergast_client.get_race_results,
        "qualifying": ergast_client.get_qualifying_results,
    }
    for name, fetch in sources.items():
        if _exists(config, "ergast", str(season), str(round_number), f"{name}.json"):
            continue
        try:
            payload = fetch(season, round_number, config)
        except Exception as error:
            logger.warning("Ergast %s failed for %s round %s: %s", name, season, round_number, error)
            report.not_available.append(name)
            continue
        if not _has_races(payload):
            report.not_available.append(name)
            continue
        write_json(config.raw_data_root, "ergast", f"{season}/{round_number}/{name}", payload)
        report.fetched.append(name)
        time.sleep(REQUEST_SPACING_SECONDS)

    # Standings exist only once the race has been run.
    if "results" in report.fetched:
        try:
            payload = ergast_client.get_constructor_standings(season, round_number, config)
            write_json(config.raw_data_root, "ergast", f"{season}/{round_number}/constructor_standings", payload)
        except Exception as error:
            logger.warning("Constructor standings failed for %s round %s: %s", season, round_number, error)


def refresh_round(season: int, race: dict, config: IngestionConfig, today: date) -> RoundReport:
    round_number = int(race["round"])
    report = RoundReport(season, round_number, race["raceName"], race["date"])
    logger.info("=== %s round %s: %s (%s) ===", season, round_number, race["raceName"], race["date"])

    _refresh_ergast(season, round_number, config, report)

    for session_type in WEEKEND_SESSIONS:
        if _exists(config, "fastf1", str(season), str(round_number), session_type, "results.parquet"):
            continue
        try:
            _backfill_fastf1(season, round_number, config, session_type)
        except Exception as error:
            # Not run yet, or not a session this weekend has (a sprint
            # weekend has no FP2 or FP3).
            logger.info("FastF1 %s not available for %s round %s: %s", session_type, season, round_number, error)
        if _exists(config, "fastf1", str(season), str(round_number), session_type, "results.parquet"):
            report.fetched.append(session_type)
        else:
            report.not_available.append(session_type)

    if date.fromisoformat(race["date"]) <= today:
        circuit = race["Circuit"]
        try:
            _backfill_openf1(season, round_number, race["date"], config)
        except Exception:
            logger.exception("OpenF1 ingestion failed for %s round %s", season, round_number)
        try:
            _backfill_weather(
                circuit["circuitId"], float(circuit["Location"]["lat"]), float(circuit["Location"]["long"]), race["date"], config
            )
        except Exception:
            logger.exception("Weather ingestion failed for %s round %s", season, round_number)
    return report


def rebuild_warehouse() -> None:
    from pipelines.bronze import run as bronze
    from pipelines.gold import run as gold
    from pipelines.silver import run as silver

    bronze.run(sorted(bronze.LOADERS))
    silver.run()
    gold.run()


def refresh(season: int, round_number: int | None = None, rebuild: bool = True, today: date | None = None) -> list[RoundReport]:
    config = load_config()
    today = today or date.today()

    ingest_circuits(season)
    ingest_schedule(season)  # the calendar itself: dates move, rounds get added
    races = ergast_client.get_season_schedule(season, config)["MRData"]["RaceTable"]["Races"]
    if round_number is not None:
        chosen = [r for r in races if int(r["round"]) == round_number]
        if not chosen:
            raise ValueError(f"{season} has no round {round_number}")
    else:
        chosen = rounds_to_refresh(races, season, today, config)

    reports = [refresh_round(season, race, config, today) for race in chosen]
    if rebuild and any(r.fetched for r in reports):
        logger.info("Rebuilding Bronze, Silver and Gold")
        rebuild_warehouse()
    return reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch the current race weekend's sessions and rebuild the warehouse")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--round", type=int, dest="round_number", help="One round, whatever its date")
    parser.add_argument("--no-rebuild", action="store_true", help="Fetch only; don't rebuild Bronze/Silver/Gold")
    parser.add_argument("--rebuild-only", action="store_true", help="Don't fetch; rebuild Bronze/Silver/Gold from disk")
    args = parser.parse_args()

    if args.rebuild_only:
        rebuild_warehouse()
        print("\nWarehouse rebuilt. Restart the API (scripts/dev.sh) so the planner reads the new data.")
        return

    reports = refresh(args.season, args.round_number, rebuild=not args.no_rebuild)

    print()
    if not reports:
        print(f"Nothing to refresh: every {args.season} round run so far is ingested and no weekend is underway.")
        return
    for report in reports:
        print(report.line())
    fetched_any = any(r.fetched for r in reports)
    if not fetched_any:
        print("Nothing new at the sources, so the warehouse wasn't rebuilt.")
    elif args.no_rebuild:
        print("Fetched only. Rebuild with: uv run python -m ingestion.refresh_weekend --rebuild-only")
    else:
        print("Warehouse rebuilt. Restart the API (scripts/dev.sh) so the planner reads the new data.")


if __name__ == "__main__":
    main()
