"""Which tyre sets to spend in practice, and which to keep sealed for
qualifying and the race.

This is mostly a backwards constraint problem rather than a prediction
one. A driver gets a fixed allocation for the weekend, the race plan
(race_plan/plan.py) says which compounds the race needs, qualifying needs
fresh softs to be worth anything, and whatever is left over is the
practice budget. So the recommendation is: reserve the race, reserve
qualifying, spend the rest — and say plainly when the arithmetic doesn't
leave enough.

What's measured versus what's assumed:

- **Measured.** Sets actually used, reconstructed from FastF1's per-lap
  `FreshTyre` and `TyreLife`. A set begins at a lap flagged fresh and
  continues through later stints whose tyre life keeps incrementing, so
  running four laps in FP1 and nine more later is one set with 13 laps on
  it, not two sets. Checked against Leclerc at Madring 2026: 10 fresh sets
  across FP1-Q, leaving 3 of his 13 for the race, which matches the
  three-stint race he actually ran.

- **Assumed.** The allocation itself (13 dry sets: 2 hard, 3 medium, 8
  soft) is a regulation constant, not something in this data. The
  progressive hand-back schedule through practice is deliberately NOT
  modelled — the exact number due back after each session has moved
  around between seasons, and encoding a version of it that's wrong for
  the year being planned would be worse than leaving it to the strategist
  who knows the current rules.

The earlier version of this module raised rather than guessing, because
only race sessions had ever been ingested. Practice and qualifying are
now ingestible (`--sessions FP1,FP2,FP3,Q,R`, see ingestion/backfill.py),
which is what made the measured half of this possible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


from models.common.db import get_connection

# FIA dry-weather allocation per driver per weekend.
REGULATION_ALLOCATION: dict[str, int] = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}

# Qualifying realistically needs a fresh soft for each segment a driver
# expects to run. Reserving three is the front-running assumption; a car
# that won't reach Q3 needs fewer, which `quali_segments` allows for.
DEFAULT_QUALI_SEGMENTS = 3

PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")


@dataclass
class TyreSet:
    compound: str
    laps_run: int
    first_session: str
    sessions: tuple[str, ...]

    @property
    def is_scrubbed(self) -> bool:
        return self.laps_run > 0


@dataclass
class AllocationPlan:
    driver_id: str
    race_id: str
    allocation: dict[str, int]
    race_reserved: dict[str, int]
    quali_reserved: dict[str, int]
    practice_budget: dict[str, int]
    used_in_practice: list[TyreSet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def practice_sets_available(self) -> int:
        return sum(self.practice_budget.values())


def reconstruct_sets(race_id: str, driver_id: str) -> list[TyreSet]:
    """Rebuild the physical tyre sets a driver used across the weekend.

    A new set starts wherever FastF1 flags a fresh tyre; laps that follow
    on the same compound with still-increasing tyre life belong to that
    same set, even across a session boundary.
    """
    season, rnd = (int(part) for part in race_id.split("_"))
    con = get_connection()
    try:
        laps = con.execute(
            """
            SELECT session_type, driver, compound, tyre_age_laps, is_fresh_tyre, lap_number, stint_number
            FROM bronze.fastf1_laps
            WHERE season = ? AND round = ? AND session_type != 'R'
            ORDER BY session_type, stint_number, lap_number
            """,
            [season, rnd],
        ).df()
    finally:
        con.close()

    if laps.empty:
        return []

    # `driver` here is FastF1's three-letter code, which is what this table
    # carries; the caller passes an Ergast driver_id, so resolve it.
    code = _driver_code(season, driver_id)
    laps = laps[laps.driver == code]
    if laps.empty:
        return []

    session_order = {name: i for i, name in enumerate([*PRACTICE_SESSIONS, "Q"])}
    laps = laps.sort_values(
        by=["session_type", "stint_number", "lap_number"],
        key=lambda col: col.map(session_order) if col.name == "session_type" else col,
    )

    sets: list[TyreSet] = []
    current: dict | None = None
    for row in laps.itertuples():
        starts_new_set = bool(row.is_fresh_tyre) and row.tyre_age_laps <= 1
        if starts_new_set or current is None or current["compound"] != row.compound:
            if current is not None:
                sets.append(_finalise(current))
            current = {
                "compound": row.compound,
                "laps": 0,
                "first_session": row.session_type,
                "sessions": [],
            }
        current["laps"] += 1
        if row.session_type not in current["sessions"]:
            current["sessions"].append(row.session_type)

    if current is not None:
        sets.append(_finalise(current))
    return sets


def _finalise(state: dict) -> TyreSet:
    return TyreSet(
        compound=state["compound"],
        laps_run=state["laps"],
        first_session=state["first_session"],
        sessions=tuple(state["sessions"]),
    )


def _driver_code(season: int, driver_id: str) -> str | None:
    con = get_connection()
    try:
        row = con.execute(
            "SELECT driver_code FROM bronze.ergast_results WHERE season = ? AND driver_id = ? LIMIT 1",
            [season, driver_id],
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def recommend_tyre_allocation(
    race_id: str,
    driver_id: str,
    race_compounds: list[str],
    quali_segments: int = DEFAULT_QUALI_SEGMENTS,
) -> AllocationPlan:
    """Reserve the race, reserve qualifying, spend what's left on practice.

    `race_compounds` is the compound sequence the race plan recommends —
    `RacePlan.compound_sequence` — so the allocation follows the strategy
    rather than being decided independently of it.
    """
    allocation = dict(REGULATION_ALLOCATION)
    warnings: list[str] = []

    race_reserved = Counter(compound.upper() for compound in race_compounds)
    quali_reserved = Counter({"SOFT": max(0, quali_segments)})

    practice_budget: dict[str, int] = {}
    for compound, total in allocation.items():
        remaining = total - race_reserved.get(compound, 0) - quali_reserved.get(compound, 0)
        if remaining < 0:
            warnings.append(
                f"{compound}: the plan needs {race_reserved.get(compound, 0)} for the race plus "
                f"{quali_reserved.get(compound, 0)} for qualifying, which is more than the "
                f"{total} allocated — the race plan or the qualifying approach has to give"
            )
            remaining = 0
        practice_budget[compound] = remaining

    used = reconstruct_sets(race_id, driver_id)
    if not used:
        warnings.append(
            "No practice or qualifying laps ingested for this race, so actual set usage can't be "
            "checked against the plan (ingest with --sessions FP1,FP2,FP3,Q,R)."
        )

    return AllocationPlan(
        driver_id=driver_id,
        race_id=race_id,
        allocation=allocation,
        race_reserved=dict(race_reserved),
        quali_reserved={k: v for k, v in quali_reserved.items() if v},
        practice_budget=practice_budget,
        used_in_practice=used,
        warnings=warnings,
    )


def format_allocation(plan: AllocationPlan) -> str:
    lines = ["TYRE ALLOCATION", "-" * 78]
    lines.append(
        "  allocation      " + "  ".join(f"{c} x{n}" for c, n in plan.allocation.items())
    )
    lines.append(
        "  reserve: race   " + ("  ".join(f"{c} x{n}" for c, n in plan.race_reserved.items()) or "none")
    )
    lines.append(
        "  reserve: quali  " + ("  ".join(f"{c} x{n}" for c, n in plan.quali_reserved.items()) or "none")
    )
    lines.append(
        f"  practice budget " + "  ".join(f"{c} x{n}" for c, n in plan.practice_budget.items())
        + f"   ({plan.practice_sets_available} sets)"
    )

    if plan.used_in_practice:
        lines.append("")
        lines.append(f"  actually used in practice/quali ({len(plan.used_in_practice)} sets):")
        for tyre_set in plan.used_in_practice:
            lines.append(
                f"    {tyre_set.compound:<7} {tyre_set.laps_run:>3} laps   "
                f"{'+'.join(tyre_set.sessions)}"
            )

    for warning in plan.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)
