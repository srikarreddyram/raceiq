"""Record the planner's predictions before a race, and score them after.

Every accuracy figure elsewhere in this project is a backtest: a past
race replayed with the data that would have been available. That is the
right way to build a model and the wrong way to trust one — a backtest
can quietly read something it shouldn't. This is the other check: before
the lights go out, write down what the planner says for every car, with
a timestamp; after the race, score it. Nothing about the result can leak
into a prediction made before it existed.

Recorded per car: grid slot (and whether it was the real grid or an
expected one), expected finish on a typical strategy and on the
recommended plan, win and points chances, the recommended start tyre and
stops, and the wet-plan call. Files live under data/predictions/ (not in
git), one per recording, so a Friday guess and a Saturday-night plan can
both be kept and scored.

Usage:
    uv run python -m race_plan.prediction_log record 2026_17
    uv run python -m race_plan.prediction_log score 2026_17
    uv run python -m race_plan.prediction_log season 2026
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.common.data import is_classified
from models.common.db import get_connection

PREDICTIONS_ROOT = Path(__file__).resolve().parent.parent / "data" / "predictions"
N_SIMULATIONS = 1000


def _entry_list(race_id: str) -> list[str]:
    from race_plan.future import entry_list, has_race_data

    season = int(race_id.split("_")[0])
    con = get_connection()
    try:
        if has_race_data(con, race_id):
            raise ValueError(f"{race_id} has already been run — a prediction recorded now would be a backtest")
        date = con.execute("SELECT date FROM silver.calendar WHERE race_id = ?", [race_id]).fetchone()
        if date is None:
            raise ValueError(f"{race_id} isn't on the calendar")
        return entry_list(con, season, date[0])["driver_id"].tolist()
    finally:
        con.close()


def record(race_id: str, n_simulations: int = N_SIMULATIONS) -> Path:
    """Plan the race for every car on the entry list and write it down."""
    from race_plan.long_runs import practice_laps
    from race_plan.plan import build_race_plan
    from race_plan.weekend_pace import qualifying_gaps

    drivers = _entry_list(race_id)
    recorded_at = datetime.now(timezone.utc)
    sessions = sorted(practice_laps(race_id)["session"].unique().tolist()) if not practice_laps(race_id).empty else []
    rows = []
    for driver_id in drivers:
        try:
            p = build_race_plan(race_id, driver_id, n_simulations=n_simulations)
        except (ValueError, RuntimeError) as exc:
            print(f"  skipped {driver_id}: {exc}")
            continue
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver_id,
                "recorded_at": recorded_at.isoformat(),
                "grid": p.grid_position,
                "grid_is_expected": p.grid_is_expected,
                "typical_finish": p.typical_expected_finish,
                "plan_finish": p.expected_finish,
                "win_probability": p.typical_win_probability,
                "points_probability": p.typical_points_probability,
                "start_compound": p.starting_compound,
                "stops": len(p.stops),
                "stop_laps": ",".join(str(s.nominal_lap) for s in p.stops),
                "stop_compounds": ",".join(s.compound for s in p.stops),
                "wet_plan": p.wet is not None,
                "wet_leads": bool(p.wet and p.wet.leads),
                "rain_probability": p.rain_probability,
                "conditions_source": p.conditions_source,
                "practice_sessions": ",".join(sessions),
                "qualifying_in": bool(qualifying_gaps(race_id)),
            }
        )
    if not rows:
        raise RuntimeError(f"No plans could be built for {race_id}")
    out = PREDICTIONS_ROOT / race_id / f"{recorded_at.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _race_start(race_id: str) -> pd.Timestamp | None:
    con = get_connection()
    try:
        row = con.execute("SELECT date, time_utc FROM silver.calendar WHERE race_id = ?", [race_id]).fetchone()
    finally:
        con.close()
    if not row:
        return None
    return pd.Timestamp(f"{str(row[0])[:10]}T{str(row[1] or '13:00:00Z').rstrip('Z')}", tz="UTC")


def latest_before_start(race_id: str) -> pd.DataFrame | None:
    """The last recording made before the race started."""
    start = _race_start(race_id)
    files = sorted((PREDICTIONS_ROOT / race_id).glob("*.parquet"))
    for f in reversed(files):
        df = pd.read_parquet(f)
        if start is None or pd.Timestamp(df["recorded_at"].iloc[0]) < start:
            return df
    return None


def score(race_id: str) -> dict | None:
    """The pre-race prediction against what happened."""
    pred = latest_before_start(race_id)
    if pred is None:
        return None
    season, rnd = (int(x) for x in race_id.split("_"))
    con = get_connection()
    try:
        results = con.execute(
            "SELECT driver_id, position, status FROM bronze.ergast_results WHERE season = ? AND round = ?", [season, rnd]
        ).df()
        stints = con.execute(
            """
            SELECT driver_id, MAX(stint_number) - 1 AS stops,
                   ARG_MIN(compound, lap_number) AS start_compound
            FROM gold.lap_features WHERE race_id = ? GROUP BY driver_id
            """,
            [race_id],
        ).df()
    finally:
        con.close()
    if results.empty:
        return None
    results = results[results["status"].map(is_classified)]
    df = pred.merge(results[["driver_id", "position"]], on="driver_id").merge(stints, on="driver_id", how="left", suffixes=("", "_real"))
    if df.empty:
        return None
    # Expected finish is among cars that finish; compare ranks within the finishers.
    df["actual"] = df["position"].rank()
    out = {
        "race_id": race_id,
        "recorded_at": pred["recorded_at"].iloc[0],
        "cars": len(df),
        "grid_was_real": bool(not pred["grid_is_expected"].any()),
        "mae": float((df["typical_finish"].rank() - df["actual"]).abs().mean()),
        "mae_grid": float((df["grid"].rank() - df["actual"]).abs().mean()),
        "rank_corr": float(spearmanr(df["typical_finish"], df["actual"]).statistic),
        "rank_corr_grid": float(spearmanr(df["grid"], df["actual"]).statistic),
        "winner_called": bool(df.loc[df["typical_finish"].idxmin(), "position"] == 1),
        "podium_hits": int(df.nsmallest(3, "typical_finish")["position"].le(3).sum()),
        "stops_right": float((df["stops"] == df["stops_real"]).mean()),
        "start_tyre_right": float((df["start_compound"] == df["start_compound_real"]).mean()),
        "wet_plan_led": bool(pred["wet_leads"].any()),
        "field_started_on_wet_tyres": float(df["start_compound_real"].isin(["INTERMEDIATE", "WET"]).mean()),
    }
    return out


def recorded_races(season: int) -> list[dict]:
    """Every race of the season with a recording before its start: the
    prediction's top cars, and its score once the result is in."""
    out = []
    for folder in sorted(PREDICTIONS_ROOT.glob(f"{season}_*"), key=lambda p: int(p.name.split("_")[1])):
        pred = latest_before_start(folder.name)
        if pred is None:
            continue
        s = score(folder.name)
        actual = {}
        if s is not None:
            season_, rnd = (int(x) for x in folder.name.split("_"))
            con = get_connection()
            try:
                actual = dict(con.execute(
                    "SELECT driver_id, position FROM bronze.ergast_results WHERE season = ? AND round = ?", [season_, rnd]
                ).fetchall())
            finally:
                con.close()
        top = pred.nsmallest(5, "typical_finish")
        out.append(
            {
                "race_id": folder.name,
                "recorded_at": str(pred["recorded_at"].iloc[0]),
                "cars": int(len(pred)),
                "grid_real": bool(not pred["grid_is_expected"].any()),
                "wet_plan_led": bool(pred["wet_leads"].any()),
                "score": s,
                "top": [
                    {
                        "driver_id": r.driver_id,
                        "grid": int(r.grid),
                        "expected_finish": float(r.typical_finish),
                        "win_probability": None if pd.isna(r.win_probability) else float(r.win_probability),
                        "actual": None if actual.get(r.driver_id) is None or pd.isna(actual.get(r.driver_id)) else int(actual[r.driver_id]),
                    }
                    for r in top.itertuples()
                ],
            }
        )
    return out


def season(season: int) -> pd.DataFrame:
    rows = [score(p.name) for p in sorted(PREDICTIONS_ROOT.glob(f"{season}_*"))]
    return pd.DataFrame([r for r in rows if r])


def _print_score(s: dict) -> None:
    print(f"{s['race_id']} — recorded {s['recorded_at']} — {s['cars']} classified cars, grid {'real' if s['grid_was_real'] else 'expected'}")
    print(f"  finishing order   error {s['mae']:.2f} places (grid order {s['mae_grid']:.2f})   rank corr {s['rank_corr']:.2f} (grid {s['rank_corr_grid']:.2f})")
    print(f"  winner called {'yes' if s['winner_called'] else 'no'}   podium {s['podium_hits']}/3")
    print(f"  stop count right {s['stops_right']:.0%}   start tyre right {s['start_tyre_right']:.0%}")
    print(f"  wet plan led: {'yes' if s['wet_plan_led'] else 'no'}   field on wet tyres at the start: {s['field_started_on_wet_tyres']:.0%}")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("record", "score", "season"):
        print(__doc__)
        return
    command, target = sys.argv[1], sys.argv[2]
    if command == "record":
        path = record(target)
        df = pd.read_parquet(path)
        print(f"recorded {len(df)} cars to {path}")
        print(df.sort_values("typical_finish")[["driver_id", "grid", "typical_finish", "win_probability", "start_compound", "stops", "stop_laps", "wet_leads"]].head(10).round(2).to_string(index=False))
    elif command == "score":
        s = score(target)
        if s is None:
            print(f"Nothing to score for {target}: no recording before the start, or no results yet.")
        else:
            _print_score(s)
    else:
        df = season(int(target))
        if df.empty:
            print("No scored races yet.")
            return
        for s in df.to_dict("records"):
            _print_score(s)
        print(f"\n{len(df)} races: mean error {df['mae'].mean():.2f} places (grid order {df['mae_grid'].mean():.2f}), "
              f"rank corr {df['rank_corr'].mean():.2f} (grid {df['rank_corr_grid'].mean():.2f})")


if __name__ == "__main__":
    main()
