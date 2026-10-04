"""Does practice predict the race? The evidence behind what
race_plan/long_runs.py feeds into the strategy.

Every weekend of 2025-26 with practice laps ingested and a race without
rain or a red flag. Three questions:

  Q1  Does long-run pace (each driver's push laps against the field's, on
      the same compound and session) predict race pace beyond season form
      and qualifying? Fitted leaving one weekend out at a time.
      -> correlation 0.43 with race pace, but MAE 0.382 -> 0.376 s/lap on
         top of form and 0.323 -> 0.321 on top of form + qualifying. Not
         used.
  Q3  Does the field's practice degradation predict how hard the race is
      on tyres (per compound), better than history (season rate x
      circuit factor)?
      -> correlation 0.68 against 0.37 (softs 0.76 against 0.25). Used:
         tyre_pace.race_wear blends 0.32 x practice + 0.54 x history,
         error 0.0310 -> 0.0253 s/lap per lap, each weekend predicted from
         the others (fitted at the end of this script).
  Q2  Does a team's practice degradation against the field's predict its
      race degradation against the field's?
      -> correlation 0.02, and season-to-date team degradation 0.04. Not
         used: it is shown on the Race Weekend page as Friday's story.

Usage:
    uv run python -m race_plan.long_runs_validation
"""

import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from models.common.db import get_connection
import race_plan.long_runs as lr
import race_plan.weekend_pace as wp
from car_profiles.degradation_curves import season_model


def main() -> None:
    con = get_connection()
    weekends = con.execute("""select distinct season||'_'||round race_id from bronze.fastf1_laps
        where session_type in ('FP1','FP2','FP3') and season >= 2025""").df().race_id.tolist()
    ok = con.execute("""select race_id from gold.lap_features group by 1
        having not bool_or(coalesce(rainfall_flag,false)) and not bool_or(coalesce(red_flag_active,false))""").df().race_id.tolist()
    race_laps = con.execute("""select lf.race_id, lf.driver_id, lf.team_id, lf.stint_number, lf.compound, lf.tyre_age, lf.lap_number, lf.lap_time_seconds t
        from gold.lap_features lf where lf.race_id like '202%' and not lf.is_pit_lap and not lf.safety_car_active and not lf.vsc_active
        and not lf.red_flag_active and not coalesce(lf.yellow_active,false) and lf.compound in ('SOFT','MEDIUM','HARD')
        and lf.tyre_age between 2 and 30 and lf.lap_number > 1 and lf.lap_time_seconds is not null""").df()
    con.close()
    weekends = [w for w in weekends if w in ok]
    print("dry, uninterrupted weekends with practice:", len(weekends))

    # --- practice measures per weekend ---------------------------------------
    pace_rows, deg_rows, field_rows = [], [], []
    for w in weekends:
        laps = lr.practice_laps(w)
        push = laps[laps.push]
        if push.empty: continue
        # compound-adjusted pace: gap to the field median on the same compound, per session, lap-weighted
        g = push.groupby(["session", "compound", "driver_id"]).lap_time.agg(["mean", "size"]).reset_index()
        g["gap"] = g["mean"] - g.groupby(["session", "compound"])["mean"].transform("median")
        g = g[g.groupby(["session", "compound"]).driver_id.transform("size") >= 6]
        p = g.groupby("driver_id").apply(lambda x: pd.Series({"practice": np.average(x.gap, weights=x["size"]), "plaps": x["size"].sum()}))
        p["race_id"] = w; pace_rows.append(p.reset_index())
        t = lr.team_summary(w); t["race_id"] = w; deg_rows.append(t)
        f = lr.field_degradation(w); f["race_id"] = w; field_rows.append(f)
    pace = pd.concat(pace_rows); tdeg = pd.concat(deg_rows); fdeg = pd.DataFrame(field_rows)
    pace["practice"] = pace.practice.clip(-2, 2)

    # --- Q1: pace -------------------------------------------------------------
    d = wp._dataset()
    d = d.merge(pace, on=["race_id", "driver_id"])
    print(f"\nQ1 pace: {len(d)} driver-races, {d.race_id.nunique()} weekends")
    print("  correlation with race pace:", {c: round(d[c].corr(d.race_pace), 3) for c in ["form", "quali", "practice"]})
    def cv(cols):
        errs = []
        for w in d.race_id.unique():   # leave one weekend out
            tr, te = d[d.race_id != w], d[d.race_id == w]
            b, *_ = np.linalg.lstsq(tr[cols].to_numpy(), tr.race_pace.to_numpy(), rcond=None)
            errs.append(np.abs(te[cols].to_numpy() @ b - te.race_pace))
        e = np.concatenate(errs); b, *_ = np.linalg.lstsq(d[cols].to_numpy(), d.race_pace.to_numpy(), rcond=None)
        return e.mean(), np.median(e), np.round(b, 2)
    for cols in (["form"], ["form", "practice"], ["practice"], ["form", "quali"], ["form", "quali", "practice"]):
        m, md, b = cv(cols); print(f"  {' + '.join(cols):26s} leave-one-weekend-out MAE {m:.3f}  median {md:.3f}   coef {b}")

    # --- race degradation per team and per field --------------------------------
    fuel = {s: season_model(s)["fuel_track_seconds_per_lap"] for s in (2025, 2026)}
    race_laps["season"] = race_laps.race_id.str[:4].astype(int)
    race_laps["stint_key"] = race_laps.race_id + race_laps.driver_id + race_laps.stint_number.astype(str)
    race_laps["c"] = race_laps.t - race_laps.season.map(fuel) * race_laps.lap_number
    med = race_laps.groupby("stint_key").c.transform("median")
    race_laps = race_laps[(race_laps.c - med).abs() < 0.03 * med]
    def slope(x):
        a = x.tyre_age - x.groupby("stint_key").tyre_age.transform("mean"); t = x.c - x.groupby("stint_key").c.transform("mean")
        den = (a * a).sum(); return (a * t).sum() / den if den > 0 and len(x) >= 15 else np.nan
    rdeg_team = race_laps.groupby(["race_id", "team_id", "compound"]).apply(slope).rename("race_deg").reset_index()
    rdeg_field = race_laps.groupby(["race_id", "compound"]).apply(slope).rename("race_field").reset_index()

    # --- Q3: field degradation this weekend -----------------------------------
    from strategy_engine.tyre_pace import history_wear
    q3 = []
    for _, r in fdeg.iterrows():
        for c in ("SOFT", "MEDIUM", "HARD"):
            q3.append({"race_id": r.race_id, "compound": c, "practice_field": r[c], "planner": history_wear(r.race_id).get(c)})
    q3 = pd.DataFrame(q3).merge(rdeg_field, on=["race_id", "compound"]).dropna()
    print(f"\nQ3 field degradation, {len(q3)} weekend-compounds: race slope mean {q3.race_field.mean():.3f}, practice mean {q3.practice_field.mean():.3f}, planner mean {q3.planner.mean():.3f}")
    print(f"  corr with race: practice {q3.practice_field.corr(q3.race_field):.2f}   planner (season x circuit) {q3.planner.corr(q3.race_field):.2f}")
    for c, g in q3.groupby("compound"):
        print(f"   {c}: n {len(g)}  corr practice {g.practice_field.corr(g.race_field):.2f}  planner {g.planner.corr(g.race_field):.2f}")

    # --- Q2: team degradation relative to the field -----------------------------
    t = []
    for _, r in tdeg.iterrows():
        for c in ("SOFT", "MEDIUM", "HARD"):
            v = r.get(f"deg_{c.lower()}")
            if v is not None and not pd.isna(v): t.append({"race_id": r.race_id, "team_id": r.team_id, "compound": c, "practice_deg": v})
    t = pd.DataFrame(t).merge(rdeg_team, on=["race_id", "team_id", "compound"]).dropna()
    t["practice_rel"] = t.practice_deg - t.groupby(["race_id", "compound"]).practice_deg.transform("mean")
    t["race_rel"] = t.race_deg - t.groupby(["race_id", "compound"]).race_deg.transform("mean")
    t = t[t.groupby(["race_id", "compound"]).team_id.transform("size") >= 4]
    print(f"\nQ2 team degradation vs the field, {len(t)} team-weekend-compounds")
    print(f"  corr(practice relative deg, race relative deg) = {t.practice_rel.corr(t.race_rel):.2f}")
    # season-to-date team deg (race-based, earlier races only) as the comparison
    rdeg_team["season"] = rdeg_team.race_id.str[:4].astype(int); rdeg_team["rnd"] = rdeg_team.race_id.str.split("_").str[1].astype(int)
    rdeg_team["rel"] = rdeg_team.race_deg - rdeg_team.groupby(["race_id", "compound"]).race_deg.transform("mean")
    prior = []
    for _, r in t.iterrows():
        s, rnd = int(r.race_id[:4]), int(r.race_id.split("_")[1])
        h = rdeg_team[(rdeg_team.season == s) & (rdeg_team.rnd < rnd) & (rdeg_team.team_id == r.team_id)].rel
        prior.append(h.mean() if len(h) else np.nan)
    t["season_rel"] = prior
    k = t.dropna(subset=["season_rel"])
    print(f"  with a season-to-date figure ({len(k)}): corr practice {k.practice_rel.corr(k.race_rel):.2f}   season-to-date {k.season_rel.corr(k.race_rel):.2f}")
    X = k[["practice_rel", "season_rel"]].to_numpy(); b, *_ = np.linalg.lstsq(np.c_[np.ones(len(k)), X], k.race_rel, rcond=None)
    print(f"  race_rel = {b[0]:+.3f} + {b[1]:.3f} practice + {b[2]:.3f} season")

    # The blend tyre_pace.race_wear uses, refitted and scored leaving one weekend out.
    def lowo(cols):
        errs = []
        for w in q3.race_id.unique():
            tr, te = q3[q3.race_id != w], q3[q3.race_id == w]
            b, *_ = np.linalg.lstsq(tr[cols].to_numpy(), tr.race_field.to_numpy(), rcond=None)
            errs.append(np.abs(np.maximum(te[cols].to_numpy() @ b, 0) - te.race_field))
        b, *_ = np.linalg.lstsq(q3[cols].to_numpy(), q3.race_field.to_numpy(), rcond=None)
        return np.concatenate(errs).mean(), np.round(b, 3)
    print(f"\nrace degradation, error (s/lap per lap): history as is {(q3.planner - q3.race_field).abs().mean():.4f}")
    for cols in (["practice_field"], ["practice_field", "planner"]):
        m, b = lowo(cols)
        print(f"  {' + '.join(cols):24s} leave-one-weekend-out {m:.4f}   coef {b}")


if __name__ == "__main__":
    main()
