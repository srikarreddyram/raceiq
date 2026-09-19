"""Tyre set allocation across a race weekend — NOT BUILT, and this
records exactly why and what it would take.

The ask: recommend which of a driver's tyre sets to burn in FP1-FP3 and
which to keep sealed for qualifying and the race. Under the current
regulations each car gets 13 dry sets for a weekend (2 hard, 3 medium,
8 soft) and must hand sets back progressively through practice, so by
Q1 the choice of what's left — and how scrubbed it is — is already
partly decided. Getting that wrong costs a Q3 lap or a race stint.

Why it isn't here:

This project has only ever ingested race sessions. `bronze.fastf1_laps`
holds 202,577 laps and every one of them is session_type 'R'; there is
not a single practice or qualifying lap in the warehouse. A recommender
for practice running would therefore be inferring practice behaviour from
data containing no practice, which is not a modelling difficulty, it's an
absence of the subject matter.

What it would take, in order:

1. **Ingest FP1/FP2/FP3 and Q sessions.** ingestion/fastf1/client.py's
   `load_session` already takes a session type, so this is a change to
   the backfill loop rather than new client code. The cost is the rate
   limit: FastF1 allows ~500 calls/hour and this would roughly quadruple
   the number of sessions fetched, so the existing backoff logic in
   ingestion/backfill.py would be doing real work over a long run.

2. **Carry session_type through Silver.** pipelines/silver/laps.py filters
   `WHERE session_type = 'R'` today, and every downstream Gold assumption
   — one race per race_id, lap numbers that run 1..N once, a single
   classified result — depends on that filter. Practice laps can't simply
   be unioned in; they need either their own fact table or an explicit
   session dimension threaded through, or Gold's grain quietly breaks.

3. **Reconstruct per-set usage.** This is the genuinely uncertain part and
   should be scoped before the first two are paid for. FastF1 exposes
   compound, tyre life and a fresh-tyre flag per stint, which is enough to
   infer *how many laps a set has done*, but it does not expose the FIA's
   per-set identity. Whether sets can be tracked reliably across sessions
   from life-and-compound alone — a set run 6 laps in FP1 and 4 more in
   FP3 has to be recognised as the same set — needs checking against real
   weekends before anything is promised.

4. **Model what "saving" a set is worth.** The recommendation is a trade:
   a scrubbed soft is slower in Q3 but a fresh one spent in FP2 buys setup
   information. Quantifying that needs qualifying data (step 1) and a
   view of how much a practice lap is worth, which nothing in this project
   currently measures.

Steps 1 and 2 are mechanical. Step 3 is a real unknown and decides
whether the feature is possible at all; step 4 decides whether it's
useful. Recommending sets without any of them would mean inventing the
allocation rather than inferring it.
"""

from __future__ import annotations


class TyreAllocationNotAvailable(NotImplementedError):
    """Raised instead of returning a fabricated allocation."""


def recommend_tyre_allocation(*_args, **_kwargs):
    raise TyreAllocationNotAvailable(
        "Practice tyre allocation needs FP1-FP3 session data. The warehouse contains race "
        "sessions only (all 202,577 laps in bronze.fastf1_laps are session_type 'R') — see "
        "this module's docstring for the four steps required."
    )
