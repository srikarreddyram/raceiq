"""Historical average pit stop cost per circuit — a data-grounded stand-in
for PRD's circuit-specific `pit_lane_delta` (a `track_maps/`-derived
feature; not built), taken from real observed stop durations instead of
being fabricated as a constant.

`silver.pit_stops.stop_duration_seconds` includes a small number of
extreme outliers — e.g. a car that pitted lap 1 with damage and didn't
re-emerge until lap 2, computing as a genuine but ~4,700-second "stop."
Those are real events, not bad data, but they're not a normal scheduled
tyre change and would badly skew a *typical* pit-loss estimate (one such
row turns a ~25s median into a ~250s mean at some circuits). Filtered to
a plausible scheduled-stop window before averaging.
"""

from __future__ import annotations

from functools import lru_cache

from models.common.db import get_connection

_DEFAULT_PIT_LOSS_SECONDS = 22.0  # roughly typical across F1 circuits, used only if a circuit has no history
_PLAUSIBLE_STOP_RANGE = (15.0, 60.0)  # excludes damage/red-flag/drive-through outliers, not normal tyre changes


@lru_cache(maxsize=None)
def _circuit_pit_loss() -> dict[str, float]:
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT r.circuit_id, AVG(ps.stop_duration_seconds) AS avg_loss
            FROM silver.pit_stops ps
            JOIN silver.races r ON r.race_id = ps.race_id
            WHERE ps.stop_duration_seconds BETWEEN ? AND ?
            GROUP BY r.circuit_id
            """,
            list(_PLAUSIBLE_STOP_RANGE),
        ).df()
    finally:
        con.close()
    return {row.circuit_id: row.avg_loss for row in rows.itertuples()}


def typical_pit_loss_seconds(circuit_id: str) -> float:
    return float(_circuit_pit_loss().get(circuit_id, _DEFAULT_PIT_LOSS_SECONDS))
