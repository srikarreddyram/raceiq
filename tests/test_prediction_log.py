"""Pre-race predictions are written before the race and scored after."""

from __future__ import annotations

import pandas as pd
import pytest

from race_plan import prediction_log


def test_a_race_already_run_cannot_be_recorded():
    with pytest.raises(ValueError, match="already been run"):
        prediction_log._entry_list("2025_14")


def test_only_recordings_made_before_the_start_count(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_log, "PREDICTIONS_ROOT", tmp_path)
    monkeypatch.setattr(prediction_log, "_race_start", lambda race_id: pd.Timestamp("2026-10-11T12:00:00Z"))
    folder = tmp_path / "2026_17"
    folder.mkdir()
    pd.DataFrame({"recorded_at": ["2026-10-10T20:00:00+00:00"], "driver_id": ["a"]}).to_parquet(folder / "20261010T200000Z.parquet")
    pd.DataFrame({"recorded_at": ["2026-10-11T13:00:00+00:00"], "driver_id": ["b"]}).to_parquet(folder / "20261011T130000Z.parquet")
    kept = prediction_log.latest_before_start("2026_17")
    assert kept is not None and kept["driver_id"].iloc[0] == "a"  # the one after the start is ignored


def test_a_provisional_result_comes_from_the_lap_data_until_the_official_one():
    result, source = prediction_log.race_result("2025_14")  # Hungary 2025: official result ingested
    assert source == "official" and result["position"].min() == 1
    if prediction_log.race_result("2026_17")[1].startswith("provisional"):
        provisional, _ = prediction_log.race_result("2026_17")
        assert provisional["position"].tolist() == sorted(provisional["position"].tolist())
        assert provisional.iloc[0]["driver_id"] == "max_verstappen"  # Singapore 2026 winner
