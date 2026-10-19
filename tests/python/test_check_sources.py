"""Unit tests for python/09_check_sources.py on a two-shot synthetic game."""
from __future__ import annotations

import polars as pl
import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def src():
    return load_script("python/09_check_sources.py", "shotq_check_sources")


def _feeds(tmp_path, monkeypatch, src, v3_make="Made"):
    data, pbp = tmp_path / "data", tmp_path / "pbp"
    (pbp / "raw_pbp").mkdir(parents=True)
    data.mkdir()
    # shotdetail stores GAME_ID without its leading zeros, as the archive does
    pl.DataFrame({
        "GAME_ID": ["22300001", "22300001"], "GAME_EVENT_ID": ["5", "9"],
        "PLAYER_ID": ["11", "22"], "SHOT_MADE_FLAG": ["1", "0"],
        "SHOT_TYPE": ["3PT Field Goal", "2PT Field Goal"], "PERIOD": ["1", "2"],
        "LOC_X": ["-220", "10"], "LOC_Y": ["30", "40"],
        "MINUTES_REMAINING": ["11", "0"], "SECONDS_REMAINING": ["40", "33"],
    }).write_csv(data / "shotdetail_2023.csv")
    pl.DataFrame({
        "gameId": ["0022300001"] * 3, "actionNumber": ["5", "7", "9"],
        "isFieldGoal": ["1", "0", "1"], "personId": ["11", "11", "22"],
        "shotResult": [v3_make, "", "Missed"], "shotValue": ["3", "0", "2"],
        "period": ["1", "1", "2"], "xLegacy": ["-220", "", "10"],
        "yLegacy": ["30", "", "40"],
        "clock": ["PT11M40.00S", "PT11M20.00S", "PT00M33.40S"],
    }).write_parquet(pbp / "raw_pbp" / "0022300001.parquet")
    pl.DataFrame({"game_id": ["0022300001"], "source": ["api"]}).write_csv(
        pbp / "raw_pbp_sources.csv")
    monkeypatch.setattr(src, "DATA", data)
    monkeypatch.setattr(src, "PBP", pbp)


def test_matching_feeds_pass_and_leading_zeros_join(src, tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, src)
    row, detail = src.compare("NBA 2023-24", "shotdetail_2023", "00223")
    assert row["matched"] == 2 and row["only_shotdetail"] == 0 and row["only_v3"] == 0
    assert row["make_agree"] == 1.0 and row["xy_exact"] == 1.0
    assert row["clock_within_1s"] == 1.0
    assert row["passed"] and detail.height == 0


def test_a_flipped_make_fails_and_is_listed(src, tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, src, v3_make="Missed")
    row, detail = src.compare("NBA 2023-24", "shotdetail_2023", "00223")
    assert row["make_agree"] == 0.5
    assert not row["passed"]
    assert detail.filter(pl.col("kind") == "disagree")["ev"].to_list() == ["5"]
