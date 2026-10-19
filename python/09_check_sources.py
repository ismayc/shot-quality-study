"""Check every shot against a second feed: the play-by-play's own shot events.

Why
---
The R/Python reconcile (03_reconcile.py) checks the arithmetic, not the data:
both implementations read the same shotdetail CSV, so a dropped shot, a bad
coordinate, or a wrong make flag would be in both. The sibling
playbyplay-study learned this the hard way on 2026-10-19 (a shared ordering
choice corrupted 15% of its windows and only a second feed caught it).

This joins each shotdetail row (GAME_ID, GAME_EVENT_ID) to the matching
PlayByPlayV3 field-goal event (gameId, actionNumber) cached in
../playbyplay-study/data/raw_pbp/ and requires them to agree.

What the second feed can and cannot catch
-----------------------------------------
V3 and ShotChartDetail are separate stats.nba.com endpoints, so this catches
dropped, duplicated, or misjoined shots and disagreements in make, value,
player, period, clock, and court location. It does not independently
measure location: xLegacy/yLegacy are the same tracking coordinates as
LOC_X/LOC_Y. Independence also varies by season. playbyplay-study harvested
NBA 2023-24, part of 2024-25, and WNBA 2024 from the API itself; the rest of
its V3 rows came from the shufinskiy/nba_data archive, the same producer as
these shotdetail files (data/raw_pbp_sources.csv there records which). The
share of each season's games that came from the API is reported per season.

Known disagreements, found when this check was built (2026-10-19): V3 holds
a placeholder location on a handful of shots a season (x about +/-223, y
about 2, distance 0) where shotdetail has a plausible one; a few clocks
differ by 3-7 s (a time correction applied in one feed); one or two shots a
season exist in only one feed. Every disagreement is written out, not only
counted.

Seasons: the shotdetail files this study uses that the 2-for-1 study also
covers: NBA 2023-24, 2024-25, 2025-26; WNBA 2024, 2025. (G League has no V3
cache; it is not covered here.)

Gates per season:
  * the two feeds cover the same games;
  * at most 0.01% of shots on either side are unmatched;
  * at most 0.5% of games differ in field-goal count;
  * make/miss and shot value agree on every matched shot;
  * period agrees on every matched shot;
  * player agrees on at least 99.95% of matched shots;
  * LOC_X and LOC_Y agree exactly on at least 99.9%;
  * the clock (whole seconds) agrees within 1 s on at least 99.9%.

Outputs: output/source_check.csv (one row per season),
         output/source_disagreements.csv (every disagreeing shot)
Run:     python python/09_check_sources.py          build
         python python/09_check_sources.py --check  rerun; fail on a gate or
                                                     on drift from the committed table
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
PBP = ROOT.parent / "playbyplay-study" / "data"

# (label, shotdetail file stem, V3 game-ID prefix)
SEASONS = [
    ("NBA 2023-24", "shotdetail_2023", "00223"),
    ("NBA 2024-25", "shotdetail_2024", "00224"),
    ("NBA 2025-26", "shotdetail_2025", "00225"),
    ("WNBA 2024", "wnba_shotdetail_2024", "10224"),
    ("WNBA 2025", "wnba_shotdetail_2025", "10225"),
]

MAX_UNMATCHED = 0.0001
MAX_GAMES_COUNT_DIFF = 0.005
MIN_PLAYER = 0.9995
MIN_XY = 0.999
MIN_CLOCK = 0.999


def shotdetail(stem: str) -> pl.DataFrame:
    csv = DATA / f"{stem}.csv"
    if not csv.exists():
        with tarfile.open(DATA / f"{stem}.tar.xz", mode="r:xz") as tar:
            tar.extractall(DATA, filter="data")
    sd = pl.read_csv(csv, infer_schema_length=0)
    # the archive stores GAME_ID as an integer: "0022300001" arrives as
    # "22300001", and a join on the raw value matches nothing
    return sd.with_columns(gid=pl.col("GAME_ID").str.zfill(10),
                           ev=pl.col("GAME_EVENT_ID").cast(pl.Int64))


def v3_shots(prefix: str) -> pl.DataFrame:
    files = sorted((PBP / "raw_pbp").glob(f"{prefix}*.parquet"))
    if not files:
        raise SystemExit(f"no V3 play-by-play for prefix {prefix} in {PBP / 'raw_pbp'}; "
                         f"this check needs the playbyplay-study sibling checkout")
    v3 = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    return (v3.filter(pl.col("isFieldGoal") == "1")
            .with_columns(gid=pl.col("gameId"), ev=pl.col("actionNumber").cast(pl.Int64)))


def clock_secs(col: str) -> pl.Expr:
    return (pl.col(col).str.extract(r"PT(\d+)M", 1).cast(pl.Float64) * 60
            + pl.col(col).str.extract(r"M([\d.]+)S", 1).cast(pl.Float64))


def api_share(prefix: str) -> float:
    path = PBP / "raw_pbp_sources.csv"
    src = pl.read_csv(path, schema_overrides={"game_id": pl.String})
    sub = src.filter(pl.col("game_id").str.starts_with(prefix))
    return float((sub["source"] == "api").mean()) if sub.height else float("nan")


def compare(label: str, stem: str, prefix: str) -> tuple[dict, pl.DataFrame]:
    sd = shotdetail(stem)
    v3 = v3_shots(prefix)
    games_sd = set(sd["gid"].unique())
    games_v3 = set(v3["gid"].unique())

    j = sd.join(v3, on=["gid", "ev"], how="full", coalesce=True)
    only_sd = j.filter(pl.col("personId").is_null())
    only_v3 = j.filter(pl.col("PLAYER_ID").is_null())
    both = (j.filter(pl.col("PLAYER_ID").is_not_null() & pl.col("personId").is_not_null())
            .with_columns(
                make_sd=pl.col("SHOT_MADE_FLAG").cast(pl.Int64),
                make_v3=(pl.col("shotResult") == "Made").cast(pl.Int64),
                value_sd=pl.when(pl.col("SHOT_TYPE") == "3PT Field Goal").then(3).otherwise(2),
                value_v3=pl.col("shotValue").cast(pl.Int64),
                dx=pl.col("LOC_X").cast(pl.Float64) - pl.col("xLegacy").cast(pl.Float64),
                dy=pl.col("LOC_Y").cast(pl.Float64) - pl.col("yLegacy").cast(pl.Float64),
                dclock=(pl.col("MINUTES_REMAINING").cast(pl.Int64) * 60
                        + pl.col("SECONDS_REMAINING").cast(pl.Int64))
                       - clock_secs("clock").floor(),
            ))
    per_game = (j.group_by("gid")
                .agg(n_sd=pl.col("PLAYER_ID").is_not_null().sum(),
                     n_v3=pl.col("personId").is_not_null().sum()))
    games_differ = per_game.filter(pl.col("n_sd") != pl.col("n_v3")).height

    n_sd, n_v3, n = sd.height, v3.height, both.height
    row = {
        "season": label,
        "games_shotdetail": len(games_sd), "games_v3": len(games_v3),
        "v3_games_from_api": api_share(prefix),
        "shots_shotdetail": n_sd, "shots_v3": n_v3, "matched": n,
        "only_shotdetail": only_sd.height, "only_v3": only_v3.height,
        "games_count_differ": games_differ,
        "make_agree": float((both["make_sd"] == both["make_v3"]).mean()),
        "value_agree": float((both["value_sd"] == both["value_v3"]).mean()),
        "period_agree": float((both["PERIOD"] == both["period"]).mean()),
        "player_agree": float((both["PLAYER_ID"] == both["personId"]).mean()),
        "xy_exact": float(((both["dx"] == 0) & (both["dy"] == 0)).mean()),
        "clock_within_1s": float((both["dclock"].abs() <= 1).mean()),
    }
    gates = {
        "same games": games_sd == games_v3,
        f"unmatched <= {MAX_UNMATCHED:.2%} each side":
            only_sd.height <= MAX_UNMATCHED * n_sd and only_v3.height <= MAX_UNMATCHED * n_v3,
        f"games with a different FG count <= {MAX_GAMES_COUNT_DIFF:.1%}":
            games_differ <= MAX_GAMES_COUNT_DIFF * len(games_sd),
        "make/miss agrees on every shot": row["make_agree"] == 1.0,
        "shot value agrees on every shot": row["value_agree"] == 1.0,
        "period agrees on every shot": row["period_agree"] == 1.0,
        f"player agrees >= {MIN_PLAYER:.2%}": row["player_agree"] >= MIN_PLAYER,
        f"location exact >= {MIN_XY:.1%}": row["xy_exact"] >= MIN_XY,
        f"clock within 1 s >= {MIN_CLOCK:.1%}": row["clock_within_1s"] >= MIN_CLOCK,
    }
    row["passed"] = all(gates.values())

    cols = ["gid", "ev", "PLAYER_ID", "personId", "make_sd", "make_v3", "value_sd",
            "value_v3", "LOC_X", "xLegacy", "LOC_Y", "yLegacy", "dclock"]
    bad = (both.filter((pl.col("make_sd") != pl.col("make_v3"))
                       | (pl.col("value_sd") != pl.col("value_v3"))
                       | (pl.col("PLAYER_ID") != pl.col("personId"))
                       | (pl.col("dx") != 0) | (pl.col("dy") != 0)
                       | (pl.col("dclock").abs() > 1))
           .select(cols).with_columns(kind=pl.lit("disagree")))
    unmatched = pl.concat([
        only_sd.select(gid="gid", ev="ev", PLAYER_ID="PLAYER_ID").with_columns(kind=pl.lit("only_shotdetail")),
        only_v3.select(gid="gid", ev="ev", personId="personId").with_columns(kind=pl.lit("only_v3")),
    ], how="diagonal")
    detail = (pl.concat([bad.with_columns(pl.all().cast(pl.String)),
                         unmatched.with_columns(pl.all().cast(pl.String))], how="diagonal")
              .with_columns(season=pl.lit(label))
              .sort(["gid", "ev", "kind"]))

    print(f"{'PASS' if row['passed'] else 'FAIL'}  {label}: {n:,} of {n_sd:,} shotdetail shots "
          f"matched ({row['only_shotdetail']} only there, {row['only_v3']} only in V3; "
          f"{games_differ} games differ in count); make {row['make_agree']:.4%}, "
          f"value {row['value_agree']:.4%}, player {row['player_agree']:.4%}, "
          f"location {row['xy_exact']:.4%}, clock {row['clock_within_1s']:.4%}; "
          f"V3 from the API for {row['v3_games_from_api']:.0%} of games")
    for name, ok in gates.items():
        if not ok:
            print(f"      FAIL  {name}")
    return row, detail


def main(argv: list[str]) -> int:
    rows, details = [], []
    for label, stem, prefix in SEASONS:
        row, detail = compare(label, stem, prefix)
        rows.append(row)
        details.append(detail)
    table = pl.DataFrame(rows)
    detail = pl.concat(details, how="diagonal")
    ok = bool(table["passed"].all())

    if "--check" in argv:
        committed = pl.read_csv(OUT / "source_check.csv")
        same = table.equals(committed.select(table.columns).cast(table.schema))
        if not same:
            print("committed output/source_check.csv differs from a rerun")
        ok &= same
    else:
        table.write_csv(OUT / "source_check.csv")
        detail.write_csv(OUT / "source_disagreements.csv")

    print("SHOT SOURCE CHECK " + ("PASSED" if ok else "FAILED")
          + f" ({table.height} seasons against PlayByPlayV3; "
            f"{detail.height} disagreeing or unmatched shots listed in "
            f"output/source_disagreements.csv)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
