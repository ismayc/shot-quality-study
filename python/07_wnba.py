"""WNBA: the identical pipeline, refit on a league almost nobody models.

The shufinskiy archive carries WNBA ShotChartDetail back to 1997 in the same
schema, so the study's feature engineering, IRLS, and player decomposition
run unchanged. Refit per WNBA season (2024, 2025 — the two most recent
complete seasons; the WNBA's shorter 3-point line and shot vocabulary land
in the refit coefficients, so nothing is priced by NBA geometry).

Questions:
  1. STRUCTURE  — how do the leagues' shot diets and per-zone conversion
                  actually differ, on identical definitions?
  2. PLAYERS    — selection vs making decomposition for WNBA players
                  (face-validity check on names).
  3. INVARIANCE — does the NBA persistence result (selection repeats,
                  making half-repeats) hold in a different league? If yes,
                  that is a statement about basketball, not about the NBA.

Python-only extension on top of the reconciled core model (02).

Run: python python/07_wnba.py    (harvests are cached by 01)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
FIG = ROOT / "figures"

spec = importlib.util.spec_from_file_location(
    "shotq_model", ROOT / "python" / "02_model.py")
m = importlib.util.module_from_spec(spec)
sys.modules["shotq_model"] = m
spec.loader.exec_module(m)

WNBA_SEASONS = ["2024", "2025"]
MIN_FGA_W = 200          # 40-44 game seasons; 200 FGA is a rotation player
NBA_REF = "shotdetail_2024"   # 2024-25, most recent NBA season with pbp gate


def prepare(path: Path) -> pl.DataFrame:
    return (pl.read_csv(path, infer_schema_length=0)
            .with_columns(pl.col("SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD",
                                 "MINUTES_REMAINING", "SECONDS_REMAINING",
                                 "SHOT_MADE_FLAG", "PLAYER_ID").cast(pl.Int64))
            .with_columns(
                value=pl.when(pl.col("SHOT_TYPE") == "3PT Field Goal")
                .then(3).otherwise(2),
                action_family=pl.col("ACTION_TYPE")
                .map_elements(m.action_family, return_dtype=pl.String))
            .with_columns(points=pl.col("SHOT_MADE_FLAG") * pl.col("value")))


def fit_season(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, _ = m.design_matrix(df)
    y = df["SHOT_MADE_FLAG"].to_numpy().astype(float)
    beta = m.logistic_irls(X, y)
    return beta, m.predict_make(X, beta)


def structure_row(league: str, df: pl.DataFrame) -> dict:
    three = df.filter(pl.col("value") == 3)
    corner = three.filter(pl.col("SHOT_ZONE_BASIC")
                          .is_in(["Left Corner 3", "Right Corner 3"]))
    return {
        "league": league,
        "n_shots": df.height,
        "pps": float(df["points"].mean()),
        "make": float(df["SHOT_MADE_FLAG"].mean()),
        "pct_three": float((df["value"] == 3).mean()),
        "pct_rim": float((df["SHOT_ZONE_BASIC"] == "Restricted Area").mean()),
        "pct_midrange": float((df["SHOT_ZONE_BASIC"] == "Mid-Range").mean()),
        "rim_make": float(df.filter(pl.col("SHOT_ZONE_BASIC")
                                    == "Restricted Area")
                          ["SHOT_MADE_FLAG"].mean()),
        "three_make": float(three["SHOT_MADE_FLAG"].mean()),
        "corner_share_of_3s": float(corner.height / three.height),
        "corner_3_make": float(corner["SHOT_MADE_FLAG"].mean()),
        "dunks_per_1000": float((df["action_family"] == "dunk").mean() * 1000),
    }


def main() -> int:
    wnba = {s: prepare(DATA / f"wnba_shotdetail_{s}.csv")
            for s in WNBA_SEASONS}
    nba = prepare(DATA / f"{NBA_REF}.csv")
    for s, df in wnba.items():
        print(f"WNBA {s}: {df.height:,} shots, {df['GAME_ID'].n_unique()} games")

    # ---- 1. structure -------------------------------------------------------
    struct = pl.DataFrame([
        structure_row("NBA 2024-25", nba),
        *[structure_row(f"WNBA {s}", wnba[s]) for s in WNBA_SEASONS],
    ])
    struct.write_csv(OUT / "wnba_league_structure.csv")
    print(struct)

    # ---- 2. per-season refits, player tables, calibration -------------------
    tables = {}
    for s in WNBA_SEASONS:
        beta, pred = fit_season(wnba[s])
        y = wnba[s]["SHOT_MADE_FLAG"].to_numpy().astype(float)
        cal = m.decile_calibration(pred, y)
        cal.with_columns(season=pl.lit(s)).write_csv(
            OUT / f"wnba_calibration_{s}.csv")
        f = wnba[s].with_columns(xpts=pl.Series(pred) * pl.col("value"))
        tables[s] = m.player_table(f).filter(pl.col("fga") >= MIN_FGA_W)
    tables[WNBA_SEASONS[-1]].write_csv(OUT / "wnba_player_table.csv")

    # ---- 3. persistence, same design as the NBA backtest --------------------
    j = tables["2024"].join(tables["2025"], on="PLAYER_ID", suffix="_2")
    prows = []
    for metric in ["xpps", "making_100", "pps"]:
        a, b = j[metric].to_numpy(), j[f"{metric}_2"].to_numpy()
        prows.append({"pair": "WNBA 2024 to 2025", "metric": metric,
                      "r": float(np.corrcoef(a, b)[0, 1]), "n": j.height})
    persistence = pl.DataFrame(prows)
    persistence.write_csv(OUT / "wnba_persistence.csv")
    print(persistence)

    # ---- gates --------------------------------------------------------------
    totals = m.pbp_shot_totals(DATA / "wnba_nbastats_2024.csv")
    f24 = wnba["2024"]
    fga, fgm = f24.height, int(f24["SHOT_MADE_FLAG"].sum())
    fg3m = f24.filter((pl.col("value") == 3)
                      & (pl.col("SHOT_MADE_FLAG") == 1)).height
    worst = max(
        float((pl.read_csv(OUT / f"wnba_calibration_{s}.csv")
               .select((pl.col("mean_pred") - pl.col("actual")).abs().max()))
              .item()) for s in WNBA_SEASONS)
    checks = pl.DataFrame({
        "check": [
            "WNBA 2024 FGA matches independent play-by-play (relative)",
            "WNBA 2024 FGM matches independent play-by-play (relative)",
            "WNBA 2024 3PM matches independent play-by-play (relative)",
            "worst WNBA decile calibration gap (own-season fits)",
        ],
        "value": [
            abs(fga - totals["fga"]) / totals["fga"],
            abs(fgm - totals["fgm"]) / totals["fgm"],
            abs(fg3m - totals["fg3m"]) / totals["fg3m"],
            worst,
        ],
        "threshold": [2e-3, 2e-3, 2e-3, 0.02],
    }).with_columns(passed=pl.col("value") <= pl.col("threshold"))
    checks.write_csv(OUT / "wnba_validation.csv")
    print(checks)

    # ---- figure: shot diet by league ---------------------------------------
    import plotly.graph_objects as go
    cats = ["pct_rim", "pct_midrange", "pct_three"]
    labels = ["Rim (RA)", "Mid-range", "Threes"]
    fig = go.Figure()
    for r in struct.iter_rows(named=True):
        fig.add_trace(go.Bar(name=r["league"], x=labels,
                             y=[r[c] for c in cats]))
    fig.update_layout(barmode="group", title="Shot diet by league",
                      yaxis_title="share of all FGA", yaxis_tickformat=".0%",
                      template="plotly_white")
    fig.write_html(FIG / "fig4_wnba_shot_diet_py.html", include_plotlyjs="cdn")

    ok = bool(checks["passed"].all())
    print("WNBA VALIDATION PASSED" if ok else "WNBA VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
