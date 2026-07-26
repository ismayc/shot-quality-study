"""G League extension: the same shot-pricing model, one league down.

Differentiator: G League data is public through the same ShotChartDetail
schema (league_id="20") but almost nobody uses it. Two questions:

  1. TRANSFER — does the NBA-fit model's shot pricing transfer? (apply NBA
     coefficients to G League shots, read the calibration)
  2. ENVIRONMENT — how do the leagues differ, priced on identical shot
     definitions? xPPS(NBA model | G League shot mix) vs actual G League
     PPS isolates a per-100 execution gap between the leagues, holding the
     shot profile fixed — a number the raw FG% comparison cannot give.

Python-only exploration on top of the reconciled core model (02): the
dual-implementation discipline covers the model itself; this reuses those
exact functions by import.

Run: python python/05_gleague.py    (after 02_model.py)
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

spec = importlib.util.spec_from_file_location(
    "shotq_model", ROOT / "python" / "02_model.py")
m = importlib.util.module_from_spec(spec)
sys.modules["shotq_model"] = m
spec.loader.exec_module(m)


def harvest() -> Path:
    out = DATA / "gleague_shotdetail_2023.csv"
    if out.exists():
        print(f"already present: {out}")
        return out
    from nba_api.stats.library import http as nba_http
    nba_http.STATS_HEADERS["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    from nba_api.stats.endpoints import shotchartdetail
    df = shotchartdetail.ShotChartDetail(
        team_id=0, player_id=0, context_measure_simple="FGA",
        season_nullable="2023-24", league_id="20",
        timeout=60).get_data_frames()[0]
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df):,} shots")
    return out


def prepare(path: Path) -> pl.DataFrame:
    return (pl.read_csv(path, infer_schema_length=0)
            .with_columns(pl.col("SHOT_DISTANCE", "LOC_X", "LOC_Y",
                                 "SHOT_MADE_FLAG").cast(pl.Int64))
            .with_columns(
                value=pl.when(pl.col("SHOT_TYPE") == "3PT Field Goal")
                .then(3).otherwise(2),
                action_family=pl.col("ACTION_TYPE")
                .map_elements(m.action_family, return_dtype=pl.String))
            .with_columns(points=pl.col("SHOT_MADE_FLAG") * pl.col("value")))


def league_row(name: str, df: pl.DataFrame, pred: np.ndarray) -> dict:
    return {
        "league": name,
        "n_shots": df.height,
        "pps": float(df["points"].mean()),
        "xpps_nba_model": float((pred * df["value"].to_numpy()).mean()),
        "pct_three": float((df["value"] == 3).mean()),
        "pct_rim": float((df["SHOT_ZONE_BASIC"] == "Restricted Area").mean()),
    }


def main() -> int:
    gl = prepare(harvest())
    nba = prepare(DATA / "shotdetail_2023.csv")

    beta = pl.read_csv(OUT / "coefficients.csv")["coef"].to_numpy()
    Xg, _ = m.design_matrix(gl)
    Xn, _ = m.design_matrix(nba)
    pg = m.predict_make(Xg, beta)
    pn = m.predict_make(Xn, beta)

    rows = [league_row("NBA", nba, pn), league_row("G League", gl, pg)]
    comp = pl.DataFrame(rows).with_columns(
        execution_gap_100=(pl.col("pps") - pl.col("xpps_nba_model")) * 100)
    comp.write_csv(OUT / "gleague_comparison.csv")
    print(comp)

    cal = m.decile_calibration(pg, gl["SHOT_MADE_FLAG"].to_numpy().astype(float))
    cal.write_csv(OUT / "gleague_calibration.csv")
    worst = float((cal["mean_pred"] - cal["actual"]).abs().max())
    print(f"NBA-model-on-G-League worst decile gap: {worst:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
