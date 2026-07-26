"""Season-out backtest: the 2023-24 model scored on seasons it never saw.

The core study's stated limitation was "in-sample calibration; a season-out
backtest is the natural extension." This closes it, on TWO held-out seasons
(2024-25 and 2025-26), and asks the three questions a backtest can answer
that a fit cannot:

  1. TRANSFER   — does 2023-24 shot pricing hold on unseen seasons?
                  (out-of-season decile calibration)
  2. STABILITY  — do the coefficients move when refit per season?
  3. PERSISTENCE — which player skills repeat year over year: shot
                  SELECTION (xPPS) or shot MAKING (points above expected)?
                  Each season's making uses its own refit model, so level
                  drift cannot masquerade as skill.

Python-only extension on top of the reconciled core model (02), reusing its
exact functions by import — the design matrix is data-independent by
construction, which is what makes cross-season scoring valid.

Run: python python/06_backtest.py   (after 02_model.py; harvests are cached)
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

SEASONS = {"2023-24": "shotdetail_2023", "2024-25": "shotdetail_2024",
           "2025-26": "shotdetail_2025"}
HELD_OUT = ["2024-25", "2025-26"]
MIN_FGA = 300


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


def log_loss(pred: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(pred, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> int:
    frames: dict[str, pl.DataFrame] = {}
    X: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    for season, name in SEASONS.items():
        frames[season] = prepare(DATA / f"{name}.csv")
        X[season], names = m.design_matrix(frames[season])
        y[season] = frames[season]["SHOT_MADE_FLAG"].to_numpy().astype(float)
        print(f"{season}: {frames[season].height:,} shots")

    beta23 = pl.read_csv(OUT / "coefficients.csv")["coef"].to_numpy()

    # ---- 1. transfer: 2023-24 pricing on unseen seasons ---------------------
    rows = []
    cal_frames = []
    for season in SEASONS:
        pred = m.predict_make(X[season], beta23)
        cal = m.decile_calibration(pred, y[season])
        cal_frames.append(cal.with_columns(season=pl.lit(season)))
        rows.append({
            "season": season,
            "held_out": season in HELD_OUT,
            "n": frames[season].height,
            "actual_make": float(y[season].mean()),
            "pred_make": float(pred.mean()),
            "worst_decile_gap": float((cal["mean_pred"] - cal["actual"])
                                      .abs().max()),
            "log_loss": log_loss(pred, y[season]),
        })
    metrics = pl.DataFrame(rows)
    metrics.write_csv(OUT / "backtest_metrics.csv")
    pl.concat(cal_frames).write_csv(OUT / "backtest_calibration.csv")
    print(metrics)

    # ---- 2. stability: refit per season, compare coefficients ---------------
    betas = {"2023-24": beta23}
    for season in HELD_OUT:
        betas[season] = m.logistic_irls(X[season], y[season])
    drift = pl.DataFrame({
        "term": names,
        **{f"coef_{s}": betas[s] for s in SEASONS},
    }).with_columns(
        max_drift=pl.max_horizontal(
            (pl.col("coef_2024-25") - pl.col("coef_2023-24")).abs(),
            (pl.col("coef_2025-26") - pl.col("coef_2023-24")).abs()))
    drift.write_csv(OUT / "coefficient_drift.csv")

    # ---- 3. persistence: which shot skill repeats? --------------------------
    tables = {}
    for season in SEASONS:
        pred_own = m.predict_make(X[season], betas[season])
        f = frames[season].with_columns(
            xpts=pl.Series(pred_own) * pl.col("value"))
        tables[season] = (m.player_table(f)
                          .filter(pl.col("fga") >= MIN_FGA)
                          .select("PLAYER_ID", "PLAYER_NAME", "fga", "pps",
                                  "xpps", "making_100"))

    pairs = [("2023-24", "2024-25"), ("2024-25", "2025-26")]
    prows = []
    for s1, s2 in pairs:
        j = tables[s1].join(tables[s2], on="PLAYER_ID", suffix="_2")
        for metric in ["xpps", "making_100", "pps"]:
            a = j[metric].to_numpy()
            b = j[f"{metric}_2"].to_numpy()
            prows.append({"pair": f"{s1} to {s2}", "metric": metric,
                          "r": float(np.corrcoef(a, b)[0, 1]), "n": j.height})
    persistence = pl.DataFrame(prows)
    persistence.write_csv(OUT / "persistence.csv")
    print(persistence)

    # ---- 4. clock decomposition replication on unseen seasons ---------------
    clock_frames = []
    for season in SEASONS:
        pred_own = m.predict_make(X[season], betas[season])
        f = frames[season].with_columns(
            xpts=pl.Series(pred_own) * pl.col("value"))
        clock_frames.append(m.clock_decomposition(f)
                            .with_columns(season=pl.lit(season)))
    pl.concat(clock_frames).write_csv(OUT / "clock_backtest.csv")

    # ---- gates --------------------------------------------------------------
    totals24 = m.pbp_shot_totals(DATA / "nbastats_2024.csv")
    f24 = frames["2024-25"]
    fga24, fgm24 = f24.height, int(f24["SHOT_MADE_FLAG"].sum())
    fg3m24 = f24.filter((pl.col("value") == 3)
                        & (pl.col("SHOT_MADE_FLAG") == 1)).height
    worst_oos = float(metrics.filter(pl.col("held_out"))
                      ["worst_decile_gap"].max())
    checks = pl.DataFrame({
        "check": [
            "2024-25 FGA matches independent play-by-play (relative)",
            "2024-25 FGM matches independent play-by-play (relative)",
            "2024-25 3PM matches independent play-by-play (relative)",
            "worst OUT-OF-SEASON decile gap (2023-24 model)",
        ],
        "value": [
            abs(fga24 - totals24["fga"]) / totals24["fga"],
            abs(fgm24 - totals24["fgm"]) / totals24["fgm"],
            abs(fg3m24 - totals24["fg3m"]) / totals24["fg3m"],
            worst_oos,
        ],
        "threshold": [2e-3, 2e-3, 2e-3, 0.02],
    }).with_columns(passed=pl.col("value") <= pl.col("threshold"))
    checks.write_csv(OUT / "backtest_validation.csv")
    print(checks)
    print("note: no bulk play-by-play exists yet for 2025-26, so its totals "
          "have no independent cross-check; reported, not gated.")

    # ---- figure: does shot-making repeat? -----------------------------------
    import plotly.graph_objects as go
    j = tables["2023-24"].join(tables["2024-25"], on="PLAYER_ID", suffix="_2")
    fig = go.Figure(go.Scatter(
        x=j["making_100"], y=j["making_100_2"], mode="markers",
        text=j["PLAYER_NAME"],
        hovertemplate="%{text}<br>2023-24 %{x:+.1f}<br>2024-25 %{y:+.1f}"
        "<extra></extra>"))
    r_mk = persistence.filter((pl.col("pair") == "2023-24 to 2024-25")
                              & (pl.col("metric") == "making_100"))["r"][0]
    fig.update_layout(
        title=f"Does shot-making repeat? 2023-24 vs 2024-25 (r = {r_mk:.2f}, "
              f"players with {MIN_FGA}+ FGA both seasons)",
        xaxis_title="2023-24 points above expected /100",
        yaxis_title="2024-25 points above expected /100",
        template="plotly_white")
    fig.write_html(FIG / "fig3_persistence_py.html", include_plotlyjs="cdn")

    ok = bool(checks["passed"].all())
    print("BACKTEST VALIDATION PASSED" if ok else "BACKTEST VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
