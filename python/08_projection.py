"""Empirical-Bayes projection: turning the persistence finding into a rule.

The backtest (06) established WHAT repeats: shot selection (xPPS) at r≈.9,
shot making at r≈.6 — the diet is nearly a fixed trait, making is half
noise. This script turns that into the projection rule the finding implies
— "trust the diet, regress the making" — and measures it out of sample:

  proj PPS(next) = xPPS(this season)  +  EB-shrunken making(this season)

The making shrinkage is classic empirical Bayes on a normal-normal model.
Each player's observed making_100 comes with an analytic SE from
player_table(); the league's true-skill spread tau^2 is estimated by the
method of moments (observed variance minus mean sampling variance), and
each player is pulled toward the league mean with weight
tau^2 / (tau^2 + SE_i^2) — high-volume shooters keep their number, small
samples get regressed hard.

The family it competes against brackets it exactly:
  weight = 1 for everyone  ->  naive carry-forward of raw PPS
  weight = 0 for everyone  ->  diet-only (xPPS + league-mean making)
EB chooses the weight per player from the data. Validated on both season
pairs (2023-24 -> 2024-25, 2024-25 -> 2025-26), players with 300+ FGA in
both seasons, each season's making measured under its own refit model
(same protocol as 06, so level drift cannot leak in).

Python-only extension on top of the reconciled core model (02), like 06/07.

Run: python python/08_projection.py   (after 02_model.py; data cached)
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
PAIRS = [("2023-24", "2024-25"), ("2024-25", "2025-26")]
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


def eb_posterior(obs: np.ndarray, se: np.ndarray
                 ) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Normal-normal empirical Bayes: shrink each observation toward the
    population mean by its reliability.

    Returns (posterior means, population mean mu, true-skill variance tau2,
    per-observation weights). tau2 by method of moments, floored at 0 — if
    the observed spread is all sampling noise, everyone shrinks to mu.
    """
    mu = float(obs.mean())
    tau2 = max(0.0, float(obs.var(ddof=1)) - float((se ** 2).mean()))
    w = tau2 / (tau2 + se ** 2)
    return mu + w * (obs - mu), mu, tau2, w


def error_row(pair: str, method: str, pred: np.ndarray, actual: np.ndarray
              ) -> dict:
    return {"pair": pair, "method": method,
            "mae": float(np.abs(pred - actual).mean()),
            "rmse": float(np.sqrt(((pred - actual) ** 2).mean())),
            "r": float(np.corrcoef(pred, actual)[0, 1]),
            "n": len(actual)}


def main() -> int:
    # Per-season player tables under each season's own refit model (06's
    # protocol; 2023-24 uses the committed reconciled coefficients).
    tables: dict[str, pl.DataFrame] = {}
    for season, name in SEASONS.items():
        df = prepare(DATA / f"{name}.csv")
        X, _ = m.design_matrix(df)
        y = df["SHOT_MADE_FLAG"].to_numpy().astype(float)
        if season == "2023-24":
            beta = pl.read_csv(OUT / "coefficients.csv")["coef"].to_numpy()
        else:
            beta = m.logistic_irls(X, y)
        f = df.with_columns(xpts=pl.Series(m.predict_make(X, beta))
                            * pl.col("value"))
        tables[season] = m.player_table(f).filter(pl.col("fga") >= MIN_FGA)
        print(f"{season}: {tables[season].height} players with "
              f"{MIN_FGA}+ FGA")

    proj_frames, metric_rows, hyper_rows = [], [], []
    for s1, s2 in PAIRS:
        pair = f"{s1} to {s2}"
        # Hyperparameters from the full source-season qualifier pool ...
        pool = tables[s1]
        _, mu, tau2, _ = eb_posterior(pool["making_100"].to_numpy(),
                                      pool["making_100_se"].to_numpy())
        hyper_rows.append({"season": s1, "mu_making_100": mu, "tau2": tau2,
                           "tau": float(np.sqrt(tau2)),
                           "n_pool": pool.height})
        # ... applied to the players observable in both seasons.
        j = (pool.join(tables[s2].select("PLAYER_ID", "pps"),
                       on="PLAYER_ID", suffix="_next"))
        obs = j["making_100"].to_numpy()
        se = j["making_100_se"].to_numpy()
        w = tau2 / (tau2 + se ** 2)
        making_eb = mu + w * (obs - mu)
        xpps = j["xpps"].to_numpy()
        actual = j["pps_next"].to_numpy()
        preds = {
            "eb (diet + shrunken making)": xpps + making_eb / 100,
            "naive carry-forward (raw pps)": j["pps"].to_numpy(),
            "diet-only (xpps + league making)": xpps + mu / 100,
        }
        for method, pred in preds.items():
            metric_rows.append(error_row(pair, method, pred, actual))
        proj_frames.append(j.select("PLAYER_ID", "PLAYER_NAME", "fga", "pps",
                                    "xpps", "making_100", "making_100_se")
                           .with_columns(pair=pl.lit(pair),
                                         weight=pl.Series(w),
                                         making_eb=pl.Series(making_eb),
                                         proj_pps=pl.Series(preds[
                                             "eb (diet + shrunken making)"]),
                                         actual_pps_next=pl.Series(actual)))

    pl.concat(proj_frames).write_csv(OUT / "projection.csv")
    metrics = pl.DataFrame(metric_rows)
    metrics.write_csv(OUT / "projection_metrics.csv")
    hyper = pl.DataFrame(hyper_rows)
    hyper.write_csv(OUT / "projection_hyperparams.csv")
    print(hyper)
    with pl.Config(fmt_str_lengths=40):
        print(metrics)

    # ---- gates --------------------------------------------------------------
    def mae(pair: str, method_prefix: str) -> float:
        return float(metrics.filter(
            (pl.col("pair") == pair)
            & pl.col("method").str.starts_with(method_prefix))["mae"][0])

    weights_all = np.concatenate(
        [f["weight"].to_numpy() for f in proj_frames])
    rows = []
    for s1, s2 in PAIRS:
        pair = f"{s1} to {s2}"
        rows.append({
            "check": f"EB beats naive carry-forward, {pair} (MAE ratio)",
            "value": mae(pair, "eb") / mae(pair, "naive"),
            "threshold": 1.0})
        rows.append({
            "check": f"EB beats or ties diet-only, {pair} (MAE ratio)",
            "value": mae(pair, "eb") / mae(pair, "diet-only"),
            "threshold": 1.0})
    rows.append({"check": "true-skill variance tau2 > 0 in both source "
                          "seasons (negated)",
                 "value": -float(hyper["tau2"].min()), "threshold": 0.0})
    rows.append({"check": "shrinkage weights strictly inside (0, 1) "
                          "(max weight)",
                 "value": float(weights_all.max()), "threshold": 1.0 - 1e-9})
    checks = pl.DataFrame(rows).with_columns(
        passed=pl.col("value") <= pl.col("threshold"))
    checks.write_csv(OUT / "projection_validation.csv")
    with pl.Config(fmt_str_lengths=70):
        print(checks)

    # ---- figure: shrinkage in action ---------------------------------------
    import plotly.graph_objects as go
    f1 = proj_frames[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f1["making_100"], y=f1["making_eb"], mode="markers",
        marker=dict(size=np.sqrt(f1["fga"].to_numpy()) / 3,
                    color=f1["weight"], colorscale="Viridis",
                    colorbar=dict(title="EB weight"), showscale=True),
        text=[f"{n} · {g} FGA · w={w:.2f}" for n, g, w in
              zip(f1["PLAYER_NAME"], f1["fga"], f1["weight"])],
        hovertemplate="%{text}<br>observed %{x:+.1f} -> EB %{y:+.1f}"
        "<extra></extra>"))
    lim = float(np.abs(f1["making_100"].to_numpy()).max()) * 1.05
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                             line=dict(dash="dot", color="gray"),
                             showlegend=False))
    fig.update_layout(
        title="Empirical-Bayes shrinkage of shot-making, 2023-24 "
              "(marker size = volume; the diagonal is 'believe the raw "
              "number')",
        xaxis_title="observed making, pts above expected /100 shots",
        yaxis_title="EB posterior making /100",
        template="plotly_white")
    fig.write_html(FIG / "fig5_eb_projection_py.html", include_plotlyjs="cdn")

    ok = bool(checks["passed"].all())
    print("PROJECTION VALIDATION PASSED" if ok
          else "PROJECTION VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
