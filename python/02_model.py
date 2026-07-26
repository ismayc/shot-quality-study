"""Shot quality (xMake / xPTS) model for every 2023-24 field-goal attempt.

The public descendant of the qSQ/qSI idea (Chang et al., Sloan 2014) that the
public data can actually support: model each shot's make probability from
WHERE it is taken and WHAT KIND of shot it is — deliberately excluding any
game-clock context — then read three things off the fitted model:

  xPTS        expected points of a shot     = p(make) x shot value
  xPPS        a player's/situation's shot SELECTION quality (mix of shots)
  shot-making actual minus expected points per 100 shots — EXECUTION above
              the location/type baseline

Excluding the clock from the model is the design decision that makes the
end-of-period analysis meaningful: the model says what the shot mix should
yield; the residual says how execution changes under the clock. That
decomposition (selection vs execution) is what the eFG-only proxy in
../playbyplay-study could not separate.

Model: unpenalized-in-spirit logistic regression fit by hand-rolled IRLS
(Newton) with a 1e-8 identity stabilizer — implemented identically in R
(../R/02_model.R) so coefficients reconcile to numeric tolerance rather than
"close enough". Features: shot-zone dummies, distance linear splines, angle
from the rim, heave flag, and nine deterministic action families.

Run: python python/02_model.py     (after 01_harvest_shots.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
FIG = ROOT / "figures"
PBP_BULK = ROOT.parent / "lineup-valuation-study" / "data" / "pbp_bulk" / "nbastats_2023.csv"

DIST_KNOTS = [1.0, 3.0, 6.0, 10.0, 16.0, 22.0, 26.0]
DIST_CAP = 35.0
HEAVE_FT = 30
MIN_FGA_TABLE = 300

# (family, keywords) — FIRST match wins, so order is part of the definition.
# The R implementation must keep the same order.
ACTION_FAMILIES = [
    ("putback", ["Tip", "Putback"]),
    ("alley_oop", ["Alley Oop"]),
    ("dunk", ["Dunk"]),
    ("hook", ["Hook"]),
    ("floater", ["Floating"]),
    ("layup", ["Layup"]),
    ("off_dribble", ["Pullup", "Pull-Up", "Step Back"]),
    ("post_jumper", ["Fadeaway", "Turnaround"]),
    ("jumper", []),                       # reference level / fallback
]
ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Above the Break 3",
         "Left Corner 3", "Right Corner 3", "Backcourt"]  # ref: Mid-Range
INTERACT_DIST = ["putback", "hook", "floater", "layup"]   # own distance slope
MOVE_MODIFIERS = ["Driving", "Cutting", "Running"]        # cross-family flags
AREAS = ["Left Side(L)", "Left Side Center(LC)", "Right Side Center(RC)",
         "Right Side(R)", "Back Court(BC)"]               # ref: Center(C)


def action_family(action_type: str) -> str:
    """Map a raw ACTION_TYPE string to its family (first keyword match wins)."""
    for family, keywords in ACTION_FAMILIES:
        if any(k in action_type for k in keywords):
            return family
    return "jumper"


def shot_angle(loc_x: np.ndarray, loc_y: np.ndarray) -> np.ndarray:
    """Absolute angle from the straight-on line to the rim, radians in [0, pi].

    atan2(|x|, y): 0 = dead center in front of the rim, pi/2 = the baseline,
    > pi/2 = behind the backboard.
    """
    return np.arctan2(np.abs(loc_x), loc_y)


def design_matrix(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Deterministic model matrix; column order is part of the contract."""
    n = df.height
    dist = np.minimum(df["SHOT_DISTANCE"].to_numpy().astype(float), DIST_CAP)
    cols: list[np.ndarray] = [np.ones(n), dist]
    names = ["intercept", "dist"]
    for k in DIST_KNOTS:
        cols.append(np.maximum(dist - k, 0.0))
        names.append(f"dist_gt{k:g}")
    cols.append((df["SHOT_DISTANCE"].to_numpy() >= HEAVE_FT).astype(float))
    names.append("heave")
    cols.append(shot_angle(df["LOC_X"].to_numpy().astype(float),
                           df["LOC_Y"].to_numpy().astype(float)))
    names.append("angle")
    zone = df["SHOT_ZONE_BASIC"].to_numpy()
    for z in ZONES:
        cols.append((zone == z).astype(float))
        names.append("zone_" + z.lower().replace(" ", "_").replace("(", "").replace(")", ""))
    fam = df["action_family"].to_numpy()
    for f, _ in ACTION_FAMILIES[:-1]:                    # jumper = reference
        cols.append((fam == f).astype(float))
        names.append("act_" + f)
    # Close-range families keep their own distance slope: a 5-ft putback is
    # not a 5-ft jump shot, and without these the mid-probability deciles
    # miscalibrate by ~2 points.
    for f in INTERACT_DIST:
        cols.append((fam == f).astype(float) * dist)
        names.append(f"act_{f}_x_dist")
    # On-the-move modifiers cut across families (a driving layup is not a
    # cutting layup at the same spot); without them decile 7 misses by ~1pt.
    action = df["ACTION_TYPE"].to_numpy()
    for mod in MOVE_MODIFIERS:
        cols.append(np.char.find(action.astype(str), mod) >= 0)
        names.append("mod_" + mod.lower())
    area = df["SHOT_ZONE_AREA"].to_numpy()
    for a in AREAS:                                       # ref: Center(C)
        cols.append(area == a)
        names.append("area_" + a.split("(")[0].strip().lower().replace(" ", "_"))
    return np.column_stack(cols).astype(float), names


def logistic_irls(X: np.ndarray, y: np.ndarray, *, ridge: float = 1e-8,
                  tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
    """Newton/IRLS for logistic regression; tiny ridge only stabilizes the
    solve. Deterministic: same data -> same coefficients to ~1e-12, which is
    what lets R and Python reconcile exactly instead of approximately."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def predict_make(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


def decile_calibration(pred: np.ndarray, made: np.ndarray) -> pl.DataFrame:
    """Mean predicted vs actual make rate by prediction decile.

    Rank-based; predictions are rounded to 10 decimals before the stable
    sort so R and Python bin identically despite ~1e-14 fit differences.
    """
    order = np.argsort(np.round(pred, 10), kind="stable")
    decile = np.empty(len(pred), dtype=int)
    decile[order] = (np.arange(len(pred)) * 10) // len(pred)
    return (pl.DataFrame({"decile": decile + 1, "pred": pred, "made": made})
            .group_by("decile")
            .agg(n=pl.len(), mean_pred=pl.col("pred").mean(),
                 actual=pl.col("made").mean())
            .sort("decile"))


def player_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-player selection (xPPS) vs execution (shot-making per 100 shots),
    with an analytic SE for the making metric."""
    return (df.group_by("PLAYER_ID", "PLAYER_NAME")
            .agg(fga=pl.len(),
                 pts=pl.col("points").sum(),
                 xpts=pl.col("xpts").sum(),
                 making_sd=(pl.col("points") - pl.col("xpts")).std())
            .with_columns(pps=pl.col("pts") / pl.col("fga"),
                          xpps=pl.col("xpts") / pl.col("fga"),
                          making_100=(pl.col("pts") - pl.col("xpts"))
                          / pl.col("fga") * 100)
            .with_columns(making_100_se=pl.col("making_sd")
                          / pl.col("fga").sqrt() * 100)
            .drop("making_sd")
            .sort("making_100", descending=True))


def clock_decomposition(df: pl.DataFrame) -> pl.DataFrame:
    """Selection vs execution by seconds left in the period.

    Buckets chosen for the questions they answer: the final-4s rush, the
    25-36s two-for-one launch window, and everything between.
    """
    secs = pl.col("MINUTES_REMAINING") * 60 + pl.col("SECONDS_REMAINING")
    bucket = (pl.when(secs <= 4).then(pl.lit("00-04s (rush)"))
              .when(secs <= 24).then(pl.lit("05-24s"))
              .when(secs <= 36).then(pl.lit("25-36s (2-for-1 window)"))
              .otherwise(pl.lit("37s+")))
    return (df.with_columns(bucket=bucket)
            .group_by("bucket")
            .agg(n=pl.len(),
                 pps=pl.col("points").mean(),
                 xpps=pl.col("xpts").mean())
            .with_columns(execution_100=(pl.col("pps") - pl.col("xpps")) * 100)
            .sort("bucket"))


def pbp_shot_totals(pbp_path: Path) -> dict[str, int]:
    """Independent cross-dataset check: FGA/FGM/3PM counted from the season
    play-by-play file (different upstream feed than ShotChartDetail)."""
    pbp = pl.scan_csv(pbp_path, infer_schema_length=0).select(
        pl.col("EVENTMSGTYPE").cast(pl.Int64),
        pl.col("HOMEDESCRIPTION"), pl.col("VISITORDESCRIPTION")).collect()
    made = pbp.filter(pl.col("EVENTMSGTYPE") == 1)
    miss = pbp.filter(pl.col("EVENTMSGTYPE") == 2)
    three = made.filter(
        pl.col("HOMEDESCRIPTION").str.contains("3PT").fill_null(False)
        | pl.col("VISITORDESCRIPTION").str.contains("3PT").fill_null(False))
    return {"fga": made.height + miss.height, "fgm": made.height,
            "fg3m": three.height}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    shots = (pl.read_csv(DATA / "shotdetail_2023.csv", infer_schema_length=0)
             .with_columns(pl.col("SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD",
                                  "MINUTES_REMAINING", "SECONDS_REMAINING",
                                  "SHOT_MADE_FLAG", "PLAYER_ID").cast(pl.Int64))
             .with_columns(
                 value=pl.when(pl.col("SHOT_TYPE") == "3PT Field Goal")
                 .then(3).otherwise(2),
                 action_family=pl.col("ACTION_TYPE")
                 .map_elements(action_family, return_dtype=pl.String))
             .with_columns(points=pl.col("SHOT_MADE_FLAG") * pl.col("value")))
    print(f"{shots.height:,} shots, made rate {shots['SHOT_MADE_FLAG'].mean():.4f}")

    X, names = design_matrix(shots)
    y = shots["SHOT_MADE_FLAG"].to_numpy().astype(float)
    beta = logistic_irls(X, y)
    pred = predict_make(X, beta)
    shots = shots.with_columns(xmake=pl.Series(pred),
                               xpts=pl.Series(pred) * pl.col("value"))

    pl.DataFrame({"term": names, "coef": beta}).write_csv(OUT / "coefficients.csv")

    cal = decile_calibration(pred, y)
    cal.write_csv(OUT / "calibration.csv")

    players = player_table(shots)
    players.write_csv(OUT / "player_table.csv")

    clock = clock_decomposition(shots)
    clock.write_csv(OUT / "clock_decomposition.csv")

    # ---- validation gates ---------------------------------------------------
    totals = pbp_shot_totals(PBP_BULK)
    fga, fgm = shots.height, int(shots["SHOT_MADE_FLAG"].sum())
    fg3m = shots.filter((pl.col("value") == 3)
                        & (pl.col("SHOT_MADE_FLAG") == 1)).height
    cal_gap = float((cal["mean_pred"] - cal["actual"]).abs().max())
    league_gap = abs(float(shots["xpts"].mean()) - float(shots["points"].mean()))
    checks = pl.DataFrame({
        "check": [
            "FGA matches play-by-play count (relative)",
            "FGM matches play-by-play count (relative)",
            "3PM matches play-by-play count (relative)",
            "max decile |predicted - actual| make rate",
            "league mean xPTS equals mean actual PTS",
        ],
        "value": [
            abs(fga - totals["fga"]) / totals["fga"],
            abs(fgm - totals["fgm"]) / totals["fgm"],
            abs(fg3m - totals["fg3m"]) / totals["fg3m"],
            cal_gap,
            league_gap,
        ],
        "threshold": [2e-3, 2e-3, 2e-3, 0.01, 0.005],
    }).with_columns(passed=pl.col("value") <= pl.col("threshold"))
    checks.write_csv(OUT / "validation.csv")
    print(checks)

    # ---- figures ------------------------------------------------------------
    import plotly.graph_objects as go

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["actual"],
                            mode="markers+lines", name="deciles"))
    f1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                            line={"dash": "dash"}))
    f1.update_layout(title="Calibration: predicted vs actual make rate by decile",
                     xaxis_title="mean predicted P(make)",
                     yaxis_title="actual make rate", template="plotly_white",
                     # bottom-right is always empty under a y=x diagonal, so
                     # the legend can never collide with the traces there
                     legend=dict(x=0.98, y=0.04, xanchor="right",
                                 yanchor="bottom",
                                 bgcolor="rgba(255,255,255,0.75)"))
    f1.write_html(FIG / "fig1_calibration_py.html", include_plotlyjs="cdn")

    big = players.filter(pl.col("fga") >= MIN_FGA_TABLE)
    f2 = go.Figure(go.Scatter(
        x=big["xpps"], y=big["making_100"], mode="markers",
        text=big["PLAYER_NAME"], hovertemplate="%{text}<br>xPPS %{x:.3f}"
        "<br>making %{y:+.1f}/100<extra></extra>"))
    f2.update_layout(title="Shot selection (xPPS) vs shot-making above expected "
                     f"(≥{MIN_FGA_TABLE} FGA, 2023-24)",
                     xaxis_title="expected points per shot (selection)",
                     yaxis_title="points above expected per 100 shots (execution)",
                     template="plotly_white")
    f2.write_html(FIG / "fig2_selection_vs_making_py.html", include_plotlyjs="cdn")
    try:
        f2.write_image(FIG / "fig2_selection_vs_making_py.png", scale=2)
    except Exception:
        pass  # kaleido optional

    ok = bool(checks["passed"].all())
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
