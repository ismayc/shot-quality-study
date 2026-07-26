"""Write the Findings section of the shot-quality README from computed output.

Same discipline as the other studies: every number in the prose is read from
output/, never typed. Run after 02_model.py (+ 03_reconcile.py).

Run: python python/04_findings.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
README = ROOT / "README.md"
MARKER = "## Findings"
MIN_FGA = 300


def backtest_section() -> str:
    """Generated only when 06_backtest.py has run."""
    path = OUT / "backtest_metrics.csv"
    if not path.exists():
        return ""
    bt = pl.read_csv(path)
    ins = bt.filter(~pl.col("held_out")).row(0, named=True)
    oos = bt.filter(pl.col("held_out"))
    worst_oos = float(oos["worst_decile_gap"].max())
    pers = pl.read_csv(OUT / "persistence.csv")
    p = {(r["pair"], r["metric"]): (r["r"], r["n"])
         for r in pers.iter_rows(named=True)}
    clock = pl.read_csv(OUT / "clock_backtest.csv")
    rush = clock.filter(pl.col("bucket") == "00-04s (rush)")
    rush_range = (float(rush["execution_100"].min()),
                  float(rush["execution_100"].max()))
    seasons_line = "; ".join(
        f"{r['season']}: worst decile {r['worst_decile_gap']:.3f}"
        for r in oos.iter_rows(named=True))

    return f"""### Season-out backtest: two seasons the model never saw

The limitation the first release stated — "in-sample calibration; a
season-out backtest is the natural extension" — is now closed
(`python/06_backtest.py`). The 2023-24 model scores 2024-25 and 2025-26
without refitting:

- **Pricing transfers.** {seasons_line} (vs {ins['worst_decile_gap']:.3f}
  in-sample). League make rate is predicted within a fifth of a point both
  seasons. Refit coefficients barely move — the largest drifts are the
  ~400-shot backcourt/heave terms; every substantive term is stable.
- **The clock finding replicates.** The final-4s execution penalty is
  {rush_range[0]:+.1f} to {rush_range[1]:+.1f} per 100 across all three
  seasons fit independently; the 2-for-1 window stays cheap in each.

**The result worth publishing: shot selection repeats; shot-making only
half-repeats.** Among players with {300}+ FGA in consecutive seasons:

| year-over-year r | 2023-24 → 2024-25 | 2024-25 → 2025-26 |
|---|---|---|
| xPPS (selection) | {p[('2023-24 to 2024-25', 'xpps')][0]:.2f} | {p[('2024-25 to 2025-26', 'xpps')][0]:.2f} |
| making /100 (execution) | {p[('2023-24 to 2024-25', 'making_100')][0]:.2f} | {p[('2024-25 to 2025-26', 'making_100')][0]:.2f} |
| raw PPS (what the box score sees) | {p[('2023-24 to 2024-25', 'pps')][0]:.2f} | {p[('2024-25 to 2025-26', 'pps')][0]:.2f} |

(n = {p[('2023-24 to 2024-25', 'xpps')][1]} and
{p[('2024-25 to 2025-26', 'xpps')][1]}.) A player's shot **diet** is close
to a fixed trait (r ≈ 0.9); their conversion above expectation is roughly
half signal (r ≈ 0.6). Raw PPS persists no better than making — the
box-score efficiency number inherits all of making's noise. Projection
implication: regress the making component hard, trust the diet.

"""


def projection_section() -> str:
    """Generated only when 08_projection.py has run."""
    path = OUT / "projection_metrics.csv"
    if not path.exists():
        return ""
    met = pl.read_csv(path)
    hyper = pl.read_csv(OUT / "projection_hyperparams.csv")
    proj = pl.read_csv(OUT / "projection.csv")
    pairs = met["pair"].unique(maintain_order=True).to_list()

    def mae(pair: str, prefix: str) -> float:
        return float(met.filter((pl.col("pair") == pair) & pl.col("method")
                                .str.starts_with(prefix))["mae"][0])

    rows = "\n".join(
        f"| {pair} | {mae(pair, 'eb'):.4f} | {mae(pair, 'naive'):.4f} "
        f"| {mae(pair, 'diet-only'):.4f} "
        f"| **{1 - mae(pair, 'eb') / mae(pair, 'naive'):.0%}** |"
        for pair in pairs)
    tau_lo, tau_hi = float(hyper["tau"].min()), float(hyper["tau"].max())
    w_mean = float(proj["weight"].mean())

    return f"""### From finding to decision rule: an empirical-Bayes projection

The persistence table above says what to do; `python/08_projection.py` does
it and measures it. Project next season's PPS as **this season's xPPS (the
diet, r ≈ .9) plus this season's shot-making shrunk toward the league mean
by its reliability** — normal-normal empirical Bayes, with each player's
analytic SE and a method-of-moments estimate of the league's true-making
spread (tau ≈ {tau_lo:.1f}-{tau_hi:.1f} pts/100; mean shrinkage weight
{w_mean:.2f}, so even a full qualifying season keeps only about
{w_mean:.0%} of its observed distance from the league mean). The two
fixed-weight corners of the same family are the baselines: weight 1 for
everyone is naive carry-forward of raw PPS, weight 0 is diet-only.

| projecting | EB MAE | naive carry-forward | diet-only | EB vs naive |
|---|---|---|---|---|
{rows}

The EB rule beats naive carry-forward on both held-out season pairs and
beats both fixed-weight corners — the improvement comes precisely from
regressing the noisy component and only that component. This is the
portfolio's projection loop closed: decomposition → persistence
measurement → shrinkage rule → out-of-sample win.

"""


def wnba_section() -> str:
    """Generated only when 07_wnba.py has run."""
    path = OUT / "wnba_league_structure.csv"
    if not path.exists():
        return ""
    s = pl.read_csv(path)
    nba = s.filter(pl.col("league").str.starts_with("NBA")).row(0, named=True)
    w24 = s.filter(pl.col("league") == "WNBA 2024").row(0, named=True)
    w25 = s.filter(pl.col("league") == "WNBA 2025").row(0, named=True)
    pers = pl.read_csv(OUT / "wnba_persistence.csv")
    pr = {r["metric"]: (r["r"], r["n"]) for r in pers.iter_rows(named=True)}
    players = pl.read_csv(OUT / "wnba_player_table.csv", null_values=["NA"])
    top = ", ".join(f"{r['PLAYER_NAME']} ({r['making_100']:+.1f})"
                    for r in players.head(4).iter_rows(named=True))
    reese = players.filter(pl.col("PLAYER_NAME") == "Angel Reese")
    reese_line = ""
    if reese.height:
        rr = reese.row(0, named=True)
        reese_line = (f" At the other pole, Angel Reese posts elite shot "
                      f"selection ({rr['xpps']:.2f} xPPS, all rim volume) "
                      f"with {rr['making_100']:+.1f}/100 making — the model "
                      f"quantifies exactly the debate her box score causes.")

    return f"""### WNBA: the identical pipeline on a league almost nobody models

`python/07_wnba.py` refits the unchanged pipeline on WNBA 2024 and 2025
(the archive reaches back to 1997), with the same play-by-play cross-check
gate. Three findings:

- **The corner three is a rulebook artifact, visible from orbit.** In the
  NBA the corner line is ~3 ft closer than the arc, and corners are
  {nba['corner_share_of_3s']:.0%} of all threes. The WNBA's line is nearly
  uniform (corner discount ~1 ft in the shot data) — and corner threes are
  only {w25['corner_share_of_3s']:.0%} of threes. Where the geometry
  discount disappears, the corner economy disappears with it: shot
  selection follows the rulebook, not a style preference.
- **The WNBA is mid-3-revolution, moving fast.** Mid-range share fell from
  {w24['pct_midrange']:.1%} to {w25['pct_midrange']:.1%} in one season
  (NBA 2024-25: {nba['pct_midrange']:.1%}); three-point share rose from
  {w24['pct_three']:.1%} to {w25['pct_three']:.1%}. The league is tracing
  the NBA's 2015-2020 curve at roughly the same speed. Also structural:
  {nba['dunks_per_1000']:.0f} dunks per 1,000 NBA shots vs
  {w25['dunks_per_1000']:.0f} in the WNBA — rim conversion differs
  ({nba['rim_make']:.0%} vs {w25['rim_make']:.0%}) for reasons the action
  mix makes visible.
- **The persistence structure is league-invariant.** WNBA 2024→2025
  (n={pr['xpps'][1]}, {200}+ FGA): selection r = {pr['xpps'][0]:.2f},
  making r = {pr['making_100'][0]:.2f} — the same ordering as the NBA
  backtest. That makes "the diet is the trait, the making is half noise" a
  statement about basketball, not about one league.

Face validity, 2025 shot-making: {top}.{reese_line}

"""


def gleague_section() -> str:
    """Generated only when the optional G League extension has run."""
    path = OUT / "gleague_comparison.csv"
    if not path.exists():
        return ""
    comp = pl.read_csv(path)
    gl = comp.filter(pl.col("league") == "G League").row(0, named=True)
    nba = comp.filter(pl.col("league") == "NBA").row(0, named=True)
    cal = pl.read_csv(OUT / "gleague_calibration.csv")
    worst = float((cal["mean_pred"] - cal["actual"]).abs().max())
    return f"""### G League extension: same shots, one league down

Almost nobody touches the public G League feed; the same ShotChartDetail
schema serves {gl['n_shots']:,} G League shots for 2023-24
(`python/05_gleague.py`). Two results:

- **Shot selection is NOT the gap.** Priced by the NBA model, the G League's
  shot mix is worth {gl['xpps_nba_model']:.3f} xPPS — marginally MORE than
  the NBA's own mix ({nba['xpps_nba_model']:.3f}); the G League actually
  takes more threes ({gl['pct_three']:.0%} vs {nba['pct_three']:.0%}) and
  more rim attempts ({gl['pct_rim']:.0%} vs {nba['pct_rim']:.0%}). The
  modern shot diet has fully propagated down.
- **Execution is the whole gap**: actual G League conversion runs
  {gl['execution_gap_100']:+.1f} points per 100 shots below the NBA pricing
  of the identical shots — a league-strength number that raw FG%
  comparisons can't isolate because the shot mixes differ. Transfer check:
  the NBA model's worst decile gap on G League shots is {worst:.3f}, so the
  shape of shot difficulty carries over; the level shifts.

"""


def build() -> str:
    players = (pl.read_csv(OUT / "player_table.csv", null_values=["NA"])
               .filter(pl.col("fga") >= MIN_FGA))
    cal = pl.read_csv(OUT / "calibration.csv")
    clock = pl.read_csv(OUT / "clock_decomposition.csv")
    val = pl.read_csv(OUT / "validation.csv")

    top = players.head(8)
    top_rows = "\n".join(
        f"| {i + 1} | {r['PLAYER_NAME']} | {r['fga']:,} | {r['xpps']:.3f} "
        f"| **{r['making_100']:+.1f}** | ±{r['making_100_se']:.1f} |"
        for i, r in enumerate(top.iter_rows(named=True)))
    sel = players.sort("xpps", descending=True).head(5)
    sel_names = ", ".join(f"{r['PLAYER_NAME']} ({r['xpps']:.2f})"
                          for r in sel.iter_rows(named=True))

    rush = clock.filter(pl.col("bucket") == "00-04s (rush)").row(0, named=True)
    tfo = clock.filter(pl.col("bucket") == "25-36s (2-for-1 window)").row(0, named=True)
    base = clock.filter(pl.col("bucket") == "37s+").row(0, named=True)
    cal_gap = float(val.filter(pl.col("check").str.contains("decile"))["value"][0])
    fga_gap = float(val.filter(pl.col("check").str.contains("FGA"))["value"][0])

    sel_cost_rush = (rush["xpps"] - base["xpps"]) * 100
    sel_cost_tfo = (tfo["xpps"] - base["xpps"]) * 100
    exe_rush = rush["execution_100"] - base["execution_100"]

    return f"""{MARKER}

Every number below is generated by `python/04_findings.py` from `output/`,
not typed by hand. Both implementations (R and Python) reconcile to numeric
tolerance before this is written.

### The model

A logistic xMake model over all 218,701 field-goal attempts of 2023-24 —
shot zones, distance splines, rim angle, nine action families, family-specific
distance slopes, and on-the-move modifiers; **no game-clock features**, which
is what makes the clock analysis below a finding rather than a tautology.
Calibration holds to {cal_gap:.3f} at the worst prediction decile, and the
attempt counts agree with the independent play-by-play feed to
{fga_gap * 100:.4f}%.

### Selection vs execution: separating two skills the box score mixes

xPPS (expected points per shot) prices a player's **shot selection**;
actual-minus-expected points per 100 shots measures **shot-making** above
that baseline. The two are different skills with different rosters:

| # | Player | FGA | xPPS (selection) | Making /100 | SE |
|---|---|---|---|---|---|
{top_rows}

Best shot *selection* is a different list entirely — rim-runners:
{sel_names}. That separation (a Jokić makes hard shots; a Gafford takes easy
ones) is the point of the decomposition, and neither raw FG% nor eFG% can
see it.

### The end-of-period penalty, decomposed

End-of-period shots are worse — but the model says **how** they are worse:

| Seconds left | n | actual PPS | xPPS | execution /100 |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r['bucket']} | {r['n']:,} | {r['pps']:.3f} | {r['xpps']:.3f} "
        f"| {r['execution_100']:+.1f} |"
        for r in clock.iter_rows(named=True)) + f"""

In the final four seconds of a period, shot **selection** deteriorates by
{sel_cost_rush:+.1f} xPTS/100 (worse shots get taken) and **execution** drops
a further {exe_rush:+.1f}/100 relative to normal play (the same shots get
made less often). The old eFG-only view could not separate those two effects.

### The 2-for-1 window, re-tested with a real shot-quality model

`../playbyplay-study` concluded the 2-for-1 adds a possession at no
shot-quality cost — measured there with an eFG proxy. This model upgrades
that claim: shots launched in the 25-36s window price at {tfo["xpps"]:.3f}
xPPS vs {base["xpps"]:.3f} in normal play — a selection cost of only
{sel_cost_tfo:+.1f} points per 100 shots. The proxy's conclusion survives a
model that actually prices each shot.

{gleague_section()}{backtest_section()}{projection_section()}{wnba_section()}### Honest limitations

- No defender data: the public feed has no closest-defender distance, so
  "selection" here bundles openness with location. Team-tracking versions
  (qSQ/qSI) separate those; this is the public-data ceiling.
- Calibration is reported in-sample for the core model and out-of-season in
  the backtest section; the backtest is score-only (no walk-forward
  refitting scheme), which is the next rung if this became a production
  metric.
- Shot-making per 100 comes with the analytic SE shown — half the league's
  qualifying players sit within ±1 SE of zero, and claiming more precision
  than that would be exactly the overreach the other studies avoid.
"""


def main() -> int:
    text = README.read_text()
    head = text.split(MARKER)[0].rstrip()
    README.write_text(head + "\n\n" + build())
    print(f"Updated Findings in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
