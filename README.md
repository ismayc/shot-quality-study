# Shot quality: an xMake / xPTS model for every 2023-24 shot

**Question.** What should each shot have been worth — and once every shot is
priced, which players are good because of the shots they *take* (selection)
versus the shots they *make* (execution)?

This is the public-data descendant of the qSQ/qSI idea (Chang et al., Sloan
2014): team versions price shots with defender distance from tracking data;
the public feed caps out at location + shot type, and this study is explicit
about sitting at that ceiling.

<!-- terms -->
> **Terms used in this analysis.** Dotted-underlined terms anywhere below repeat these definitions on hover ([full glossary](../docs/glossary.md)).
>
> - **PPS** — Points per shot: total points on field-goal attempts divided by attempts. Free throws excluded.
> - **xPPS** — Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.
> - **xPTS** — Expected points: the modeled make probability times the shot's point value, summed over attempts.
> - **xMake** — The model's probability that a given shot goes in, estimated from location and shot type; the per-shot building block behind xPPS and xPTS.
> - **shot-making** — Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.
> - **FGA** — Field-goal attempts.
> - **eFG%** — Effective field-goal percentage: field-goal percentage with made threes counted 1.5x, putting twos and threes on one points scale.
> - **2-for-1** — Shooting early with 25-36 seconds left in a period so your team gets two possessions to the opponent's one before the buzzer.
> - **rush** — Shots in the final four seconds of a period - the desperation window where both shot selection and execution collapse.
> - **decile calibration** — Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.
> - **IRLS** — Iteratively reweighted least squares - the standard algorithm for fitting a logistic regression.
> - **empirical Bayes** — Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.
> - **shrinkage weight** — The fraction of an observed number that survives empirical-Bayes regression: 1 means fully trusted, 0 means replaced by the league mean.
> - **tau** — The estimated spread of true, noise-free skill across players (per 100 shots here); the knob that sets how hard empirical Bayes shrinks.
> - **MAE** — Mean absolute error.
> - **corner three** — A three-pointer from the corner, where the NBA line is roughly 3 ft closer than the arc - the geometry discount that makes it the cheapest three.
<!-- /terms -->

## Data

| Source | What | Size |
|---|---|---|
| [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) `shotdetail_2023` | every ShotChartDetail row of 2023-24: x/y location, zone, action type, clock, make/miss | 218,701 shots |
| `../lineup-valuation-study/data/pbp_bulk/nbastats_2023.csv` | independent play-by-play feed | cross-dataset validation only |

One bulk download (`python/01_harvest_shots.py`), no per-game API calls.

## Model

Logistic regression fit by hand-rolled <abbr title="Iteratively reweighted least squares - the standard algorithm for fitting a logistic regression.">IRLS</abbr> — identical algorithm, stopping
rule, and stabilizer in both languages, so R and Python coefficients
reconcile to ~1e-13 rather than "approximately". Features: shot-zone dummies,
linear distance splines (knots 1/3/6/10/16/22/26 ft), rim angle, heave flag,
nine deterministic action families, family-specific distance slopes for
close-range families, driving/cutting/running modifiers, and court-side
dummies. **No clock features** — deliberately, so end-of-period effects can
be read off the residuals instead of being absorbed by the model.

Per shot: `xMake` = P(make), `xPTS = xMake x shot value`. Per player:
`xPPS` (selection) and actual-minus-expected points per 100 shots
(<abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr>), with analytic SEs.

## Pipeline

```
python/01_harvest_shots.py   bulk download (idempotent)
python/02_model.py           features, IRLS fit, tables, gates, figures
R/02_model.R                 independent R/tidyverse implementation
python/03_reconcile.py       R vs Python to 1e-6 (observed ~1e-12), non-zero exit
python/04_findings.py        regenerates the Findings section below
python/05_gleague.py         optional: the same model on 94k G League shots
python/06_backtest.py        season-out backtest on 2024-25 and 2025-26 +
                             skill-persistence analysis (selection vs making)
python/07_wnba.py            the identical pipeline refit on WNBA 2024/2025
                             (archive reaches 1997); corner-economy finding
```

`01_harvest_shots.py` takes dataset names (`shotdetail_2024`,
`wnba_shotdetail_2025`, ...), so any season or league in the shufinskiy
archive is one argument away.

Validation gates (in `output/validation.csv`): attempt/make/3PM totals match
the independent play-by-play feed to 0.2%; <abbr title="Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.">decile calibration</abbr> within 0.01;
league mean <abbr title="Expected points: the modeled make probability times the shot's point value, summed over attempts.">xPTS</abbr> ties out against actual points.

**What the reconcile gate caught this time:** the first R implementation
computed the <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr> SE from `sd(points - xpts)` *after* dplyr's
sequential `summarise()` had already redefined `xpts` as its group sum —
silently turning the SE into `sd(points)`. Sums, means, and rankings all
still matched; only the two-implementation comparison surfaced it. Details
in `../docs/analysis-audit.md`.

## Findings

Every number below is generated by `python/04_findings.py` from `output/`,
not typed by hand. Both implementations (R and Python) reconcile to numeric
tolerance before this is written.

### The model

A logistic <abbr title="The model's probability that a given shot goes in, estimated from location and shot type; the per-shot building block behind xPPS and xPTS.">xMake</abbr> model over all 218,701 field-goal attempts of 2023-24 —
shot zones, distance splines, rim angle, nine action families, family-specific
distance slopes, and on-the-move modifiers; **no game-clock features**, which
is what makes the clock analysis below a finding rather than a tautology.
Calibration holds to 0.010 at the worst prediction <abbr title="Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.">decile</abbr>, and the
attempt counts agree with the independent play-by-play feed to
0.0018%.

### Selection vs execution: separating two skills the box score mixes

<abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> (expected points per shot) prices a player's **shot selection**;
actual-minus-expected points per 100 shots measures **<abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr>** above
that baseline. The two are different skills with different rosters:

| # | Player | <abbr title="Field-goal attempts.">FGA</abbr> | <abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> (selection) | <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">Making /100</abbr> | SE |
|---|---|---|---|---|---|
| 1 | Nikola Jokic | 1,411 | 1.032 | **+19.2** | ±2.8 |
| 2 | Grayson Allen | 682 | 1.121 | **+17.7** | ±5.1 |
| 3 | Stephen Curry | 1,445 | 0.991 | **+15.5** | ±3.4 |
| 4 | T.J. McConnell | 593 | 0.990 | **+15.4** | ±4.1 |
| 5 | Luka Doncic | 1,652 | 0.995 | **+15.0** | ±3.0 |
| 6 | Luke Kennard | 315 | 1.085 | **+15.0** | ±7.8 |
| 7 | Jalen Smith | 395 | 1.194 | **+14.6** | ±5.8 |
| 8 | Khris Middleton | 643 | 0.989 | **+14.4** | ±4.7 |

Best shot *selection* is a different list entirely — rim-runners:
Rudy Gobert (1.38), Trayce Jackson-Davis (1.38), Walker Kessler (1.36), Daniel Gafford (1.33), Nick Richards (1.31). That separation (a Jokić makes hard shots; a Gafford takes easy
ones) is the point of the decomposition, and neither raw FG% nor <abbr title="Effective field-goal percentage: field-goal percentage with made threes counted 1.5x, putting twos and threes on one points scale.">eFG%</abbr> can
see it.

### The end-of-period penalty, decomposed

End-of-period shots are worse — but the model says **how** they are worse:

| Seconds left | n | actual <abbr title="Points per shot: total points on field-goal attempts divided by attempts. Free throws excluded.">PPS</abbr> | <abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> | execution /100 |
|---|---|---|---|---|
| 00-04s <abbr title="Shots in the final four seconds of a period - the desperation window where both shot selection and execution collapse.">(rush)</abbr> | 4,684 | 0.712 | 0.928 | -21.6 |
| 05-24s | 4,544 | 1.033 | 1.089 | -5.5 |
| 25-36s (<abbr title="Shooting early with 25-36 seconds left in a period so your team gets two possessions to the opponent's one before the buzzer.">2-for-1</abbr> window) | 4,607 | 1.031 | 1.089 | -5.8 |
| 37s+ | 204,866 | 1.104 | 1.097 | +0.7 |

In the final four seconds of a period, shot **selection** deteriorates by
-16.9 <abbr title="Expected points: the modeled make probability times the shot's point value, summed over attempts.">xPTS</abbr>/100 (worse shots get taken) and **execution** drops
a further -22.3/100 relative to normal play (the same shots get
made less often). The old <abbr title="Effective field-goal percentage: field-goal percentage with made threes counted 1.5x, putting twos and threes on one points scale.">eFG</abbr>-only view could not separate those two effects.

### The 2-for-1 window, re-tested with a real shot-quality model

`../playbyplay-study` concluded the <abbr title="Shooting early with 25-36 seconds left in a period so your team gets two possessions to the opponent's one before the buzzer.">2-for-1</abbr> adds a possession at no
shot-quality cost — measured there with an <abbr title="Effective field-goal percentage: field-goal percentage with made threes counted 1.5x, putting twos and threes on one points scale.">eFG</abbr> proxy. This model upgrades
that claim: shots launched in the 25-36s window price at 1.089
<abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> vs 1.097 in normal play — a selection cost of only
-0.8 points per 100 shots. The proxy's conclusion survives a
model that actually prices each shot.

### G League extension: same shots, one league down

Almost nobody touches the public G League feed; the same ShotChartDetail
schema serves 94,128 G League shots for 2023-24
(`python/05_gleague.py`). Two results:

- **Shot selection is NOT the gap.** Priced by the NBA model, the G League's
  shot mix is worth 1.096 <abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> — marginally MORE than
  the NBA's own mix (1.093); the G League actually
  takes more threes (40% vs 39%) and
  more rim attempts (32% vs 30%). The
  modern shot diet has fully propagated down.
- **Execution is the whole gap**: actual G League conversion runs
  -2.9 points per 100 shots below the NBA pricing
  of the identical shots — a league-strength number that raw FG%
  comparisons can't isolate because the shot mixes differ. Transfer check:
  the NBA model's worst <abbr title="Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.">decile</abbr> gap on G League shots is 0.032, so the
  shape of shot difficulty carries over; the level shifts.

### Season-out backtest: two seasons the model never saw

The limitation the first release stated — "in-sample calibration; a
season-out backtest is the natural extension" — is now closed
(`python/06_backtest.py`). The 2023-24 model scores 2024-25 and 2025-26
without refitting:

- **Pricing transfers.** 2024-25: worst <abbr title="Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.">decile</abbr> 0.014; 2025-26: worst <abbr title="Split shots into ten bins by predicted make probability and compare predicted vs actual rates per bin; the reported number is the worst bin's gap.">decile</abbr> 0.012 (vs 0.010
  in-sample). League make rate is predicted within a fifth of a point both
  seasons. Refit coefficients barely move — the largest drifts are the
  ~400-shot backcourt/heave terms; every substantive term is stable.
- **The clock finding replicates.** The final-4s execution penalty is
  -23.3 to -20.4 per 100 across all three
  seasons fit independently; the <abbr title="Shooting early with 25-36 seconds left in a period so your team gets two possessions to the opponent's one before the buzzer.">2-for-1</abbr> window stays cheap in each.

**The result worth publishing: shot selection repeats; <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr> only
half-repeats.** Among players with 300+ <abbr title="Field-goal attempts.">FGA</abbr> in consecutive seasons:

| year-over-year r | 2023-24 → 2024-25 | 2024-25 → 2025-26 |
|---|---|---|
| <abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> (selection) | 0.91 | 0.87 |
| making /100 (execution) | 0.60 | 0.59 |
| raw <abbr title="Points per shot: total points on field-goal attempts divided by attempts. Free throws excluded.">PPS</abbr> (what the box score sees) | 0.62 | 0.59 |

(n = 215 and
207.) A player's shot **diet** is close
to a fixed trait (r ≈ 0.9); their conversion above expectation is roughly
half signal (r ≈ 0.6). Raw <abbr title="Points per shot: total points on field-goal attempts divided by attempts. Free throws excluded.">PPS</abbr> persists no better than making — the
box-score efficiency number inherits all of making's noise. Projection
implication: regress the making component hard, trust the diet.

### From finding to decision rule: an empirical-Bayes projection

The persistence table above says what to do; `python/08_projection.py` does
it and measures it. Project next season's <abbr title="Points per shot: total points on field-goal attempts divided by attempts. Free throws excluded.">PPS</abbr> as **this season's <abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr> (the
diet, r ≈ .9) plus this season's <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr> shrunk toward the league mean
by its reliability** — normal-normal <abbr title="Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.">empirical Bayes</abbr>, with each player's
analytic SE and a method-of-moments estimate of the league's true-making
spread (<abbr title="The estimated spread of true, noise-free skill across players (per 100 shots here); the knob that sets how hard empirical Bayes shrinks.">tau</abbr> ≈ 6.7-7.4 pts/100; mean <abbr title="The fraction of an observed number that survives empirical-Bayes regression: 1 means fully trusted, 0 means replaced by the league mean.">shrinkage weight</abbr>
0.71, so even a full qualifying season keeps only about
71% of its observed distance from the league mean). The two
fixed-weight corners of the same family are the baselines: weight 1 for
everyone is naive carry-forward of raw <abbr title="Points per shot: total points on field-goal attempts divided by attempts. Free throws excluded.">PPS</abbr>, weight 0 is diet-only.

| projecting | <abbr title="Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.">EB</abbr> <abbr title="Mean absolute error.">MAE</abbr> | naive carry-forward | diet-only | <abbr title="Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.">EB</abbr> vs naive |
|---|---|---|---|---|
| 2023-24 to 2024-25 | 0.0549 | 0.0620 | 0.0651 | **11%** |
| 2024-25 to 2025-26 | 0.0579 | 0.0644 | 0.0683 | **10%** |

The <abbr title="Estimate the spread of true skill across the league from the data, then pull each individual's noisy estimate toward the league mean in proportion to its noise.">EB</abbr> rule beats naive carry-forward on both held-out season pairs and
beats both fixed-weight corners — the improvement comes precisely from
regressing the noisy component and only that component. This is the
portfolio's projection loop closed: decomposition → persistence
measurement → <abbr title="The fraction of an observed number that survives empirical-Bayes regression: 1 means fully trusted, 0 means replaced by the league mean.">shrinkage</abbr> rule → out-of-sample win.

### WNBA: the identical pipeline on a league almost nobody models

`python/07_wnba.py` refits the unchanged pipeline on WNBA 2024 and 2025
(the archive reaches back to 1997), with the same play-by-play cross-check
gate. Three findings:

- **The <abbr title="A three-pointer from the corner, where the NBA line is roughly 3 ft closer than the arc - the geometry discount that makes it the cheapest three.">corner three</abbr> is a rulebook artifact, visible from orbit.** In the
  NBA the corner line is ~3 ft closer than the arc, and corners are
  26% of all threes. The WNBA's line is nearly
  uniform (corner discount ~1 ft in the shot data) — and <abbr title="A three-pointer from the corner, where the NBA line is roughly 3 ft closer than the arc - the geometry discount that makes it the cheapest three.">corner threes</abbr> are
  only 13% of threes. Where the geometry
  discount disappears, the corner economy disappears with it: shot
  selection follows the rulebook, not a style preference.
- **The WNBA is mid-3-revolution, moving fast.** Mid-range share fell from
  16.8% to 13.8% in one season
  (NBA 2024-25: 9.7%); three-point share rose from
  33.5% to 36.0%. The league is tracing
  the NBA's 2015-2020 curve at roughly the same speed. Also structural:
  44 dunks per 1,000 NBA shots vs
  0 in the WNBA — rim conversion differs
  (66% vs 63%) for reasons the action
  mix makes visible.
- **The persistence structure is league-invariant.** WNBA 2024→2025
  (n=50, 200+ <abbr title="Field-goal attempts.">FGA</abbr>): selection r = 0.80,
  making r = 0.66 — the same ordering as the NBA
  backtest. That makes "the diet is the trait, the making is half noise" a
  statement about basketball, not about one league.

Face validity, 2025 <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">shot-making</abbr>: Napheesa Collier (+17.6), Nneka Ogwumike (+16.6), Jessica Shepard (+15.3), Leonie Fiebich (+14.5). At the other pole, Angel Reese posts elite shot selection (1.14 <abbr title="Expected points per shot - the model's fair price of a shot diet: what an average shooter would score on the same attempts, judged from location and shot type alone, before knowing what went in.">xPPS</abbr>, all rim volume) with -21.0/100 making — the model quantifies exactly the debate her box score causes.

### Honest limitations

- No defender data: the public feed has no closest-defender distance, so
  "selection" here bundles openness with location. Team-tracking versions
  (qSQ/qSI) separate those; this is the public-data ceiling.
- Calibration is reported in-sample for the core model and out-of-season in
  the backtest section; the backtest is score-only (no walk-forward
  refitting scheme), which is the next rung if this became a production
  metric.
- <abbr title="Actual minus expected points per 100 shots: conversion above or below what the shot diet itself explains. Positive means making shots the model prices as hard.">Shot-making</abbr> per 100 comes with the analytic SE shown — half the league's
  qualifying players sit within ±1 SE of zero, and claiming more precision
  than that would be exactly the overreach the other studies avoid.
