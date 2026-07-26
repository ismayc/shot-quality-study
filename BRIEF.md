# Shot-quality model: the same findings for three audiences

## For the front office

Every shot last season has a fair price: a dunk is worth about 1.8 points
before it leaves the hand, a contested long two about 0.8. Once you price
every shot, player skill splits cleanly in two: **taking good shots** and
**making shots at a rate the shot doesn't explain**. Those are different
skills with different names on the leaderboards: the make-above-expected list
is Jokić, Curry, Dončić, Durant; the shot-selection list is rim-runners like
Gobert and Gafford. A player's box-score efficiency mixes the two. This
separates them, which matters when you're deciding whether a shooter's cold
season is bad shots or bad shooting.

One tactical read: end-of-period heaves aside, the two-for-one window costs
almost nothing in shot quality (about a point per hundred shots), which
backs the earlier possession-count finding with an actual per-shot price.

Two follow-ups matter for player evaluation. Tested on two later seasons
the model never saw, a player's shot **diet** repeats almost perfectly year
to year (r ≈ 0.9) while conversion above expectation only half-repeats
(r ≈ 0.6). So when projecting a shooter, trust what he takes and regress
what he made. That rule is now implemented and scored: an empirical-Bayes
projection (diet carried, making shrunk toward the league by its
reliability) predicts next season's scoring efficiency about 10% better
than simply carrying last season's number forward, on both held-out
seasons. And the same holds in the WNBA, so it's a property of
basketball, not one league's quirk. The WNBA run also shows shot selection
obeys the rulebook: with no meaningful corner distance discount there, the
corner three (a quarter of NBA threes) nearly disappears.

## For analytics peers

Logistic xMake on all 218,701 FGA of 2023-24: zone dummies, linear distance
splines (knots 1-26 ft), rim angle, nine deterministic action families,
family×distance interactions for close-range families, on-the-move modifiers,
court-side dummies. Hand-rolled IRLS (ridge 1e-8 stabilizer, |Δβ|<1e-10),
implemented independently in R and Python; coefficients reconcile at ~2e-13.
Deliberately no clock features, so clock effects live in the residual:
final-4s shots lose ~17 xPTS/100 to selection and a further ~22/100 to
execution, the decomposition an eFG proxy can't do. qSQ/qSI lineage
acknowledged; no defender distance in public data, and the study says so
rather than pretending location is openness. Gates: cross-dataset totals vs
the independent pbp feed (0.002%), decile calibration ≤0.01, league xPTS
ties out. The reconcile gate earned its keep again: it caught a dplyr
sequential-summarise masking bug that silently degraded the SE column while
every sum and ranking still matched. Extensions (Python-only, reusing the
reconciled core by import): season-out backtest on 2024-25 and 2025-26
(worst OOS decile 0.014; coefficients stable; the clock decomposition
replicates), skill persistence (selection r .91/.87, making r .60/.59,
n=215/207 at 300+ FGA, same ordering in the WNBA at n=50), an
empirical-Bayes projection closing the loop (normal-normal shrinkage of
making with method-of-moments tau ≈ 6.7-7.4/100; beats naive PPS
carry-forward by 11.4%/10.0% MAE and both fixed-weight corners of its own
family on the two held-out pairs), and WNBA 2024-25 refits with a
play-by-play gate, where the near-uniform arc removes the corner discount
and corner share of threes halves vs the NBA.

## For the executive summary (three bullets)

- Priced every shot of three NBA seasons plus two WNBA seasons and a
  G League season from location and shot type; verified against independent
  data feeds, calibrated within a point at the worst decile, and
  **backtested on two seasons the model never saw**.
- Shot *selection* and shot *making* are separate, measurable skills, and
  only selection reliably repeats year over year (r ≈ 0.9 vs ≈ 0.6), in
  both the NBA and the WNBA. The empirical-Bayes projection built on that
  rule beats naive carry-forward by ~10% out of sample on two straight
  season pairs.
- Shot selection follows the rulebook: end-of-period shots are bad for two
  separable reasons, the 2-for-1 window is nearly free, and where the
  corner-three distance discount doesn't exist (WNBA), the corner economy
  vanishes.
