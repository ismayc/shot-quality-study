# Pricing every shot, and finding out which shooting skills are real

A box score tells you what went in. It cannot tell you whether the shooter
was good or the shots were easy. That distinction is worth real money. It
is the difference between a cold shooter and a broken shot diet, between
a breakout and a hot streak. This study prices all 218,701 field-goal
attempts of 2023-24 with a logistic xMake model built from location,
distance, angle, and nine action families, with the game clock
deliberately left out. Then it follows the consequences across three NBA
seasons, the G League, and the WNBA.

Nothing downstream matters if the pricing is wrong, so the first figure is
the least glamorous and most important one.

{{fig:fig1_calibration_py|Predicted versus actual make rate by prediction decile. The license for everything else on this page: the model's prices match reality within a point at the worst decile, and the same calibration holds on two later seasons the model never saw (worst out-of-season gap 0.014). An uncalibrated pricer would make every following chart meaningless.}}

With every shot priced, player skill splits cleanly in two. **Selection**
(xPPS) is the value of the shots a player chooses; **making** is
conversion above what those shots should yield. The two skills have
different rosters, and that is the decomposition's whole point. The
make-above-expected list is Jokić, Curry, Dončić: players who convert
hard shots. The best-diet list is rim-runners like Gafford, whose value
is standing in the right place. Box-score efficiency mixes the two;
this separates them.

{{fig:fig2_selection_vs_making_py|Every qualifying player placed by shot selection (x) and shot-making (y). The figure that makes the decomposition concrete: the upper region is shot-makers, the right region is diet merchants, and the near-empty upper-right is the superstar corner. Two different skills, two different leaderboards, one chart.}}

The model's most quotable tactical result comes from what was left out of
it. Because the clock is not a feature, end-of-period effects appear in
the residuals, where they decompose: final-four-second heaves are worse
shots (−17 expected points per 100 by selection) *and* worse-executed
shots (a further −22 per 100). The 2-for-1 launch window at 25–36
seconds, meanwhile, costs almost nothing (~−0.8 per 100), upgrading the
sibling play-by-play study's eFG-proxy finding with a real per-shot
price.

Then comes the question that turns a descriptive model into a projection
tool: **which of these skills repeats?** Each season is scored with its
own refit model, so level drift cannot masquerade as skill. Shot
selection correlates at r ≈ .91/.87 year over year. Shot-making
correlates at only r ≈ .60/.59. And raw points per shot persists no better than making,
because box-score efficiency inherits all of making's noise.

{{fig:fig3_persistence_py|Shot-making in 2023-24 versus 2024-25, player by player. The study's most consequential scatter: the cloud is wide because making only half-repeats (r ≈ .60), while the equivalent selection plot is a tight diagonal (r ≈ .91). This single contrast dictates the projection rule: trust the diet, regress the making.}}

The persistence finding earns its keep by becoming a rule.
An empirical-Bayes projection carries each player's diet forward and
shrinks his making toward the league mean in proportion to its sampling
noise; its two fixed-weight corners are the natural baselines (weight 1 is
naive carry-forward, weight 0 is diet-only). On both held-out season
pairs, the EB rule beats naive carry-forward by 11.4% and 10.0% in mean
absolute error, and beats both corners, meaning the gain comes precisely
from regressing the noisy component and only that component.

{{fig:fig5_eb_projection_py|Observed shot-making versus its empirical-Bayes posterior, sized by shot volume. The shrinkage rule made visible: high-volume shooters keep their number, small samples are pulled toward the league mean. This picture is why the projection beats carrying raw numbers forward.}}

Finally, the same pipeline refit on the WNBA (with its own play-by-play
gate) shows the deepest structural result: shot selection follows the
rulebook. The WNBA's three-point line is nearly uniform (the corner
discount that defines NBA geometry barely exists), and corner threes are
12.9% of WNBA threes versus 25.5% in the NBA.

{{fig:fig4_wnba_shot_diet_py|NBA and WNBA shot diets side by side. Necessary because it shows shot selection is not a style preference: where the rulebook removes the corner-three geometry discount, the corner-three economy disappears with it. The persistence ordering (selection ≈ .80, making ≈ .66) also replicates in the WNBA: the finding is about basketball, not one league.}}

What this study cannot claim, it says out loud: the public feed has no
defender positions, so "selection" bundles openness with location. The
team-tracking versions of this model separate those, and this is the
public-data ceiling.
