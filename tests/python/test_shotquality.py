"""Unit tests for shot-quality-study/python/02_model.py pure functions."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def test_action_family_order_matters(shotq):
    # First match wins: a Tip Dunk is a putback, an Alley Oop Dunk an alley
    # oop — the ordering is part of the model definition.
    cases = {
        "Tip Dunk Shot": "putback",
        "Putback Layup Shot": "putback",
        "Alley Oop Dunk Shot": "alley_oop",
        "Driving Dunk Shot": "dunk",
        "Turnaround Bank Hook Shot": "hook",
        "Driving Floating Bank Jump Shot": "floater",
        "Cutting Finger Roll Layup Shot": "layup",
        "Step Back Bank Jump Shot": "off_dribble",
        "Running Pull-Up Jump Shot": "off_dribble",
        "Turnaround Fadeaway shot": "post_jumper",
        "Jump Bank Shot": "jumper",
        "Jump Shot": "jumper",
    }
    for action, family in cases.items():
        assert shotq.action_family(action) == family, action


def test_shot_angle_geometry(shotq):
    # Straight on -> 0; on the baseline -> pi/2; behind the rim -> > pi/2.
    assert shotq.shot_angle(np.array([0.0]), np.array([100.0]))[0] == 0.0
    assert shotq.shot_angle(np.array([100.0]), np.array([0.0]))[0] == pytest.approx(np.pi / 2)
    assert shotq.shot_angle(np.array([-100.0]), np.array([0.0]))[0] == pytest.approx(np.pi / 2)
    assert shotq.shot_angle(np.array([10.0]), np.array([-10.0]))[0] > np.pi / 2


def test_irls_reaches_zero_score(shotq):
    # At the optimum the logistic score X'(y - p) vanishes (up to the 1e-8
    # stabilizer); this pins the solver without an external reference fit.
    rng = np.random.default_rng(11)
    X = np.column_stack([np.ones(400), rng.normal(size=(400, 3))])
    beta_true = np.array([-0.5, 1.0, -2.0, 0.25])
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(-X @ beta_true))).astype(float)
    beta = shotq.logistic_irls(X, y)
    p = shotq.predict_make(X, beta)
    assert np.max(np.abs(X.T @ (y - p))) < 1e-6


def test_decile_calibration_bins_and_ties(shotq):
    # 100 predictions in [0,1); deciles must hold 10 each, actuals must
    # aggregate within the bin — including when predictions tie.
    pred = np.repeat(np.arange(10) / 10.0, 10)
    made = (np.arange(100) % 2).astype(float)
    cal = shotq.decile_calibration(pred, made)
    assert cal["n"].to_list() == [10] * 10
    assert cal["actual"].to_list() == [0.5] * 10


def test_player_table_math(shotq):
    df = pl.DataFrame({
        "PLAYER_ID": [1, 1, 1, 2],
        "PLAYER_NAME": ["A", "A", "A", "B"],
        "points": [2, 0, 3, 2],
        "xpts": [1.0, 1.0, 1.5, 2.0],
    })
    out = shotq.player_table(df)
    a = out.filter(pl.col("PLAYER_ID") == 1).row(0, named=True)
    assert a["fga"] == 3
    assert a["pps"] == pytest.approx(5 / 3)
    assert a["xpps"] == pytest.approx(3.5 / 3)
    assert a["making_100"] == pytest.approx((5 - 3.5) / 3 * 100)
    # Regression guard for the dplyr masking bug the reconcile gate caught:
    # the SE must come from sd(points - xpts), not sd(points).
    resid_sd = np.std(np.array([2, 0, 3]) - np.array([1.0, 1.0, 1.5]), ddof=1)
    assert a["making_100_se"] == pytest.approx(resid_sd / np.sqrt(3) * 100)


def test_design_matrix_columns_are_data_independent(shotq):
    # Cross-season and cross-league scoring (06_backtest, 07_wnba) applies
    # one season's coefficients to another season's matrix, which is valid
    # only if the columns never depend on what values a dataset happens to
    # contain. Two frames with disjoint content must yield identical names.
    def frame(zone, area, action, dist):
        return pl.DataFrame({
            "SHOT_DISTANCE": [dist], "LOC_X": [10], "LOC_Y": [50],
            "SHOT_ZONE_BASIC": [zone], "SHOT_ZONE_AREA": [area],
            "ACTION_TYPE": [action],
            "action_family": [shotq.action_family(action)],
        })
    _, names_a = shotq.design_matrix(
        frame("Restricted Area", "Center(C)", "Dunk Shot", 1))
    _, names_b = shotq.design_matrix(
        frame("Left Corner 3", "Left Side(L)", "Jump Shot", 23))
    assert names_a == names_b


def test_wnba_vocabulary_maps_into_known_families(shotq):
    # The WNBA refit reuses the NBA keyword families; every family the
    # mapping can emit must be a defined family (fallback included).
    known = {f for f, _ in shotq.ACTION_FAMILIES}
    for action in ["Layup Shot", "Driving Layup Shot", "Turnaround Hook Shot",
                   "Running Pull-Up Jump Shot", "Step Back Jump shot",
                   "Alley Oop Layup shot", "Tip Layup Shot", "Jump Shot"]:
        assert shotq.action_family(action) in known


def test_clock_bucket_edges(shotq):
    df = pl.DataFrame({
        "MINUTES_REMAINING": [0, 0, 0, 0, 0, 0],
        "SECONDS_REMAINING": [4, 5, 24, 25, 36, 37],
        "points": [2] * 6,
        "xpts": [1.0] * 6,
    })
    out = shotq.clock_decomposition(df)
    by = dict(zip(out["bucket"].to_list(), out["n"].to_list()))
    assert by == {"00-04s (rush)": 1, "05-24s": 2,
                  "25-36s (2-for-1 window)": 2, "37s+": 1}


def test_eb_posterior_keeps_exact_observations(shotproj):
    # Zero sampling error -> weight 1 -> the observation survives unshrunk.
    obs = np.array([-8.0, -2.0, 3.0, 9.0])
    post, mu, tau2, w = shotproj.eb_posterior(obs, np.zeros(4))
    assert np.allclose(post, obs)
    assert np.allclose(w, 1.0)
    assert tau2 > 0


def test_eb_posterior_collapses_pure_noise_to_the_mean(shotproj):
    # If the observed spread is all sampling noise, tau2 floors at 0 and
    # everyone is regressed fully to the population mean.
    obs = np.array([-1.0, 0.0, 1.0])
    post, mu, tau2, w = shotproj.eb_posterior(obs, np.full(3, 50.0))
    assert tau2 == 0.0
    assert np.allclose(w, 0.0)
    assert np.allclose(post, mu)


def test_eb_posterior_shrinks_by_reliability(shotproj):
    # Same observation, bigger SE -> pulled harder toward the mean; the
    # posterior always lands between the observation and the mean.
    obs = np.array([30.0, 30.0, -30.0, -30.0])
    se = np.array([1.0, 20.0, 1.0, 20.0])
    post, mu, tau2, w = shotproj.eb_posterior(obs, se)
    assert w[0] > w[1] and w[2] > w[3]
    assert abs(post[1] - mu) < abs(post[0] - mu)
    for o, p in zip(obs, post):
        assert min(o, mu) - 1e-12 <= p <= max(o, mu) + 1e-12
