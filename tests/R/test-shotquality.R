# Unit tests for shot-quality-study/R/functions.R

source(file.path(REPO, "shot-quality-study", "R", "functions.R"))

test_that("action family: first keyword match wins, in definition order", {
  cases <- c(
    "Tip Dunk Shot" = "putback",
    "Putback Layup Shot" = "putback",
    "Alley Oop Dunk Shot" = "alley_oop",
    "Driving Dunk Shot" = "dunk",
    "Turnaround Bank Hook Shot" = "hook",
    "Driving Floating Bank Jump Shot" = "floater",
    "Cutting Finger Roll Layup Shot" = "layup",
    "Step Back Bank Jump Shot" = "off_dribble",
    "Running Pull-Up Jump Shot" = "off_dribble",
    "Turnaround Fadeaway shot" = "post_jumper",
    "Jump Bank Shot" = "jumper",
    "Jump Shot" = "jumper")
  expect_equal(action_family(names(cases)), unname(cases))
})

test_that("shot angle geometry", {
  expect_equal(shot_angle(0, 100), 0)
  expect_equal(shot_angle(100, 0), pi / 2)
  expect_equal(shot_angle(-100, 0), pi / 2)
  expect_gt(shot_angle(10, -10), pi / 2)
})

test_that("IRLS reaches zero logistic score", {
  set.seed(11)
  X <- cbind(1, matrix(rnorm(1200), 400, 3))
  beta_true <- c(-0.5, 1, -2, 0.25)
  y <- as.numeric(runif(400) < 1 / (1 + exp(-as.vector(X %*% beta_true))))
  beta <- logistic_irls(X, y)
  p <- predict_make(X, beta)
  expect_lt(max(abs(crossprod(X, y - p))), 1e-6)
})

test_that("decile calibration bins evenly and aggregates ties", {
  pred <- rep((0:9) / 10, each = 10)
  made <- as.numeric(seq_len(100) %% 2 == 0)
  cal <- decile_calibration(pred, made)
  expect_equal(cal$n, rep(10L, 10))
  expect_equal(cal$actual, rep(0.5, 10))
})

test_that("player table: making SE uses residual sd, not sd(points)", {
  # Regression test for the dplyr sequential-summarise masking bug the
  # R-vs-Python reconcile gate caught: `xpts = sum(xpts)` evaluated before
  # `sd(points - xpts)` silently degraded the SE to sd(points).
  df <- tibble::tibble(
    PLAYER_ID = c(1, 1, 1, 2), PLAYER_NAME = c("A", "A", "A", "B"),
    points = c(2, 0, 3, 2), xpts = c(1, 1, 1.5, 2))
  out <- player_table(df) |> dplyr::filter(PLAYER_ID == 1)
  expect_equal(out$fga, 3L)
  expect_equal(out$making_100, (5 - 3.5) / 3 * 100)
  expect_equal(out$making_100_se,
               sd(c(2, 0, 3) - c(1, 1, 1.5)) / sqrt(3) * 100)
  expect_false(isTRUE(all.equal(out$making_100_se,
                                sd(c(2, 0, 3)) / sqrt(3) * 100)))
})

test_that("clock bucket edges", {
  df <- tibble::tibble(
    MINUTES_REMAINING = rep(0, 6),
    SECONDS_REMAINING = c(4, 5, 24, 25, 36, 37),
    points = rep(2, 6), xpts = rep(1, 6))
  out <- clock_decomposition(df)
  expect_equal(setNames(out$n, out$bucket),
               c("00-04s (rush)" = 1L, "05-24s" = 2L,
                 "25-36s (2-for-1 window)" = 2L, "37s+" = 1L))
})
