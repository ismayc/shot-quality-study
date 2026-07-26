# Shot-quality model — independent R implementation (tidyverse).
#
# Same definitions as ../python/02_model.py, written separately against
# ../R/functions.R; 03_reconcile.py compares every output table numerically.
# Run: Rscript R/02_model.R    (from the study root, after 01_harvest_shots)

suppressPackageStartupMessages(library(tidyverse))

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=",
  commandArgs(trailingOnly = FALSE), value = TRUE))), ".."))
source(file.path(root, "R", "functions.R"))

shots <- readr::read_csv(file.path(root, "data", "shotdetail_2023.csv"),
                         col_types = readr::cols(.default = "c")) |>
  mutate(across(c(SHOT_DISTANCE, LOC_X, LOC_Y, PERIOD, MINUTES_REMAINING,
                  SECONDS_REMAINING, SHOT_MADE_FLAG, PLAYER_ID), as.numeric),
         value = if_else(SHOT_TYPE == "3PT Field Goal", 3, 2),
         action_family = action_family(ACTION_TYPE),
         points = SHOT_MADE_FLAG * value)
cat(sprintf("%d shots, made rate %.4f\n", nrow(shots), mean(shots$SHOT_MADE_FLAG)))

X <- design_matrix(shots)
y <- shots$SHOT_MADE_FLAG
beta <- logistic_irls(X, y)
pred <- predict_make(X, beta)
shots <- shots |> mutate(xmake = pred, xpts = pred * value)

out <- file.path(root, "output")
dir.create(out, showWarnings = FALSE)

tibble(term = colnames(X), coef = beta) |>
  readr::write_csv(file.path(out, "coefficients_r.csv"))
decile_calibration(pred, y) |>
  readr::write_csv(file.path(out, "calibration_r.csv"))
player_table(shots) |>
  readr::write_csv(file.path(out, "player_table_r.csv"))
clock_decomposition(shots) |>
  readr::write_csv(file.path(out, "clock_decomposition_r.csv"))

# ---- figures (ggplot mirrors of the plotly versions) ------------------------
fig <- file.path(root, "figures")
dir.create(fig, showWarnings = FALSE)

cal <- decile_calibration(pred, y)
ggplot(cal, aes(mean_pred, actual)) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  geom_line(colour = "#4A6FA5") + geom_point(colour = "#1E3A5F", size = 2) +
  labs(title = "Calibration: predicted vs actual make rate by decile",
       x = "mean predicted P(make)", y = "actual make rate") +
  theme_minimal()
ggsave(file.path(fig, "fig1_calibration_r.png"), width = 7, height = 5, dpi = 150)

players <- player_table(shots) |> filter(fga >= 300)
ggplot(players, aes(xpps, making_100)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_point(alpha = 0.6, colour = "#1E3A5F") +
  labs(title = "Shot selection (xPPS) vs shot-making above expected (>=300 FGA)",
       x = "expected points per shot (selection)",
       y = "points above expected per 100 shots (execution)") +
  theme_minimal()
ggsave(file.path(fig, "fig2_selection_vs_making_r.png"),
       width = 7, height = 5, dpi = 150)

cat("R outputs written\n")
