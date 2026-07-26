# Pure functions for the shot-quality model, extracted so both the pipeline
# (02_model.R) and the test suite (../../tests/R) can source them.
#
# Mirrors ../python/02_model.py definition-for-definition: same feature
# names, same column order, same IRLS with the same stabilizer and stopping
# rule, so the two implementations reconcile numerically, not approximately.

DIST_KNOTS <- c(1, 3, 6, 10, 16, 22, 26)
DIST_CAP <- 35
HEAVE_FT <- 30

# family -> keywords; FIRST match wins, so order is part of the definition.
ACTION_FAMILIES <- list(
  putback = c("Tip", "Putback"),
  alley_oop = c("Alley Oop"),
  dunk = c("Dunk"),
  hook = c("Hook"),
  floater = c("Floating"),
  layup = c("Layup"),
  off_dribble = c("Pullup", "Pull-Up", "Step Back"),
  post_jumper = c("Fadeaway", "Turnaround"),
  jumper = character(0)                       # reference level / fallback
)
ZONES <- c("Restricted Area", "In The Paint (Non-RA)", "Above the Break 3",
           "Left Corner 3", "Right Corner 3", "Backcourt")  # ref: Mid-Range
INTERACT_DIST <- c("putback", "hook", "floater", "layup")
MOVE_MODIFIERS <- c("Driving", "Cutting", "Running")
AREAS <- c("Left Side(L)", "Left Side Center(LC)", "Right Side Center(RC)",
           "Right Side(R)", "Back Court(BC)")               # ref: Center(C)

#' First-match action family for a vector of raw ACTION_TYPE strings.
action_family <- function(action_type) {
  out <- rep("jumper", length(action_type))
  undecided <- rep(TRUE, length(action_type))
  for (family in names(ACTION_FAMILIES)) {
    kws <- ACTION_FAMILIES[[family]]
    if (!length(kws)) next
    hit <- rep(FALSE, length(action_type))
    for (k in kws) hit <- hit | grepl(k, action_type, fixed = TRUE)
    out[undecided & hit] <- family
    undecided <- undecided & !hit
  }
  out
}

#' Absolute angle from the straight-on line to the rim, radians in [0, pi].
shot_angle <- function(loc_x, loc_y) atan2(abs(loc_x), loc_y)

#' Deterministic model matrix; column order is part of the contract.
design_matrix <- function(df) {
  n <- nrow(df)
  dist <- pmin(df$SHOT_DISTANCE, DIST_CAP)
  cols <- list(intercept = rep(1, n), dist = dist)
  for (k in DIST_KNOTS) cols[[sprintf("dist_gt%g", k)]] <- pmax(dist - k, 0)
  cols[["heave"]] <- as.numeric(df$SHOT_DISTANCE >= HEAVE_FT)
  cols[["angle"]] <- shot_angle(df$LOC_X, df$LOC_Y)
  for (z in ZONES) {
    nm <- paste0("zone_", gsub("[()]", "", gsub(" ", "_", tolower(z))))
    cols[[nm]] <- as.numeric(df$SHOT_ZONE_BASIC == z)
  }
  fams <- head(names(ACTION_FAMILIES), -1)     # jumper = reference
  for (f in fams) cols[[paste0("act_", f)]] <- as.numeric(df$action_family == f)
  for (f in INTERACT_DIST) {
    cols[[paste0("act_", f, "_x_dist")]] <-
      as.numeric(df$action_family == f) * dist
  }
  for (m in MOVE_MODIFIERS) {
    cols[[paste0("mod_", tolower(m))]] <-
      as.numeric(grepl(m, df$ACTION_TYPE, fixed = TRUE))
  }
  for (a in AREAS) {
    nm <- paste0("area_", gsub(" ", "_", tolower(trimws(sub("\\(.*", "", a)))))
    cols[[nm]] <- as.numeric(df$SHOT_ZONE_AREA == a)
  }
  X <- do.call(cbind, cols)
  colnames(X) <- names(cols)
  X
}

#' Newton/IRLS logistic regression, identical to the Python implementation.
logistic_irls <- function(X, y, ridge = 1e-8, tol = 1e-10, max_iter = 50) {
  beta <- rep(0, ncol(X))
  for (i in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta)
    p <- 1 / (1 + exp(-eta))
    w <- p * (1 - p)
    H <- crossprod(X, X * w) + diag(ridge, ncol(X))
    step <- solve(H, crossprod(X, y - p))
    beta <- beta + as.vector(step)
    if (max(abs(step)) < tol) break
  }
  beta
}

predict_make <- function(X, beta) 1 / (1 + exp(-as.vector(X %*% beta)))

#' Mean predicted vs actual by prediction decile. Rank-based with stable
#' ties (predictions rounded to 10 decimals first, then original row order)
#' so both languages bin identically despite ~1e-14 prediction differences.
decile_calibration <- function(pred, made) {
  n <- length(pred)
  ord <- order(round(pred, 10))               # radix sort: stable
  decile <- integer(n)
  decile[ord] <- ((seq_len(n) - 1L) * 10L) %/% n
  tibble::tibble(decile = decile + 1L, pred = pred, made = made) |>
    dplyr::group_by(decile) |>
    dplyr::summarise(n = dplyr::n(), mean_pred = mean(pred),
                     actual = mean(made), .groups = "drop") |>
    dplyr::arrange(decile)
}

#' Per-player selection (xPPS) vs execution (making per 100), analytic SE.
player_table <- function(df) {
  df |>
    dplyr::group_by(PLAYER_ID, PLAYER_NAME) |>
    # making_sd MUST precede the xpts sum: summarise() evaluates sequentially,
    # so once `xpts = sum(xpts)` runs, `xpts` is a scalar and sd(points - xpts)
    # silently becomes sd(points). The reconcile gate caught exactly this.
    dplyr::summarise(fga = dplyr::n(), pts = sum(points),
                     making_sd = sd(points - xpts), xpts = sum(xpts),
                     .groups = "drop") |>
    dplyr::mutate(pps = pts / fga, xpps = xpts / fga,
                  making_100 = (pts - xpts) / fga * 100,
                  making_100_se = making_sd / sqrt(fga) * 100) |>
    dplyr::select(-making_sd) |>
    dplyr::arrange(dplyr::desc(making_100))
}

#' Selection vs execution by seconds left in the period.
clock_decomposition <- function(df) {
  secs <- df$MINUTES_REMAINING * 60 + df$SECONDS_REMAINING
  bucket <- dplyr::case_when(
    secs <= 4 ~ "00-04s (rush)",
    secs <= 24 ~ "05-24s",
    secs <= 36 ~ "25-36s (2-for-1 window)",
    TRUE ~ "37s+")
  df |>
    dplyr::mutate(bucket = bucket) |>
    dplyr::group_by(bucket) |>
    dplyr::summarise(n = dplyr::n(), pps = mean(points), xpps = mean(xpts),
                     .groups = "drop") |>
    dplyr::mutate(execution_100 = (pps - xpps) * 100) |>
    dplyr::arrange(bucket)
}
