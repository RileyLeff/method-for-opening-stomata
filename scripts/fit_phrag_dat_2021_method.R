#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(plantecophys)
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path("scripts", "fit_phrag_dat_2021_method.R")
project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
if (!dir.exists(file.path(project_root, "outputs"))) {
  project_root <- normalizePath(getwd(), mustWork = TRUE)
}

points_path <- file.path(project_root, "outputs", "phrag_aq_dark_rerun_points.csv")
summary_path <- file.path(project_root, "outputs", "phrag_aq_dark_rerun_dat2021_fit_summary.csv")
prediction_path <- file.path(project_root, "outputs", "phrag_aq_dark_rerun_dat2021_fit_predictions.csv")
input_path <- file.path(project_root, "outputs", "phrag_aq_dark_rerun_dat2021_fit_inputs.csv")

fit_one <- function(data, label, q_level, data_type, ci_max, method = "default") {
  fit_data <- data.frame(
    Photo = data$A,
    Ci = data$Ci,
    Tleaf = data$Tleaf,
    PARi = data$Q
  )
  fit_data <- subset(
    fit_data,
    is.finite(Photo) &
      is.finite(Ci) &
      is.finite(Tleaf) &
      is.finite(PARi) &
      Ci > 30 &
      Ci < ci_max
  )

  base_row <- data.frame(
    fit_label = label,
    Q_level = q_level,
    data_type = data_type,
    ci_max = ci_max,
    fitmethod = method,
    n_points = nrow(fit_data),
    ci_min = if (nrow(fit_data)) min(fit_data$Ci) else NA_real_,
    ci_fit_max = if (nrow(fit_data)) max(fit_data$Ci) else NA_real_,
    A_min = if (nrow(fit_data)) min(fit_data$Photo) else NA_real_,
    A_max = if (nrow(fit_data)) max(fit_data$Photo) else NA_real_,
    Vcmax25 = NA_real_,
    Vcmax25_se = NA_real_,
    Jmax25 = NA_real_,
    Jmax25_se = NA_real_,
    Rd = NA_real_,
    Rd_se = NA_real_,
    RMSE = NA_real_,
    R2 = NA_real_,
    Ci_transition = NA_real_,
    status = "not_enough_points",
    stringsAsFactors = FALSE
  )

  if (nrow(fit_data) < 8) {
    return(list(summary = base_row, predictions = data.frame()))
  }

  fit <- try(
    fitaci(fit_data, quiet = TRUE, fitmethod = method),
    silent = TRUE
  )
  if (inherits(fit, "try-error")) {
    base_row$status <- paste("fit_error:", gsub("[\r\n]+", " ", as.character(fit)))
    return(list(summary = base_row, predictions = data.frame()))
  }

  pars <- fit$pars
  obs <- fit$df$Ameas
  pred <- fit$df$Amodel
  r2 <- 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  rmse <- sqrt(mean((obs - pred)^2))

  base_row$Vcmax25 <- pars["Vcmax", "Estimate"]
  base_row$Vcmax25_se <- pars["Vcmax", "Std. Error"]
  base_row$Jmax25 <- pars["Jmax", "Estimate"]
  base_row$Jmax25_se <- pars["Jmax", "Std. Error"]
  base_row$Rd <- pars["Rd", "Estimate"]
  base_row$Rd_se <- pars["Rd", "Std. Error"]
  base_row$RMSE <- rmse
  base_row$R2 <- r2
  base_row$Ci_transition <- fit$Ci_transition
  base_row$status <- "ok"

  predictions <- data.frame(
    fit_label = label,
    Q_level = q_level,
    data_type = data_type,
    ci_max = ci_max,
    fitmethod = method,
    Ci = fit$df$Ci,
    A_observed = fit$df$Ameas,
    A_model = fit$df$Amodel,
    Ac = fit$df$Ac,
    Aj = fit$df$Aj,
    Rd = fit$df$Rd,
    Tleaf = fit$df$Tleaf,
    PPFD = fit$df$PPFD
  )

  list(summary = base_row, predictions = predictions)
}

if (!file.exists(points_path)) {
  stop("Missing ", points_path, ". Run scripts/analyze_phrag_aq_dark_rerun.py first.")
}

points <- read.csv(points_path, check.names = FALSE)
points <- subset(points, sane_for_curve == TRUE | sane_for_curve == "True")
points$Q_level <- as.numeric(points$Q_level)

q_levels <- c(200, 400, 700, 1000, 1300, 1600)

points$ci_bin_25 <- floor(points$Ci / 25) * 25
binned <- aggregate(cbind(A, Ci, Tleaf, Q, gsw) ~ Q_level + ci_bin_25, points, median, na.rm = TRUE)
bin_n <- aggregate(A ~ Q_level + ci_bin_25, points, length)
names(bin_n)[3] <- "bin_n"
binned <- merge(binned, bin_n, by = c("Q_level", "ci_bin_25"))
binned <- subset(binned, bin_n >= 2)

all_summaries <- list()
all_predictions <- list()
all_inputs <- list()

for (q in q_levels) {
  raw_q <- subset(points, Q_level == q)
  binned_q <- subset(binned, Q_level == q)

  variants <- list(
    list(data = raw_q, data_type = "raw_2hz", ci_max = 500, method = "default"),
    list(data = raw_q, data_type = "raw_2hz", ci_max = 1000, method = "default"),
    list(data = binned_q, data_type = "binned25_median", ci_max = 500, method = "default"),
    list(data = binned_q, data_type = "binned25_median", ci_max = 1000, method = "default")
  )

  for (variant in variants) {
    label <- paste0("Q", q, "_", variant$data_type, "_CiLT", variant$ci_max)
    result <- fit_one(
      variant$data,
      label = label,
      q_level = q,
      data_type = variant$data_type,
      ci_max = variant$ci_max,
      method = variant$method
    )
    all_summaries[[length(all_summaries) + 1]] <- result$summary
    if (nrow(result$predictions) > 0) {
      all_predictions[[length(all_predictions) + 1]] <- result$predictions
    }

    fit_input <- data.frame(
      fit_label = label,
      Q_level = q,
      data_type = variant$data_type,
      ci_max = variant$ci_max,
      A = variant$data$A,
      Ci = variant$data$Ci,
      Tleaf = variant$data$Tleaf,
      Q = variant$data$Q,
      gsw = variant$data$gsw
    )
    fit_input <- subset(fit_input, is.finite(A) & is.finite(Ci) & Ci > 30 & Ci < variant$ci_max)
    all_inputs[[length(all_inputs) + 1]] <- fit_input
  }
}

summary <- do.call(rbind, all_summaries)
predictions <- if (length(all_predictions)) do.call(rbind, all_predictions) else data.frame()
inputs <- if (length(all_inputs)) do.call(rbind, all_inputs) else data.frame()

write.csv(summary, summary_path, row.names = FALSE)
write.csv(predictions, prediction_path, row.names = FALSE)
write.csv(inputs, input_path, row.names = FALSE)

cat("Wrote:\n")
cat("  ", summary_path, "\n", sep = "")
cat("  ", prediction_path, "\n", sep = "")
cat("  ", input_path, "\n", sep = "")
cat("\nSuccessful fits:\n")
print(subset(summary, status == "ok")[, c("fit_label", "n_points", "Vcmax25", "Jmax25", "Rd", "RMSE", "R2", "Ci_transition")], row.names = FALSE)
