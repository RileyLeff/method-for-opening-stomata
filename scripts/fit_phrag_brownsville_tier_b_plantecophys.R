#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(plantecophys)
})

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else file.path("scripts", "fit_phrag_brownsville_tier_b_plantecophys.R")
project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
output_dir <- file.path(project_root, "outputs")

curves_path <- file.path(output_dir, "phrag_brownsville_cohort_binned_curves.csv")
qc_path <- file.path(output_dir, "phrag_brownsville_cohort_run_qc.csv")
summary_path <- file.path(output_dir, "phrag_brownsville_tier_b_plantecophys_q2000.csv")
prediction_path <- file.path(output_dir, "phrag_brownsville_tier_b_plantecophys_q2000_predictions.csv")

curves <- read.csv(curves_path, check.names = FALSE)
qc <- read.csv(qc_path, check.names = FALSE)
valid_codes <- qc$run_code[!as.logical(qc$suspect_run)]
curves <- subset(curves, run_code %in% valid_codes & Q_level == 2000)

summaries <- list()
predictions <- list()

for (code in valid_codes) {
  data <- subset(curves, run_code == code & is.finite(A) & is.finite(Ci) & Ci > 40 & Ci < 1450)
  data <- data[order(data$Ci), ]
  fit_data <- data.frame(Photo = data$A, Ci = data$Ci, Tleaf = 25, PARi = 2000)

  base <- data.frame(
    run_code = code,
    n_points = nrow(fit_data),
    Vcmax25_app_Ci = NA_real_,
    Vcmax25_se = NA_real_,
    Jmax25_app_Ci = NA_real_,
    Jmax25_se = NA_real_,
    Rd = NA_real_,
    Rd_se = NA_real_,
    Ci_transition = NA_real_,
    RMSE = NA_real_,
    R2 = NA_real_,
    fitmethod = NA_character_,
    status = "not_enough_points",
    stringsAsFactors = FALSE
  )
  if (nrow(fit_data) < 10) {
    summaries[[length(summaries) + 1]] <- base
    next
  }

  fitmethod <- "default"
  fit <- try(fitaci(fit_data, quiet = TRUE, fitmethod = fitmethod), silent = TRUE)
  if (inherits(fit, "try-error")) {
    fitmethod <- "bilinear"
    fit <- try(fitaci(fit_data, quiet = TRUE, fitmethod = fitmethod), silent = TRUE)
  }
  if (inherits(fit, "try-error")) {
    base$fitmethod <- fitmethod
    base$status <- paste("fit_error", gsub("[\r\n]+", " ", as.character(fit)))
    summaries[[length(summaries) + 1]] <- base
    next
  }

  observed <- fit$df$Ameas
  predicted <- fit$df$Amodel
  base$Vcmax25_app_Ci <- fit$pars["Vcmax", "Estimate"]
  base$Vcmax25_se <- fit$pars["Vcmax", "Std. Error"]
  base$Jmax25_app_Ci <- fit$pars["Jmax", "Estimate"]
  base$Jmax25_se <- fit$pars["Jmax", "Std. Error"]
  base$Rd <- fit$pars["Rd", "Estimate"]
  base$Rd_se <- fit$pars["Rd", "Std. Error"]
  base$Ci_transition <- fit$Ci_transition
  base$RMSE <- sqrt(mean((observed - predicted)^2))
  base$R2 <- 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)
  base$fitmethod <- fitmethod
  base$status <- "ok"
  summaries[[length(summaries) + 1]] <- base

  predictions[[length(predictions) + 1]] <- data.frame(
    run_code = code,
    Ci = fit$df$Ci,
    A_observed = observed,
    A_model = predicted,
    Ac = fit$df$Ac,
    Aj = fit$df$Aj,
    limitation = ifelse(fit$df$Ac <= fit$df$Aj, "Ac", "Aj")
  )
}

summary <- do.call(rbind, summaries)
prediction <- if (length(predictions)) do.call(rbind, predictions) else data.frame()
write.csv(summary, summary_path, row.names = FALSE)
write.csv(prediction, prediction_path, row.names = FALSE)

cat("Wrote ", summary_path, "\n", sep = "")
cat("Wrote ", prediction_path, "\n\n", sep = "")
print(summary, row.names = FALSE)
