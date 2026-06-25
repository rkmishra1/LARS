# LARS and Lasso Path Demo on Diabetes Dataset
# This script demonstrates the usage of the custom lars_fit implementation.

source("R/lars_impl.R")
library(lars)

# Create output directory if it doesn't exist
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

# Load the classic diabetes dataset
data(diabetes)
X <- as.matrix(diabetes$x)
y <- as.numeric(diabetes$y)

cat("==================================================\n")
cat("Least Angle Regression (LARS) Demo on Diabetes Data\n")
cat("==================================================\n\n")

# ----------------------------------------------------
# 1. Fit Standard LARS (Least Angle Regression)
# ----------------------------------------------------
cat("1. Fitting standard LARS path (type = 'lar')...\n")
fit_lar <- lars_fit(X, y, type = "lar")

cat("\nSummary of LARS steps:\n")
cat("----------------------------------\n")
cat(sprintf("%-5s | %-8s | %-12s | %-8s | %-6s\n", "Step", "Action", "Variable", "Lambda", "R2"))
cat("----------------------------------\n")
cat(sprintf("%-5d | %-8s | %-12s | %-8.2f | %-6.4f\n", 0, "Start", "-", fit_lar$lambda[1], fit_lar$R2[1]))
for (i in seq_along(fit_lar$actions)) {
  act <- fit_lar$actions[[i]]
  action_type <- if (act > 0) "Added" else "Dropped"
  var_name <- names(act)
  lam_val <- if (i + 1 <= length(fit_lar$lambda)) fit_lar$lambda[i + 1] else 0.00
  cat(sprintf("%-5d | %-8s | %-12s | %-8.2f | %-6.4f\n", 
              i, action_type, var_name, lam_val, fit_lar$R2[i + 1]))
}
cat("----------------------------------\n\n")

# Save LAR path plot
png("outputs/lars_path.png", width = 800, height = 600, res = 120)
plot(fit_lar)
dev.off()
cat("Saved LARS path plot to 'outputs/lars_path.png'\n\n")


# ----------------------------------------------------
# 2. Fit LARS-Lasso Path
# ----------------------------------------------------
cat("2. Fitting LARS-Lasso path (type = 'lasso')...\n")
fit_lasso <- lars_fit(X, y, type = "lasso")

cat("\nSummary of Lasso steps:\n")
cat("----------------------------------\n")
cat(sprintf("%-5s | %-8s | %-12s | %-8s | %-6s\n", "Step", "Action", "Variable", "Lambda", "R2"))
cat("----------------------------------\n")
cat(sprintf("%-5d | %-8s | %-12s | %-8.2f | %-6.4f\n", 0, "Start", "-", fit_lasso$lambda[1], fit_lasso$R2[1]))
for (i in seq_along(fit_lasso$actions)) {
  act <- fit_lasso$actions[[i]]
  action_type <- if (act > 0) "Added" else "Dropped"
  var_name <- names(act)
  lam_val <- if (i + 1 <= length(fit_lasso$lambda)) fit_lasso$lambda[i + 1] else 0.00
  cat(sprintf("%-5d | %-8s | %-12s | %-8.2f | %-6.4f\n", 
              i, action_type, var_name, lam_val, fit_lasso$R2[i + 1]))
}
cat("----------------------------------\n\n")

# Save Lasso path plot
png("outputs/lasso_path.png", width = 800, height = 600, res = 120)
plot(fit_lasso)
dev.off()
cat("Saved Lasso path plot to 'outputs/lasso_path.png'\n\n")


# ----------------------------------------------------
# 3. Model Selection based on Mallows' Cp
# ----------------------------------------------------
best_step <- which.min(fit_lasso$Cp)
cat(sprintf("3. Model selection: Best step based on Mallows' Cp is Step %d (Cp = %.2f)\n", 
            best_step - 1, fit_lasso$Cp[best_step]))

# Retrieve coefficients at best step
best_coefs <- predict(fit_lasso, s = best_step, type = "coefficients")
cat("Coefficients at best step:\n")
print(round(best_coefs$coefficients, 4))
cat(sprintf("Intercept: %.4f\n", best_coefs$intercepts))

cat("\nDemo finished successfully!\n")
cat("==================================================\n")
