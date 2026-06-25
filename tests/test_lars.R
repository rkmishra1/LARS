# Test suite for custom LARS implementation
# Compares custom lars_fit with CRAN's lars package on the diabetes dataset.

library(lars)
source("R/lars_impl.R")

cat("\n==========================================\n")
cat("Starting LARS Implementation Verification\n")
cat("==========================================\n\n")

# Load data
data(diabetes)
X <- as.matrix(diabetes$x)
y <- as.numeric(diabetes$y)

# -----------------------------------------------------------------
# Test 1: Lasso Path Comparison
# -----------------------------------------------------------------
cat("Running Test 1: Lasso Path comparison...\n")

# Fit CRAN lars (lasso)
fit_cran_lasso <- lars(X, y, type = "lasso")

# Fit custom lars (lasso)
fit_custom_lasso <- lars_fit(X, y, type = "lasso")

# Check lambdas (correlations)
lambda_diff <- max(abs(fit_cran_lasso$lambda - fit_custom_lasso$lambda))
cat(sprintf("  Max difference in lambdas: %.8f\n", lambda_diff))

# Check beta coefficients
beta_diff <- max(abs(fit_cran_lasso$beta - fit_custom_lasso$beta))
cat(sprintf("  Max difference in beta coefficients: %.8f\n", beta_diff))

# Check actions
# Convert CRAN actions to vector
cran_actions <- unlist(fit_cran_lasso$actions)
custom_actions <- unlist(fit_custom_lasso$actions)

cat("  CRAN actions: ", paste(cran_actions, collapse = ", "), "\n")
cat("  Custom actions:", paste(custom_actions, collapse = ", "), "\n")

# Compare actions lengths and values
actions_match <- length(cran_actions) == length(custom_actions) && all(cran_actions == custom_actions)
cat("  Actions match exactly:", actions_match, "\n\n")

# -----------------------------------------------------------------
# Test 2: Standard LARS (lar) Path Comparison
# -----------------------------------------------------------------
cat("Running Test 2: Standard LARS path comparison...\n")

# Fit CRAN lars (lar)
fit_cran_lar <- lars(X, y, type = "lar")

# Fit custom lars (lar)
fit_custom_lar <- lars_fit(X, y, type = "lar")

# Check lambdas
lambda_diff_lar <- max(abs(fit_cran_lar$lambda - fit_custom_lar$lambda))
cat(sprintf("  Max difference in lambdas: %.8f\n", lambda_diff_lar))

# Check beta coefficients
beta_diff_lar <- max(abs(fit_cran_lar$beta - fit_custom_lar$beta))
cat(sprintf("  Max difference in beta coefficients: %.8f\n", beta_diff_lar))

# Check actions
cran_actions_lar <- unlist(fit_cran_lar$actions)
custom_actions_lar <- unlist(fit_custom_lar$actions)

cat("  CRAN actions: ", paste(cran_actions_lar, collapse = ", "), "\n")
cat("  Custom actions:", paste(custom_actions_lar, collapse = ", "), "\n")

actions_match_lar <- length(cran_actions_lar) == length(custom_actions_lar) && all(cran_actions_lar == custom_actions_lar)
cat("  Actions match exactly:", actions_match_lar, "\n\n")

# -----------------------------------------------------------------
# Test 3: Predictions Interpolation Test
# -----------------------------------------------------------------
cat("Running Test 3: Prediction interpolation test...\n")

# Get predictions at a fractional step (e.g., step 4.5)
pred_custom <- predict(fit_custom_lasso, newx = X, s = 4.5, type = "fit")
pred_cran <- predict(fit_cran_lasso, newx = X, s = 4.5, type = "fit")

pred_diff <- max(abs(pred_cran$fit - pred_custom$fit))
cat(sprintf("  Max difference in predictions at step 4.5: %.8f\n\n", pred_diff))

# -----------------------------------------------------------------
# Final Result Summary
# -----------------------------------------------------------------
success <- (lambda_diff < 1e-4) && (beta_diff < 1e-4) && actions_match &&
           (lambda_diff_lar < 1e-4) && (beta_diff_lar < 1e-4) && actions_match_lar &&
           (pred_diff < 1e-4)

if (success) {
  cat("==========================================\n")
  cat("SUCCESS: All tests passed successfully!\n")
  cat("==========================================\n")
  quit(status = 0)
} else {
  cat("==========================================\n")
  cat("FAILURE: Discrepancies found in LARS fits!\n")
  cat("==========================================\n")
  quit(status = 1)
}
