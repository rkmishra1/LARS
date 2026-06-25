# LARS Simulation Study: Efron et al. (2004) Section 7 Quadratic Model
# Compares standard LARS and LARS-Lasso paths on a simulated diabetes quadratic space.

source("R/lars_impl.R")
library(lars)

cat("=========================================================\n")
cat("Starting LARS Simulation Study: Efron et al. (2004)\n")
cat("=========================================================\n\n")

# 1. Load data and construct the 64 quadratic predictors
data(diabetes)
X_orig <- as.matrix(diabetes$x)
y_orig <- as.numeric(diabetes$y)
n <- nrow(X_orig)
p <- ncol(X_orig)

# Compute main effects, interactions, and squares (except for sex column 2)
X_main <- X_orig
p_main <- ncol(X_main)

# Pairwise interactions (45 columns)
X_inter <- matrix(0, nrow = n, ncol = 0)
inter_names <- c()
for (i in 1:(p_main - 1)) {
  for (j in (i + 1):p_main) {
    X_inter <- cbind(X_inter, X_main[, i] * X_main[, j])
    inter_names <- c(inter_names, paste0(colnames(X_main)[i], ":", colnames(X_main)[j]))
  }
}
colnames(X_inter) <- inter_names

# Squares (9 columns, excluding sex which is column 2)
X_sq <- matrix(0, nrow = n, ncol = 0)
sq_names <- c()
for (i in 1:p_main) {
  if (i != 2) {
    X_sq <- cbind(X_sq, X_main[, i]^2)
    sq_names <- c(sq_names, paste0(colnames(X_main)[i], "^2"))
  }
}
colnames(X_sq) <- sq_names

# Combine into 64 predictors matrix
X_64 <- cbind(X_main, X_inter, X_sq)
colnames(X_64) <- c(colnames(X_main), colnames(X_inter), colnames(X_sq))
cat(sprintf("Constructed quadratic design matrix with %d features.\n", ncol(X_64)))

# 2. Estimate true beta by running LARS for 10 steps on original data
fit_orig_10 <- lars_fit(X_64, y_orig, type = "lar", max_steps = 10)
beta_true <- fit_orig_10$beta[11, ] # row 11 is the 10th step
mu <- fit_orig_10$intercepts[11] + X_64 %*% beta_true

# Calculate residual error variance
residuals <- y_orig - mu
sigma <- sd(residuals)
cat(sprintf("Calculated true mean vector. Residual noise sigma = %.4f\n", sigma))

# 3. Simulation loop
n_reps <- 20 # Run 20 replications for fast demo (can increase to 100 for paper precision)
cat(sprintf("\nRunning %d simulation replications...\n", n_reps))

# Track average R^2 for LARS and Lasso at each step
# Standard LARS will take exactly 64 steps, LARS-Lasso might vary. 
# We track R2 at the first 15 steps for comparison
steps_to_track <- 15
r2_lar_matrix <- matrix(NA, nrow = n_reps, ncol = steps_to_track)
r2_lasso_matrix <- matrix(NA, nrow = n_reps, ncol = steps_to_track)

set.seed(42)
for (k in 1:n_reps) {
  # Generate simulated response
  y_sim <- mu + rnorm(n, mean = 0, sd = sigma)
  
  # Fit LARS (lar)
  fit_lar <- lars_fit(X_64, y_sim, type = "lar", max_steps = steps_to_track)
  # Fit Lasso
  fit_lasso <- lars_fit(X_64, y_sim, type = "lasso", max_steps = steps_to_track)
  
  # Store R2 values (padding with final R2 if it terminated early)
  lar_r2 <- fit_lar$R2
  if (length(lar_r2) < steps_to_track) {
    lar_r2 <- c(lar_r2, rep(tail(lar_r2, 1), steps_to_track - length(lar_r2)))
  }
  r2_lar_matrix[k, ] <- lar_r2[1:steps_to_track]
  
  lasso_r2 <- fit_lasso$R2
  if (length(lasso_r2) < steps_to_track) {
    lasso_r2 <- c(lasso_r2, rep(tail(lasso_r2, 1), steps_to_track - length(lasso_r2)))
  }
  r2_lasso_matrix[k, ] <- lasso_r2[1:steps_to_track]
}

# 4. Print results
mean_r2_lar <- colMeans(r2_lar_matrix)
mean_r2_lasso <- colMeans(r2_lasso_matrix)

cat("\nAverage R^2 values explained at each step:\n")
cat("---------------------------------------------\n")
cat(sprintf("%-5s | %-12s | %-12s\n", "Step", "LARS R2", "Lasso R2"))
cat("---------------------------------------------\n")
for (s in 1:steps_to_track) {
  cat(sprintf("%-5d | %-12.4f | %-12.4f\n", s - 1, mean_r2_lar[s], mean_r2_lasso[s]))
}
cat("---------------------------------------------\n\n")
cat("Simulation study completed successfully!\n")
cat("=========================================================\n")
