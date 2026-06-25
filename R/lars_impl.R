#' Least Angle Regression (LARS) and Lasso Path Implementation
#'
#' Fits the LARS or LARS-Lasso path for linear regression.
#'
#' @param X numeric matrix of predictors (n x p).
#' @param y numeric vector of response values (n).
#' @param type character string specifying the algorithm type: "lasso" or "lar" (LARS).
#' @param max_steps integer, maximum number of steps to take. Defaults to 8*p for lasso and p for lar.
#' @return An object of class "lars_fit" containing the path details.
#' @export
lars_fit <- function(X, y, type = c("lasso", "lar"), max_steps = NULL) {
  type <- match.arg(type)
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  n <- nrow(X)
  p <- ncol(X)
  
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("V", 1:p)
  }
  
  # Center and scale X, and center y
  mean_x <- colMeans(X)
  X_centered <- scale(X, center = mean_x, scale = FALSE)
  # Scale of X is the L2 norm of the centered columns
  scale_x <- sqrt(colSums(X_centered^2))
  
  # Avoid division by zero for constant features
  scale_x[scale_x == 0] <- 1
  
  X_scaled <- scale(X_centered, center = FALSE, scale = scale_x)
  
  mean_y <- mean(y)
  y_centered <- y - mean_y
  
  if (is.null(max_steps)) {
    if (type == "lar") {
      max_steps <- p
    } else {
      # Lasso can take more steps because variables can be dropped and added again
      max_steps <- 8 * p
    }
  }
  
  # Initialize path variables
  beta_path_scaled <- matrix(0, nrow = 1, ncol = p)
  colnames(beta_path_scaled) <- colnames(X)
  
  mu <- rep(0, n)
  active_set <- c()
  actions <- c()
  
  # Compute initial correlations
  c_vec <- as.vector(t(X_scaled) %*% y_centered)
  lambdas <- c(max(abs(c_vec)))
  
  step <- 0
  
  while (step < max_steps) {
    # Current residuals correlation
    c_vec <- as.vector(t(X_scaled) %*% (y_centered - mu))
    
    # If the maximum correlation is very small, we are done
    max_c <- max(abs(c_vec))
    if (max_c < 1e-12) {
      break
    }
    
    # Identify active variables (if active_set is empty, find the largest)
    if (length(active_set) == 0) {
      next_var <- which.max(abs(c_vec))
      active_set <- c(active_set, next_var)
      actions <- c(actions, next_var)
    }
    
    s_A <- sign(c_vec[active_set])
    s_A[s_A == 0] <- 1  # Handle zero sign
    
    # Form active matrix
    X_A <- X_scaled[, active_set, drop = FALSE]
    X_A <- sweep(X_A, 2, s_A, "*")
    
    # Gram matrix G_A = X_A^T X_A
    G_A <- t(X_A) %*% X_A
    # Add a tiny ridge penalty for numerical stability
    G_A <- G_A + diag(1e-12, nrow = length(active_set))
    
    inv_G_A <- solve(G_A)
    one_A <- rep(1, length(active_set))
    inv_one <- inv_G_A %*% one_A
    
    A_A <- as.numeric(1 / sqrt(sum(inv_one)))
    w_A <- as.vector(A_A * inv_one)
    
    # Equiangular direction
    u_A <- as.vector(X_A %*% w_A)
    
    # Projection vector
    a <- as.vector(t(X_scaled) %*% u_A)
    
    # Determine step size to add next variable
    gamma_add <- Inf
    next_var <- NULL
    
    if (length(active_set) < p) {
      inactive_set <- setdiff(1:p, active_set)
      
      for (j in inactive_set) {
        # Check when correlation reaches the current max correlation
        g_p <- (max_c - c_vec[j]) / (A_A - a[j])
        g_m <- (max_c + c_vec[j]) / (A_A + a[j])
        
        if (g_p > 1e-12 && g_p < gamma_add) {
          gamma_add <- g_p
          next_var <- j
        }
        if (g_m > 1e-12 && g_m < gamma_add) {
          gamma_add <- g_m
          next_var <- j
        }
      }
    } else {
      # If all variables are active, go to OLS solution
      gamma_add <- max_c / A_A
    }
    
    # Check for Lasso drops
    gamma_drop <- Inf
    drop_var <- NULL
    w_beta <- s_A * w_A
    
    if (type == "lasso" && length(active_set) > 0) {
      beta_current <- beta_path_scaled[nrow(beta_path_scaled), active_set]
      
      for (i in seq_along(active_set)) {
        j <- active_set[i]
        wb <- w_beta[i]
        if (abs(wb) > 1e-12) {
          g_d <- -beta_current[i] / wb
          if (g_d > 1e-12 && g_d < gamma_drop) {
            gamma_drop <- g_d
            drop_var <- j
          }
        }
      }
    }
    
    # Execute step
    if (gamma_drop < gamma_add) {
      # Lasso drop event
      gamma <- gamma_drop
      mu <- mu + gamma * u_A
      
      beta_next <- beta_path_scaled[nrow(beta_path_scaled), ]
      beta_next[active_set] <- beta_next[active_set] + gamma * w_beta
      beta_next[drop_var] <- 0
      
      active_set <- setdiff(active_set, drop_var)
      actions <- c(actions, -drop_var)
      
      beta_path_scaled <- rbind(beta_path_scaled, beta_next)
      lambdas <- c(lambdas, max(abs(t(X_scaled) %*% (y_centered - mu))))
    } else {
      # LARS add event
      gamma <- gamma_add
      mu <- mu + gamma * u_A
      
      beta_next <- beta_path_scaled[nrow(beta_path_scaled), ]
      beta_next[active_set] <- beta_next[active_set] + gamma * w_beta
      
      all_active_before <- (length(active_set) == p)
      
      if (length(active_set) < p) {
        active_set <- c(active_set, next_var)
        actions <- c(actions, next_var)
      }
      
      beta_path_scaled <- rbind(beta_path_scaled, beta_next)
      lambdas <- c(lambdas, max(abs(t(X_scaled) %*% (y_centered - mu))))
      
      if (all_active_before) {
        break
      }
    }
    
    step <- step + 1
    
    # Break if step size is zero or active set is empty
    if (gamma < 1e-12 || length(active_set) == 0) {
      break
    }
  }
  
  # Scale coefficients back to the original scale
  beta_path_orig <- sweep(beta_path_scaled, 2, scale_x, "/")
  
  # Calculate intercepts
  intercepts <- mean_y - as.vector(beta_path_orig %*% mean_x)
  
  # Calculate R2, RSS, and Cp (Mallows' Cp)
  if (p < n - 1) {
    ols_fit <- lm(y ~ X)
    sigma2 <- sum(residuals(ols_fit)^2) / (n - p - 1)
  } else {
    sigma2 <- sum(y_centered^2) / (n - 1)
  }
  if (sigma2 == 0) sigma2 <- 1e-12
  
  rss_vals <- c()
  r2_vals <- c()
  cp_vals <- c()
  df_vals <- c()
  
  for (i in 1:nrow(beta_path_orig)) {
    y_pred <- intercepts[i] + X %*% beta_path_orig[i, ]
    rss <- sum((y - y_pred)^2)
    tss <- sum((y - mean_y)^2)
    r2 <- if (tss == 0) 1 else (1 - rss / tss)
    
    df <- sum(beta_path_orig[i, ] != 0) + 1
    cp <- rss / sigma2 - n + 2 * df
    
    rss_vals <- c(rss_vals, rss)
    r2_vals <- c(r2_vals, r2)
    cp_vals <- c(cp_vals, cp)
    df_vals <- c(df_vals, df)
  }
  
  # Create list of actions (each element named by variable)
  action_names <- list()
  for (i in seq_along(actions)) {
    act <- actions[i]
    var_name <- colnames(X)[abs(act)]
    action_names[[i]] <- structure(act, names = var_name)
  }
  
  # Format output object
  obj <- list(
    call = match.call(),
    type = type,
    df = df_vals,
    lambda = lambdas[-length(lambdas)],
    R2 = r2_vals,
    RSS = rss_vals,
    Cp = cp_vals,
    actions = action_names,
    beta = beta_path_orig,
    intercepts = intercepts,
    mu = mu + mean_y,
    meanx = mean_x,
    normx = scale_x
  )
  class(obj) <- "lars_fit"
  return(obj)
}

#' Prediction for LARS Fits
#'
#' Obtains predictions or coefficients from a fitted LARS object.
#'
#' @param object A fitted "lars_fit" object.
#' @param newx Matrix of new predictor values.
#' @param s Step index to predict at (can be fractional).
#' @param type Character string: "coefficients" or "fit".
#' @param ... Unused.
#' @export
predict.lars_fit <- function(object, newx, s, type = c("coefficients", "fit"), ...) {
  type <- match.arg(type)
  
  step_max <- nrow(object$beta)
  
  if (missing(s)) {
    if (type == "coefficients") {
      return(list(coefficients = object$beta, intercepts = object$intercepts))
    } else {
      if (missing(newx)) {
        stop("newx must be provided for type = 'fit'")
      }
      newx <- as.matrix(newx)
      fit_matrix <- sweep(newx %*% t(object$beta), 2, object$intercepts, "+")
      return(list(fit = fit_matrix))
    }
  }
  
  # Clamp s to the range [1, step_max]
  s_clamped <- pmax(1, pmin(step_max, s))
  
  # Interpolate between steps
  s_floor <- floor(s_clamped)
  s_ceil <- ceiling(s_clamped)
  pct <- s_clamped - s_floor
  
  beta_floor <- object$beta[s_floor, , drop = FALSE]
  beta_ceil <- object$beta[s_ceil, , drop = FALSE]
  beta_interp <- beta_floor * (1 - pct) + beta_ceil * pct
  
  intercept_floor <- object$intercepts[s_floor]
  intercept_ceil <- object$intercepts[s_ceil]
  intercept_interp <- intercept_floor * (1 - pct) + intercept_ceil * pct
  
  if (type == "coefficients") {
    return(list(coefficients = beta_interp, intercepts = intercept_interp))
  } else {
    if (missing(newx)) {
      stop("newx must be provided for type = 'fit'")
    }
    newx <- as.matrix(newx)
    fit_val <- as.vector(newx %*% t(beta_interp) + intercept_interp)
    return(list(fit = fit_val))
  }
}

#' Plot LARS Coefficient Path
#'
#' Plots the coefficient path of a fitted LARS object.
#'
#' @param x A fitted "lars_fit" object.
#' @param ... Unused.
#' @export
plot.lars_fit <- function(x, ...) {
  beta <- x$beta
  l1_norm <- rowSums(abs(beta))
  
  # Color palette
  p <- ncol(beta)
  cols <- rainbow(p)
  
  # Set margins to allow text label plotting on the right side
  old_mar <- par()$mar
  on.exit(par(mar = old_mar))
  par(mar = c(5, 4, 4, 6) + 0.1)
  
  matplot(l1_norm, beta, type = "o", lty = 1, pch = 20, col = cols,
          xlab = "L1 norm of coefficients", ylab = "Coefficients",
          main = paste("LARS Path -", toupper(x$type)))
  grid(lty = "dotted")
  
  # Draw a horizontal line at 0
  abline(h = 0, lty = "dashed", col = "gray50")
  
  # Label the variables at their final values on the right margin
  for (j in 1:p) {
    text(max(l1_norm), beta[nrow(beta), j], colnames(beta)[j],
         pos = 4, col = cols[j], cex = 0.8, xpd = TRUE)
  }
}
