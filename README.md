# 📐 Least Angle Regression (LARS) in R

[![R Version](https://img.shields.io/badge/R-%3E%3D%204.0-blue.svg)](https://www.r-project.org/)
[![Build Status](https://img.shields.io/badge/Tests-Passing-success.svg)](file:///Users/ramakrushnamishra/Documents/antigravity/noble-franklin/tests/test_lars.R)
[![Shiny App](https://img.shields.io/badge/Shiny-Interactive-blueviolet.svg)](file:///Users/ramakrushnamishra/Documents/antigravity/noble-franklin/shiny_app/app.R)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An elegant, mathematically rigorous implementation of the **Least Angle Regression (LARS)** and **LARS-Lasso** algorithms in R from scratch, complete with automated validation tests, example vignettes, and an interactive step-by-step Shiny dashboard visualization.

---

## 🚀 Features

* **Pure R Implementation**: Standard LARS (Least Angle Regression) and LARS-Lasso path algorithms written from scratch using matrix formulations.
* **Identical to CRAN**: Mathematically validated to output coefficients, path histories, action sequences, and lambdas identical to CRAN's official `lars` package.
* **Interactive Shiny Dashboard**: A beautiful, custom dark-themed Shiny application allowing you to step through the algorithm frame-by-frame and visualize correlation boundaries, active coefficients, and paths.
* **Comprehensive Documentation**: Complete math explanations (LaTeX) and an R Markdown vignette.

---

## 📁 Directory Structure

```
├── README.md                 # Project introduction, math, and usage guide
├── R/
│   └── lars_impl.R           # Core implementation (lars_fit, predict, plot)
├── tests/
│   └── test_lars.R           # Verification suite comparing against CRAN's lars
├── examples/
│   ├── diabetes_demo.R       # Example script fitting LAR & Lasso on Diabetes data
│   └── LARS_vignette.Rmd     # Detailed vignette compiled with LaTeX equations
└── shiny_app/
    ├── app.R                 # Beautiful interactive Shiny visualizer app
    └── www/
        └── styles.css        # Premium custom stylesheet for the Shiny dashboard
```

---

## 🧠 Theoretical Background & Math

LARS builds a sparse linear regression model sequentially. Rather than fitting a variable completely (like Forward Selection) or taking tiny steps (like Forward Stagewise), LARS moves coefficients along an **equiangular direction** between all currently active features until another variable reaches the same absolute correlation with the residual.

### Algorithm Flowchart

```mermaid
graph TD
    A[Start: Coefficients = 0, Residual r = y] --> B[Find predictor X_j most correlated with r]
    B --> C[Add X_j to Active Set A]
    C --> D[Compute Equiangular Direction u_A]
    D --> E[Compute step size gamma_add to next join point]
    E --> F{Lasso Mode?}
    F -- Yes --> G[Compute step size gamma_drop where any active coef hits 0]
    G --> H{gamma_drop < gamma_add?}
    H -- Yes --> I[Update coefficients to gamma_drop]
    I --> J[Drop variable from Active Set A]
    J --> D
    H -- No --> K[Update coefficients to gamma_add]
    K --> L[Add new variable to Active Set A]
    L --> M{All variables active or max correlation = 0?}
    F -- No --> K
    M -- No --> D
    M -- Yes --> N[Take final step to OLS solution]
    N --> O[End Path]
```

### Mathematical Equations

#### 1. Standardization
Predictors $X$ are standardized to have zero mean and unit $L_2$ norm, and response $y$ is centered:
$$\sum_{i=1}^n X_{ij} = 0, \quad \sum_{i=1}^n X_{ij}^2 = 1, \quad \sum_{i=1}^n y_i = 0$$

#### 2. Equiangular Vector ($u_A$)
For an active set $A$ and signs $s_A = \text{sign}(X_A^T r)$, let $X_A = [s_j X_j]_{j \in A}$ and $G_A = X_A^T X_A$. The equiangular direction vector $u_A$ is:
$$u_A = X_A w_A \quad \text{where} \quad w_A = A_A G_A^{-1} 1_A, \quad A_A = (1_A^T G_A^{-1} 1_A)^{-1/2}$$
This guarantees that the projection of the direction onto all active variables is identical: $X_A^T u_A = A_A 1_A$.

#### 3. Step Length ($\hat{\gamma}$)
The step size $\hat{\gamma}$ to add the next variable $j \in A^c$ is the smallest positive value where its correlation with the residual catches up to the active set correlation $\hat{C} = \max_j |c_j|$:
$$\hat{\gamma} = \min_{j \in A^c}^+ \left\{ \frac{\hat{C} - c_j}{A_A - a_j}, \frac{\hat{C} + c_j}{A_A + a_j} \right\} \quad \text{where} \quad a = X^T u_A$$

#### 4. Lasso Coefficient Drop ($\tilde{\gamma}$)
If `type = "lasso"`, we monitor the active coefficients $\beta_A$ moving along the path $\beta_A(\gamma) = \beta_A + \gamma \tilde{w}_A$ where $\tilde{w}_A = s_A \circ w_A$. The step length $\tilde{\gamma}$ where a coefficient hits 0 is:
$$\tilde{\gamma} = \min_{j \in A}^+ \left\{ -\frac{\beta_j}{\tilde{w}_j} \right\}$$
If $\tilde{\gamma} < \hat{\gamma}$, we truncate the step at $\tilde{\gamma}$, remove that variable, and recompute the direction.

---

## 🏁 Quick Start & Usage

Ensure you have R installed. You can clone the repository and run the scripts directly.

### Running the Demo Script
To fit LARS and LARS-Lasso on the classic diabetes dataset and output summaries and plots:
```bash
Rscript examples/diabetes_demo.R
```
This generates the coefficient path plots in the `outputs/` folder:
- `outputs/lars_path.png`
- `outputs/lasso_path.png`

### Core R Code Usage

```R
# Source the custom implementation
source("R/lars_impl.R")

# Load your data
X <- as.matrix(your_predictors)
y <- as.numeric(your_response)

# Fit LARS-Lasso Path
fit <- lars_fit(X, y, type = "lasso")

# Print summary
print(fit$actions)

# Plot coefficient path
plot(fit)

# Predict at step 5.5 (interpolated)
pred <- predict(fit, newx = X, s = 5.5, type = "fit")
head(pred$fit)
```

---

## 🖥️ Interactive Shiny App Visualizer

We provide a custom dark-themed dashboard to visually explore the LARS path. It displays the solution path, Mallows' Cp metric, and the **residual correlations** highlighting how active variables are tied at the maximum correlation boundary.

> [!TIP]
> Use the **Play/Pause** playback controls in the Shiny dashboard to watch variables join and drop from the active set automatically!

### How to Run the Shiny App
Start the Shiny app from the project root directory:
```bash
Rscript -e "shiny::runApp('shiny_app')"
```
This launches a browser window showing the interactive visualization.

---

## ✅ Verification Results

Our custom implementation is thoroughly verified against CRAN's official `lars` package. You can run the test suite to confirm:
```bash
Rscript tests/test_lars.R
```

<details>
<summary><b>Click to expand verification log</b></summary>

```
==========================================
Starting LARS Implementation Verification
==========================================

Running Test 1: Lasso Path comparison...
  Max difference in lambdas: 0.00000000
  Max difference in beta coefficients: 0.00000001
  CRAN actions:  3, 9, 4, 7, 2, 10, 5, 8, 6, 1, -7, 7 
  Custom actions: 3, 9, 4, 7, 2, 10, 5, 8, 6, 1, -7, 7 
  Actions match exactly: TRUE 

Running Test 2: Standard LARS path comparison...
  Max difference in lambdas: 0.00000000
  Max difference in beta coefficients: 0.00000006
  CRAN actions:  3, 9, 4, 7, 2, 10, 5, 8, 6, 1 
  Custom actions: 3, 9, 4, 7, 2, 10, 5, 8, 6, 1 
  Actions match exactly: TRUE 

Running Test 3: Prediction interpolation test...
  Max difference in predictions at step 4.5: 0.00000000

==========================================
SUCCESS: All tests passed successfully!
==========================================
```
</details>
