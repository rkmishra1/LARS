# Interactive Least Angle Regression (LARS) and Lasso Visualizer
# A beautiful, modern Shiny app demonstrating the LARS variable selection path.

library(shiny)
library(ggplot2)
library(lars)

# Source LARS implementation
if (file.exists("R/lars_impl.R")) {
  source("R/lars_impl.R")
} else if (file.exists("../R/lars_impl.R")) {
  source("../R/lars_impl.R")
} else {
  stop("Could not find R/lars_impl.R")
}

# UI Definition
ui <- fluidPage(
  # Include custom CSS
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "styles.css"),
    tags$title("LARS & Lasso Path Visualizer")
  ),
  
  # App Header
  div(
    class = "app-header",
    h1(class = "app-title", "Least Angle Regression (LARS) Path Visualizer"),
    p(class = "app-subtitle", "An interactive exploration of step-by-step coefficient paths, residual correlations, and equiangular progression.")
  ),
  
  sidebarLayout(
    # Sidebar Panel
    sidebarPanel(
      width = 4,
      div(
        class = "control-card",
        h3(class = "card-title", "Configuration"),
        
        # Data Selector
        selectInput(
          "dataset", "Select Dataset",
          choices = c("Diabetes Dataset (Classic)" = "diabetes", "Simulated Dataset" = "simulated")
        ),
        
        # Conditional inputs for simulated data
        conditionalPanel(
          condition = "input.dataset == 'simulated'",
          numericInput("sim_n", "Observations (n)", value = 150, min = 20, max = 500, step = 10),
          numericInput("sim_p", "Features (p)", value = 8, min = 3, max = 20, step = 1),
          sliderInput("sim_noise", "Noise Level (sd)", min = 0.1, max = 5, value = 1.0, step = 0.1),
          sliderInput("sim_corr", "Feature Correlation", min = 0, max = 0.9, value = 0.3, step = 0.05),
          actionButton("regen_sim", "Regenerate Simulated Data", class = "btn-playback", style = "width: 100%; margin-top: 10px; margin-bottom: 20px;")
        ),
        
        # Algorithm selection
        selectInput(
          "type", "Algorithm Type",
          choices = c("LARS-Lasso" = "lasso", "Standard LARS (LAR)" = "lar")
        ),
        
        # Slider for current step (limits are dynamically updated in server)
        uiOutput("step_slider_ui"),
        
        # Playback controls
        div(
          style = "margin-top: 15px; display: flex; justify-content: center;",
          actionButton("prev_step", "◀ Prev", class = "btn-playback"),
          actionButton("play_toggle", "Play / Pause", class = "btn-playback"),
          actionButton("next_step", "Next ▶", class = "btn-playback")
        ),
        
        # Playback speed slider
        sliderInput("play_speed", "Playback Delay (seconds)", min = 0.5, max = 3, value = 1.5, step = 0.5)
      )
    ),
    
    # Main Panel
    mainPanel(
      width = 8,
      
      # Action description card
      div(
        class = "control-card",
        h3(class = "card-title", "Algorithm Status & Event Log"),
        uiOutput("current_action_desc")
      ),
      
      # Plot container
      tabsetPanel(
        id = "plots_tabs",
        tabPanel(
          "Coefficient Path",
          div(
            class = "control-card",
            style = "margin-top: 15px;",
            h3(class = "card-title", "Solution Coefficient Path"),
            plotOutput("path_plot", height = "400px")
          )
        ),
        tabPanel(
          "Residual Correlations",
          div(
            class = "control-card",
            style = "margin-top: 15px;",
            h3(class = "card-title", "Residual Correlations at Current Step"),
            p("Active set variables are colored; they maintain identical absolute correlations (the boundary)."),
            plotOutput("corr_plot", height = "400px")
          )
        ),
        tabPanel(
          "Diagnostics",
          div(
            class = "control-card",
            style = "margin-top: 15px;",
            h3(class = "card-title", "Model Quality Metrics"),
            plotOutput("diagnostics_plot", height = "400px")
          )
        )
      ),
      
      # Active set summary table
      div(
        class = "control-card",
        h3(class = "card-title", "Active Coefficients Summary"),
        tableOutput("active_coefs_table")
      )
    )
  )
)

# Server Definition
server <- function(input, output, session) {
  
  # Reactive values to hold the data, fit, current step, and play status
  rv <- reactiveValues(
    X = NULL,
    y = NULL,
    fit = NULL,
    step = 1,
    playing = FALSE
  )
  
  # Generate or load data
  observe({
    if (input$dataset == "diabetes") {
      data(diabetes)
      rv$X <- as.matrix(diabetes$x)
      rv$y <- as.numeric(diabetes$y)
    } else {
      # Simulated data generation logic
      # Ensure reproducibility or respond to regen button
      input$regen_sim
      
      n <- isolate(input$sim_n)
      p <- isolate(input$sim_p)
      noise_sd <- isolate(input$sim_noise)
      rho <- isolate(input$sim_corr)
      
      # Generate correlated features
      set.seed(42 + as.numeric(input$regen_sim))
      Sigma <- matrix(rho, nrow = p, ncol = p)
      diag(Sigma) <- 1
      
      # Cholesky decomposition for correlation matrix
      L <- chol(Sigma)
      Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
      X_sim <- Z %*% L
      colnames(X_sim) <- paste0("X", 1:p)
      
      # Make only 3 features active to show sparse path clearly
      beta_true <- rep(0, p)
      beta_true[1] <- 3.5
      beta_true[2] <- -2.0
      beta_true[3] <- 1.5
      
      y_sim <- as.vector(X_sim %*% beta_true + rnorm(n, sd = noise_sd))
      
      rv$X <- X_sim
      rv$y <- y_sim
    }
    
    # Fit model on new data
    rv$fit <- lars_fit(rv$X, rv$y, type = input$type)
    # Reset step to 1 (Step 0)
    rv$step <- 1
  })
  
  # Re-fit when type changes
  observeEvent(input$type, {
    if (!is.null(rv$X)) {
      rv$fit <- lars_fit(rv$X, rv$y, type = input$type)
      rv$step <- 1
    }
  })
  
  # Dynamic Slider UI
  output["step_slider_ui"] <- renderUI({
    if (is.null(rv$fit)) return(NULL)
    max_s <- nrow(rv$fit$beta)
    sliderInput(
      "step_idx", "Algorithm Step",
      min = 0, max = max_s - 1, value = rv$step - 1, step = 1,
      animate = FALSE
    )
  })
  
  # Update step based on slider
  observeEvent(input$step_idx, {
    rv$step <- input$step_idx + 1
  })
  
  # Prev step button
  observeEvent(input$prev_step, {
    if (rv$step > 1) {
      rv$step <- rv$step - 1
      updateSliderInput(session, "step_idx", value = rv$step - 1)
    }
  })
  
  # Next step button
  observeEvent(input$next_step, {
    if (!is.null(rv$fit) && rv$step < nrow(rv$fit$beta)) {
      rv$step <- rv$step + 1
      updateSliderInput(session, "step_idx", value = rv$step - 1)
    }
  })
  
  # Play/Pause Timer logic
  observe({
    if (rv$playing) {
      invalidateLater(input$play_speed * 1000, session)
      isolate({
        if (rv$step < nrow(rv$fit$beta)) {
          rv$step <- rv$step + 1
          updateSliderInput(session, "step_idx", value = rv$step - 1)
        } else {
          rv$playing <- FALSE
        }
      })
    }
  })
  
  observeEvent(input$play_toggle, {
    rv$playing <- !rv$playing
  })
  
  # Stop playing if step manual slider adjustment is made
  observeEvent(input$step_idx, {
    # If manual step doesn't match current playing step, we stop play
    if (rv$playing && (input$step_idx + 1 != rv$step)) {
      rv$playing <- FALSE
    }
  })
  
  # Output current action description HTML
  output["current_action_desc"] <- renderUI({
    if (is.null(rv$fit)) return(NULL)
    s <- rv$step
    
    # Text formatting
    if (s == 1) {
      div(
        class = "action-box",
        span(class = "status-badge badge-start", "Initialization"),
        p(style = "margin-top: 10px;", 
          "Algorithm initialized. All regression coefficients are set to 0. ",
          "The intercept is set to the mean of the response variable ", 
          strong(sprintf("(%.2f).", mean(rv$y))), 
          " The maximum absolute correlation of the residual is ", 
          strong(sprintf("%.2f.", rv$fit$lambda[1]))
        )
      )
    } else {
      # Get the action leading to this step (which is index s - 1 in actions)
      act_idx <- s - 1
      if (act_idx <= length(rv$fit$actions)) {
        act <- rv$fit$actions[[act_idx]]
        val <- as.vector(act)
        var_name <- names(act)
        
        badge_class <- if (val > 0) "badge-add" else "badge-drop"
        action_word <- if (val > 0) "added to the active set" else "dropped from the active set due to coefficient crossing zero"
        
        div(
          class = "action-box",
          span(class = paste("status-badge", badge_class), if (val > 0) "Variable Added" else "Variable Dropped"),
          p(style = "margin-top: 10px;", 
            sprintf("Step %d: Feature ", act_idx), 
            strong(var_name), 
            sprintf(" was %s.", action_word),
            sprintf(" The maximum correlation value at this event boundary is "),
            strong(sprintf("%.2f.", rv$fit$lambda[s]))
          )
        )
      } else {
        # OLS solution reached
        div(
          class = "action-box",
          span(class = "status-badge badge-start", "OLS Reached"),
          p(style = "margin-top: 10px;", 
            "The algorithm has reached the full Ordinary Least Squares (OLS) solution. ",
            "All active coefficients have converged. The residual correlation is now effectively 0."
          )
        )
      }
    }
  })
  
  # Custom plot themes
  theme_dark_custom <- function() {
    theme_minimal() +
      theme(
        plot.background = element_rect(fill = "#111827", color = NA),
        panel.background = element_rect(fill = "#111827", color = NA),
        panel.grid.major = element_line(color = "#374151", size = 0.5),
        panel.grid.minor = element_line(color = "#1f2937", size = 0.25),
        text = element_text(color = "#e2e8f0", family = "Outfit"),
        axis.text = element_text(color = "#94a3b8"),
        axis.title = element_text(color = "#cbd5e1", face = "bold"),
        plot.title = element_text(color = "#ffffff", face = "bold", size = 14),
        legend.background = element_rect(fill = "#111827", color = NA),
        legend.text = element_text(color = "#e2e8f0")
      )
  }
  
  # Coefficient Path Plot Output
  output["path_plot"] <- renderPlot({
    if (is.null(rv$fit)) return(NULL)
    
    # Format coefficients into long df for ggplot
    beta <- rv$fit$beta
    l1_norm <- rowSums(abs(beta))
    steps <- 0:(nrow(beta) - 1)
    
    df_list <- list()
    for (j in colnames(beta)) {
      df_list[[j]] <- data.frame(
        step = steps,
        l1_norm = l1_norm,
        coef = beta[, j],
        variable = j
      )
    }
    plot_df <- do.call(rbind, df_list)
    
    # Current active set coefficients
    curr_l1 <- l1_norm[rv$step]
    
    g <- ggplot(plot_df, aes(x = l1_norm, y = coef, color = variable, group = variable)) +
      geom_line(size = 1, alpha = 0.8) +
      geom_point(size = 2) +
      geom_vline(xintercept = curr_l1, linetype = "dashed", color = "#c084fc", size = 1) +
      annotate("text", x = curr_l1, y = max(plot_df$coef)*0.9, label = paste("Step", rv$step - 1), 
               color = "#c084fc", angle = 90, vjust = -0.5, fontface = "bold") +
      labs(
        title = paste("LARS-Lasso Solution Path (Type:", toupper(input$type), ")"),
        x = "L1 Norm of Coefficients",
        y = "Regression Coefficient Value",
        color = "Features"
      ) +
      theme_dark_custom()
    
    # Highlight active points at current step
    curr_beta_df <- data.frame(
      l1_norm = rep(curr_l1, ncol(beta)),
      coef = beta[rv$step, ],
      variable = colnames(beta)
    )
    # Filter for non-zero or active coefficients to plot circle highlights
    curr_beta_df <- curr_beta_df[curr_beta_df$coef != 0, ]
    if (nrow(curr_beta_df) > 0) {
      g <- g + geom_point(data = curr_beta_df, aes(x = l1_norm, y = coef), 
                          color = "#ffffff", size = 4, shape = 1, stroke = 1.5)
    }
    
    return(g)
  })
  
  # Correlation Plot Output
  output["corr_plot"] <- renderPlot({
    if (is.null(rv$fit)) return(NULL)
    
    # Calculate current residuals
    s <- rv$step
    beta_s <- rv$fit$beta[s, ]
    intercept_s <- rv$fit$intercepts[s]
    y_pred <- intercept_s + rv$X %*% beta_s
    residuals <- rv$y - y_pred
    
    # Center X and scale it as in code
    mean_x <- colMeans(rv$X)
    X_centered <- scale(rv$X, center = mean_x, scale = FALSE)
    scale_x <- sqrt(colSums(X_centered^2))
    scale_x[scale_x == 0] <- 1
    X_scaled <- scale(X_centered, center = FALSE, scale = scale_x)
    
    res_centered <- residuals - mean(residuals)
    correlations <- as.vector(t(X_scaled) %*% res_centered)
    
    # Get current active set (indices of non-zero coefficients at step s)
    active_vars <- which(beta_s != 0)
    
    corr_df <- data.frame(
      variable = colnames(rv$X),
      correlation = correlations,
      abs_corr = abs(correlations),
      status = ifelse(1:ncol(rv$X) %in% active_vars, "Active", "Inactive")
    )
    
    max_c <- if (s <= length(rv$fit$lambda)) rv$fit$lambda[s] else 0.00
    
    ggplot(corr_df, aes(x = reorder(variable, abs_corr), y = correlation, fill = status)) +
      geom_bar(stat = "identity", width = 0.6, alpha = 0.85) +
      geom_hline(yintercept = c(max_c, -max_c), linetype = "dashed", color = "#f472b6", size = 1) +
      annotate("text", x = 1.5, y = max_c, label = sprintf("Boundary (+-%.2f)", max_c), 
               color = "#f472b6", vjust = -0.5, fontface = "bold", size = 3) +
      scale_fill_manual(values = c("Active" = "#8b5cf6", "Inactive" = "#4b5563")) +
      coord_flip() +
      labs(
        title = "Residual Correlations with Predictors",
        x = "Predictors",
        y = "Correlation value",
        fill = "Variable Status"
      ) +
      theme_dark_custom()
  })
  
  # Diagnostics Plot Output (R2 and Cp)
  output["diagnostics_plot"] <- renderPlot({
    if (is.null(rv$fit)) return(NULL)
    
    steps <- 0:(nrow(rv$fit$beta) - 1)
    df <- data.frame(
      step = steps,
      R2 = rv$fit$R2,
      Cp = rv$fit$Cp
    )
    
    best_step <- which.min(rv$fit$Cp) - 1
    
    # Let's draw Mallows' Cp path
    ggplot(df, aes(x = step, y = Cp)) +
      geom_line(color = "#ec4899", size = 1) +
      geom_point(color = "#ec4899", size = 2.5) +
      geom_point(data = df[df$step == best_step, ], aes(x = step, y = Cp), 
                 color = "#ffffff", size = 5, shape = 8, stroke = 1.5) +
      geom_vline(xintercept = rv$step - 1, linetype = "dashed", color = "#c084fc", size = 0.8) +
      labs(
        title = "Mallows' Cp Model Selection Metric",
        x = "Algorithm Step Number",
        y = "Mallows' Cp (lower is better)"
      ) +
      theme_dark_custom()
  })
  
  # Table of active coefficients
  output["active_coefs_table"] <- renderTable({
    if (is.null(rv$fit)) return(NULL)
    
    s <- rv$step
    beta_s <- rv$fit$beta[s, ]
    
    # Create output data frame
    non_zero <- which(beta_s != 0)
    
    if (length(non_zero) == 0) {
      return(data.frame(Status = "All coefficients are currently 0 (intercept model only)."))
    }
    
    df <- data.frame(
      Variable = names(beta_s)[non_zero],
      Coefficient = beta_s[non_zero],
      L1_Contribution = abs(beta_s[non_zero]) / sum(abs(beta_s))
    )
    
    # Round columns
    df$Coefficient <- round(df$Coefficient, 4)
    df$L1_Contribution <- sprintf("%.1f%%", df$L1_Contribution * 100)
    
    return(df)
  }, striped = TRUE, hover = TRUE, class = "table table-custom")
}

# Run the Shiny App
shinyApp(ui = ui, server = server)
