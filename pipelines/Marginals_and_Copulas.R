load_data <- function(
    path_to_pot_df,
    path_to_threshold_df,
    center_var_col_name_pot_df,
    associated_var1_col_name_pot_df,
    associated_var2_col_name_pot_df,
    time_col_pot_df,
    time_col_threshold_df,
    threshold_col_name_threshold_df
) {
  
  library(lubridate)
  
  # -----------------------------
  # 1. Load POT data
  # -----------------------------
  df <- read.csv(path_to_pot_df)
  df <- na.omit(df)
  
  df <- df[df[[associated_var1_col_name_pot_df]] != 0, ]
  df <- df[df[[associated_var2_col_name_pot_df]] != 0, ]
  
  df <- df[df[[center_var_col_name_pot_df]] > 0, ]
  df <- df[is.finite(df[[center_var_col_name_pot_df]]), ]
  
  
  # -----------------------------
  # 2. Parse POT dates
  # -----------------------------
  x <- df[[time_col_pot_df]]
  
  dt1 <- ymd_hms(x, quiet = TRUE)
  dt2 <- ymd(x, quiet = TRUE)
  dt3 <- mdy(x, quiet = TRUE)
  dt4 <- dmy(x, quiet = TRUE)
  
  df[[time_col_pot_df]] <- as.Date(
    coalesce(dt1, dt2, dt3, dt4)
  )
  
  
  # Remove rows where date parsing failed
  df <- df[!is.na(df[[time_col_pot_df]]), ]
  
  
  # -----------------------------
  # 3. Sort by time
  # -----------------------------
  df <- df[order(df[[time_col_pot_df]]), ]
  
  
  # -----------------------------
  # 4. Elapsed time for POT data
  # -----------------------------
  time_years <- as.numeric(
    difftime(
      df[[time_col_pot_df]],
      min(df[[time_col_pot_df]]),
      units = "days"
    )
  ) / 365.25
  
  # -----------------------------
  # 5. Load threshold data
  # -----------------------------
  
  df_location <- read.csv(path_to_threshold_df)
  
  # Check required columns
  if (!(time_col_threshold_df %in% names(df_location))) {
    stop(
      paste0(
        "Column '",
        time_col_threshold_df,
        "' was not found in threshold dataframe. Available columns: ",
        paste(names(df_location), collapse = ", ")
      )
    )
  }
  
  if (!(threshold_col_name_threshold_df %in% names(df_location))) {
    stop(
      paste0(
        "Column '",
        threshold_col_name_threshold_df,
        "' was not found in threshold dataframe. Available columns: ",
        paste(names(df_location), collapse = ", ")
      )
    )
  }
  
  # Remove invalid rows
  df_location <- df_location[
    is.finite(df_location[[time_col_threshold_df]]) &
      is.finite(df_location[[threshold_col_name_threshold_df]]),
  ]
  
  # Keep years before 2025
  df_location <- df_location[
    df_location[[time_col_threshold_df]] < 2025,
  ]
  
  # Calculate elapsed years
  elapsed_years_location <-
    df_location[[time_col_threshold_df]] -
    min(df_location[[time_col_threshold_df]])
  
  # Prepare regression dataframe
  threshold_data <- data.frame(
    threshold = df_location[[threshold_col_name_threshold_df]],
    elapsed_years_location = elapsed_years_location
  )
  
  # Fit time-varying threshold
  lm_loc <- lm(
    threshold ~ elapsed_years_location,
    data = threshold_data
  )
  
  # -----------------------------
  # Return everything needed later
  # -----------------------------
  return(list(
    df = df,
    df_location = df_location,
    time_years = time_years,
    lm_loc = lm_loc,
    time_col_pot_df = time_col_pot_df
  ))
}

gpd_dist <- function(
    data,
    center_var_col_name_pot_df
) {
  
  df <- data$df
  time_years <- data$time_years
  lm_loc <- data$lm_loc
  
  
  # -----------------------------
  # 1. Time-varying threshold
  # -----------------------------
  mu_t <- predict(
    lm_loc,
    newdata = data.frame(
      elapsed_years_location = time_years
    )
  )
  
  
  # -----------------------------
  # 2. Keep exceedances
  # -----------------------------
  mask <- which(
    df[[center_var_col_name_pot_df]] > mu_t
  )
  
  x_obs <- df[[center_var_col_name_pot_df]][mask]
  mu_obs <- mu_t[mask]
  t_obs <- time_years[mask]
  
  # Excess above threshold
  y_obs <- x_obs - mu_obs
  
  
  # -----------------------------
  # 3. Negative log-likelihood
  # -----------------------------
  neg_log_likelihood <- function(params, x, t) {
    
    a <- params[1]
    b <- params[2]
    xi <- params[3]
    
    sigma <- a + b * t
    
    if (any(!is.finite(sigma)) || any(sigma <= 0)) {
      return(1e10)
    }
    
    ll <- evd::dgpd(
      x,
      loc = 0,
      scale = sigma,
      shape = xi,
      log = TRUE
    )
    
    if (any(!is.finite(ll))) {
      return(1e10)
    }
    
    return(-sum(ll))
  }
  
  
  # -----------------------------
  # 4. Fit GPD
  # -----------------------------
  init_guess <- c(
    a = 0.2,
    b = 0.001,
    xi = 0.1
  )
  
  fit <- optim(
    init_guess,
    neg_log_likelihood,
    x = y_obs,
    t = t_obs,
    method = "Nelder-Mead"
  )
  
  
  a <- fit$par[1]
  b <- fit$par[2]
  xi <- fit$par[3]
  
  
  # -----------------------------
  # 5. GPD CDF
  # -----------------------------
  gpd_cdf <- function(x, t) {
    
    mu <- predict(
      lm_loc,
      newdata = data.frame(
        elapsed_years_location = t
      )
    )
    
    sigma <- a + b * t
    
    z <- 1 + xi * (x - mu) / sigma
    
    ifelse(
      z <= 0,
      NA,
      1 - z^(-1 / xi)
    )
  }
  
  
  u_rd <- mapply(
    gpd_cdf,
    x_obs,
    t_obs
  )
  
  
  # -----------------------------
  # Return results
  # -----------------------------
  return(list(
    fit = fit,
    a = a,
    b = b,
    xi = xi,
    x_obs = x_obs,
    mu_obs = mu_obs,
    t_obs = t_obs,
    y_obs = y_obs,
    u_rd = u_rd,
    mask = mask
  ))
}

fit_marginal <- function(
    df,
    mask,
    associated_var
) {
  
  library(fitdistrplus)
  
  # ------------------------------------------------------------
  # 1. Extract values
  # ------------------------------------------------------------
  
  vals <- df[[associated_var]][mask]
  
  
  # ------------------------------------------------------------
  # 2. Shift values if zero or negative values are present
  # ------------------------------------------------------------
  
  min_val <- min(vals, na.rm = TRUE)
  
  if (min_val <= 0) {
    
    shift_val <- abs(min_val) + 1e-6
    
    vals_shifted <- vals + shift_val
    
  } else {
    
    shift_val <- 0
    vals_shifted <- vals
    
  }
  
  
  # ------------------------------------------------------------
  # 3. Fit candidate distributions
  # ------------------------------------------------------------
  
  fits <- list(
    
    gamma = tryCatch(
      fitdist(vals_shifted, "gamma"),
      error = function(e) NULL
    ),
    
    lnorm = tryCatch(
      fitdist(vals_shifted, "lnorm"),
      error = function(e) NULL
    ),
    
    weibull = tryCatch(
      fitdist(vals_shifted, "weibull"),
      error = function(e) NULL
    ),
    
    norm = tryCatch(
      fitdist(vals_shifted, "norm"),
      error = function(e) NULL
    ),
    
    exp = tryCatch(
      fitdist(vals_shifted, "exp"),
      error = function(e) NULL
    )
    
  )
  
  
  # Remove distributions that failed
  fits <- fits[!sapply(fits, is.null)]
  
  if (length(fits) == 0) {
    stop(
      paste(
        "No distribution could be fitted for",
        associated_var
      )
    )
  }
  
  
  # ------------------------------------------------------------
  # 4. Compare distributions
  # ------------------------------------------------------------
  
  gof <- gofstat(fits)
  
  best_aic <- which.min(gof$aic)
  best_bic <- which.min(gof$bic)
  
  best_fit_aic <- fits[[best_aic]]
  best_fit_bic <- fits[[best_bic]]
  
  
  best_fit_name <- best_fit_aic$distname
  params <- best_fit_aic$estimate
  
  
  cat("\nVariable:", associated_var, "\n")
  cat("Best AIC fit:", names(fits)[best_aic], "\n")
  cat("Best BIC fit:", names(fits)[best_bic], "\n")
  cat("Shift applied:", shift_val, "\n")
  
  
  # ------------------------------------------------------------
  # 5. Convert observations to uniform marginal
  # ------------------------------------------------------------
  
  if (best_fit_name == "exp") {
    
    u <- pexp(
      vals_shifted,
      rate = as.numeric(params["rate"])
    )
    
  } else if (best_fit_name == "gamma") {
    
    u <- pgamma(
      vals_shifted,
      shape = as.numeric(params["shape"]),
      rate = as.numeric(params["rate"])
    )
    
  } else if (best_fit_name == "lnorm") {
    
    u <- plnorm(
      vals_shifted,
      meanlog = as.numeric(params["meanlog"]),
      sdlog = as.numeric(params["sdlog"])
    )
    
  } else if (best_fit_name == "weibull") {
    
    u <- pweibull(
      vals_shifted,
      shape = as.numeric(params["shape"]),
      scale = as.numeric(params["scale"])
    )
    
  } else if (best_fit_name == "norm") {
    
    u <- pnorm(
      vals_shifted,
      mean = as.numeric(params["mean"]),
      sd = as.numeric(params["sd"])
    )
    
  } else {
    
    stop("Unsupported distribution for CDF transformation.")
    
  }
  
  
  # ------------------------------------------------------------
  # 6. Return results
  # ------------------------------------------------------------
  
  return(list(
    
    u = u,
    
    original_values = vals,
    shifted_values = vals_shifted,
    shift = shift_val,
    
    best_fit_name = best_fit_name,
    params = params,
    
    AIC_best = best_fit_aic,
    BIC_best = best_fit_bic,
    
    fits = fits,
    gof = gof
    
  ))
}

fit_simulate_ns_vine <- function(
    copula_df,
    n_events_year,
    N = 100,
    u_var1,
    u_var2,
    u_center,
    year_col,
    familyset = c(1, 2, 3, 4, 5),
    selectioncrit = "AIC",
    bivariate_var1 = u_var1,
    bivariate_var2 = u_var2
) {
  library(pracma)
  library(fGarch)
  
  ### center variable is the third variable in this function ###
  
  library(NSVineCopula)
  library(VineCopula)
  library(dplyr)
  
  
  # ============================================================
  # 1. Prepare copula data
  # ============================================================
  
  copula_data <- copula_df[, c(
    u_var1,
    u_var2,
    u_center
  )]
  
  
  # Remove NA / NaN
  valid_rows <- complete.cases(copula_data)
  
  # Remove Inf / -Inf
  valid_rows <- valid_rows &
    apply(
      copula_data,
      1,
      function(row) all(is.finite(row))
    )
  
  # Copula values must be strictly between 0 and 1
  valid_rows <- valid_rows &
    apply(
      copula_data,
      1,
      function(row) all(row > 0 & row < 1)
    )
  
  
  copula_data <- copula_data[valid_rows, , drop = FALSE]
  
  # Keep corresponding years
  years <- copula_df[[year_col]][valid_rows]
  
  
  # ============================================================
  # 2. Define vine structure
  # ============================================================
  
  Matrix <- c(
    1, 2, 3,
    0, 2, 3,
    0, 0, 3
  )
  
  vine_matrix <- matrix(
    Matrix,
    nrow = 3,
    ncol = 3
  )
  
  
  # ============================================================
  # 3. Fit nonstationary vine copula
  # ============================================================
  
  ns_model <- NSVineCopPar(
    data = copula_data,
    familyset = familyset,
    Matrix = vine_matrix
  )
  
  
  # ============================================================
  # 4. Simulate from fitted nonstationary vine
  # ============================================================
  
  simulated_data_array <- NSVineCopSim(
    N = N,
    RVM = ns_model,
    NSfamily = ns_model$NSfamily,
    NSpar1 = ns_model$NSParam,
    NSpar2 = ns_model$NSParam2,
    U = NULL
  )
  
  
  # Dimensions:
  # N x number_variables x number_time_steps
  
  dims <- dim(simulated_data_array)
  
  N_sim <- dims[1]
  d <- dims[2]
  Tmax <- dims[3]
  
  
  # ============================================================
  # 5. Flatten simulation array
  # ============================================================
  
  sim_all <- matrix(
    NA,
    nrow = N_sim * Tmax,
    ncol = d + 1
  )
  
  colnames(sim_all) <- c(
    colnames(copula_data),
    "Time"
  )
  
  
  for (t in seq_len(Tmax)) {
    
    rows <- ((t - 1) * N_sim + 1):(t * N_sim)
    
    sim_all[rows, 1:d] <-
      simulated_data_array[, , t]
    
    sim_all[rows, d + 1] <- t
  }
  
  
  sim_all_df <- as.data.frame(sim_all)
  
  
  # ============================================================
  # 6. Extract simulated uniforms
  # ============================================================
  
  U1 <- sim_all_df[[u_var1]]
  U2 <- sim_all_df[[u_var2]]
  UC <- sim_all_df[[u_center]]
  
  tt <- as.integer(sim_all_df$Time)
  
  
  # ============================================================
  # 7. Compute bivariate copula C12
  #
  # var1-var2 pair is stationary here, following your original
  # implementation
  # ============================================================
  
  fit12 <- BiCopSelect(
    u1 = copula_data[, 1],
    u2 = copula_data[, 2],
    familyset = NA,
    selectioncrit = selectioncrit,
    indeptest = TRUE
  )
  
  
  C12 <- BiCopCDF(
    U1,
    U2,
    family = fit12$family,
    par = fit12$par,
    par2 = fit12$par2
  )
  
  
  # ============================================================
  # 8. Compute time-varying pair-copula CDFs
  # ============================================================
  
  C13 <- numeric(length(U1))
  C23 <- numeric(length(U1))
  
  
  for (t in seq_len(Tmax)) {
    
    rows <- which(tt == t)
    
    
    # ----------------------------------------------------------
    # var1 - center
    # ----------------------------------------------------------
    
    fam_13 <- ns_model$NSfamily[3, 1]
    
    par_13 <- ns_model$NSParam[3, 1, t]
    
    par2_13 <- ns_model$NSParam2[3, 1, t]
    
    
    C13[rows] <- BiCopCDF(
      U1[rows],
      UC[rows],
      family = fam_13,
      par = par_13,
      par2 = par2_13
    )
    
    
    # ----------------------------------------------------------
    # var2 - center
    # ----------------------------------------------------------
    
    fam_23 <- ns_model$NSfamily[3, 2]
    
    par_23 <- ns_model$NSParam[3, 2, t]
    
    par2_23 <- ns_model$NSParam2[3, 2, t]
    
    
    C23[rows] <- BiCopCDF(
      U2[rows],
      UC[rows],
      family = fam_23,
      par = par_23,
      par2 = par2_23
    )
  }
  
  
  # ============================================================
  # 9. Compute trivariate vine CDF C123
  # ============================================================
  
  C123 <- numeric(length(U1))
  
  
  for (t in seq_len(Tmax)) {
    
    rows <- which(tt == t)
    
    
    RVM_t <- RVineMatrix(
      Matrix = ns_model$Matrix,
      family = ns_model$NSfamily[, ],
      par = ns_model$NSParam[, , t],
      par2 = ns_model$NSParam2[, , t],
      names = colnames(copula_data)
    )
    
    
    temp_data <- cbind(
      U1[rows],
      U2[rows],
      UC[rows]
    )
    
    
    C123[rows] <- RVineCDF(
      temp_data,
      RVM = RVM_t
    )
  }
  
  
  # ============================================================
  # 10. Collect simulation results
  # ============================================================
  
  sim_results <- data.frame(
    Time = tt
  )
  
  sim_results[[u_var1]] <- U1
  sim_results[[u_var2]] <- U2
  sim_results[[u_center]] <- UC
  
  sim_results$C12 <- C12
  sim_results$C13 <- C13
  sim_results$C23 <- C23
  sim_results$C123 <- C123
  
  
  # ============================================================
  # 11. Trivariate joint exceedance probability
  #
  # P(U1>u1, U2>u2, UC>uc)
  # ============================================================
  
  p_exceed_tri <-
    1 -
    U1 -
    U2 -
    UC +
    C12 +
    C13 +
    C23 -
    C123
  
  
  # Remove invalid / numerical-zero probabilities
  p_exceed_tri[
    !is.finite(p_exceed_tri) |
      p_exceed_tri <= 1e-12
  ] <- NA
  
  
  sim_results$ReturnPeriod_Trivariate <-
    1 / (p_exceed_tri * n_events_year)
  
  
  # ============================================================
  # 12. Bivariate joint exceedance probability
  #
  # Automatically determines which pair of variables is used
  # ============================================================
  
  # Check that requested variables are valid
  valid_vars <- c(u_var1, u_var2, u_center)
  
  if (!(bivariate_var1 %in% valid_vars) ||
      !(bivariate_var2 %in% valid_vars)) {
    
    stop(
      paste0(
        "bivariate_var1 and bivariate_var2 must be one of: ",
        paste(valid_vars, collapse = ", ")
      )
    )
  }
  
  if (bivariate_var1 == bivariate_var2) {
    stop("bivariate_var1 and bivariate_var2 must be different variables.")
  }
  
  
  # ------------------------------------------------------------
  # Determine the corresponding U values and pairwise copula CDF
  # ------------------------------------------------------------
  
  pair_key <- paste(
    sort(c(bivariate_var1, bivariate_var2)),
    collapse = "|"
  )
  
  
  key_12 <- paste(
    sort(c(u_var1, u_var2)),
    collapse = "|"
  )
  
  key_13 <- paste(
    sort(c(u_var1, u_center)),
    collapse = "|"
  )
  
  key_23 <- paste(
    sort(c(u_var2, u_center)),
    collapse = "|"
  )
  
  
  if (pair_key == key_12) {
    
    U_bi1 <- U1
    U_bi2 <- U2
    C_bi <- C12
    
  } else if (pair_key == key_13) {
    
    U_bi1 <- U1
    U_bi2 <- UC
    C_bi <- C13
    
  } else if (pair_key == key_23) {
    
    U_bi1 <- U2
    U_bi2 <- UC
    C_bi <- C23
    
  } else {
    
    stop("Could not identify the requested bivariate pair.")
  }
  
  
  # ------------------------------------------------------------
  # Joint exceedance probability
  #
  # P(Ua > ua, Ub > ub)
  # = 1 - ua - ub + C(ua, ub)
  # ------------------------------------------------------------
  
  p_exceed_bi <-
    1 -
    U_bi1 -
    U_bi2 +
    C_bi
  
  
  # Remove invalid probabilities
  p_exceed_bi[
    !is.finite(p_exceed_bi) |
      p_exceed_bi <= 1e-12
  ] <- NA
  
  
  # ------------------------------------------------------------
  # Return period
  # ------------------------------------------------------------
  
  sim_results$ReturnPeriod_Bivariate <-
    1 / (p_exceed_bi * n_events_year)
  
  sim_results$Density <- C_bi
  
  
  # ============================================================
  # 13. Attach original year to each simulated time step
  #
  # Each time step has N simulated realizations.
  # ============================================================
  
  if (length(years) != Tmax) {
    
    warning(
      paste0(
        "Number of valid years (",
        length(years),
        ") does not equal the number of nonstationary time steps (",
        Tmax,
        "). Year was not attached."
      )
    )
    
    df_simulated <- sim_results
    
  } else {
    
    sim_results$Year <- years[sim_results$Time]
    
    
    # Put Year first
    df_simulated <- sim_results[
      , c("Year", "Time", setdiff(names(sim_results), c("Year", "Time")))
    ]
  }
  
  
  # ============================================================
  # 14. Return all useful outputs
  # ============================================================
  
  return(
    list(
      
      simulated = df_simulated,
      
      model = ns_model,
      
      stationary_pair_fit = fit12,
      
      copula_data = copula_data,
      
      valid_rows = valid_rows,
      
      vine_matrix = vine_matrix
    )
  )
}

backtransform_center_gpd <- function(
    df_simulated,
    gpd_result,
    data,
    uniform_col,
    year_col = "Year",
    output_col = "Sim_Center"
) {
  
  # ------------------------------------------------------------
  # Extract fitted GPD parameters
  # ------------------------------------------------------------
  
  a <- gpd_result$a
  b <- gpd_result$b
  xi <- gpd_result$xi
  
  lm_loc <- data$lm_loc
  df_original <- data$df
  
  
  # ------------------------------------------------------------
  # Determine original starting date
  # ------------------------------------------------------------
  
  if (is.null(data$time_col_pot_df)) {
    stop(
      "time_col_pot_df needs to be stored in the output of load_data()."
    )
  }
  
  time_col <- data$time_col_pot_df
  
  start_date <- min(
    df_original[[time_col]],
    na.rm = TRUE
  )
  
  
  # ------------------------------------------------------------
  # Convert simulated Year to time in years since original start
  # ------------------------------------------------------------
  
  simulated_dates <- as.Date(
    paste0(
      df_simulated[[year_col]],
      "-01-01"
    )
  )
  
  t_sim <- as.numeric(
    difftime(
      simulated_dates,
      start_date,
      units = "days"
    )
  ) / 365.25
  
  
  # ------------------------------------------------------------
  # Time-varying threshold mu(t)
  # ------------------------------------------------------------
  
  mu_sim <- predict(
    lm_loc,
    newdata = data.frame(
      elapsed_years_location = t_sim
    )
  )
  
  
  # ------------------------------------------------------------
  # Time-varying GPD scale sigma(t)
  # ------------------------------------------------------------
  
  sigma_sim <- a + b * t_sim
  
  
  if (any(sigma_sim <= 0, na.rm = TRUE)) {
    stop(
      "Some simulated GPD scale parameters are <= 0."
    )
  }
  
  
  # ------------------------------------------------------------
  # Simulated uniform values
  # ------------------------------------------------------------
  
  u_vals <- df_simulated[[uniform_col]]
  
  # Prevent Inf from qGPD when u = 1
  eps <- 1e-10
  
  u_vals <- pmin(
    pmax(u_vals, eps),
    1 - eps
  )
  
  
  # ------------------------------------------------------------
  # Inverse nonstationary GPD
  # ------------------------------------------------------------
  
  if (abs(xi) < 1e-8) {
    
    simulated_center <-
      mu_sim -
      sigma_sim * log(1 - u_vals)
    
  } else {
    
    simulated_center <-
      mu_sim +
      (sigma_sim / xi) *
      (
        (1 - u_vals)^(-xi) - 1
      )
  }
  
  
  # ------------------------------------------------------------
  # Save useful quantities
  # ------------------------------------------------------------
  
  df_simulated$t <- t_sim
  
  df_simulated$mu_sim <- mu_sim
  
  df_simulated$sigma_sim <- sigma_sim
  
  df_simulated[[output_col]] <- simulated_center
  
  
  return(df_simulated)
}

backtransform_marginal <- function(
    df_simulated,
    marginal_result,
    uniform_col,
    output_col
) {
  
  # ------------------------------------------------------------
  # Extract fitted marginal information
  # ------------------------------------------------------------
  
  best_fit_name <- marginal_result$best_fit_name
  params <- marginal_result$params
  shift_val <- marginal_result$shift
  
  
  # ------------------------------------------------------------
  # Extract simulated uniform values
  # ------------------------------------------------------------
  
  u_vals <- df_simulated[[uniform_col]]
  
  
  # Avoid exactly 0 or 1
  eps <- 1e-10
  
  u_vals <- pmin(
    pmax(u_vals, eps),
    1 - eps
  )
  
  
  # ------------------------------------------------------------
  # Inverse CDF transformation
  # ------------------------------------------------------------
  
  if (best_fit_name == "exp") {
    
    simulated_shifted <- qexp(
      u_vals,
      rate = as.numeric(params["rate"])
    )
    
    
  } else if (best_fit_name == "gamma") {
    
    simulated_shifted <- qgamma(
      u_vals,
      shape = as.numeric(params["shape"]),
      rate = as.numeric(params["rate"])
    )
    
    
  } else if (best_fit_name == "lnorm") {
    
    simulated_shifted <- qlnorm(
      u_vals,
      meanlog = as.numeric(params["meanlog"]),
      sdlog = as.numeric(params["sdlog"])
    )
    
    
  } else if (best_fit_name == "weibull") {
    
    simulated_shifted <- qweibull(
      u_vals,
      shape = as.numeric(params["shape"]),
      scale = as.numeric(params["scale"])
    )
    
    
  } else if (best_fit_name == "norm") {
    
    simulated_shifted <- qnorm(
      u_vals,
      mean = as.numeric(params["mean"]),
      sd = as.numeric(params["sd"])
    )
    
    
  } else {
    
    stop(
      paste(
        "Unsupported distribution:",
        best_fit_name
      )
    )
  }
  
  
  # ------------------------------------------------------------
  # Remove shift to return to original physical scale
  # ------------------------------------------------------------
  
  simulated_original <- simulated_shifted - shift_val
  
  
  # ------------------------------------------------------------
  # Add result to dataframe
  # ------------------------------------------------------------
  
  df_simulated[[output_col]] <- simulated_original
  
  
  return(df_simulated)
}

library(ggplot2)
library(patchwork)
library(dplyr)


# ============================================================
# Helper function:
# Joint scatter plot + marginal histograms
# ============================================================

joint_plot_r <- function(
    obs_x,
    obs_y,
    sim_x,
    sim_y,
    xlab,
    ylab,
    bins = 35,
    obs_color = "blue",
    sim_color = "#E76F51",
    alpha_sim = 0.35,
    alpha_obs = 0.75,
    size_sim = 1.2,
    size_obs = 2.0,
    obs_label = NULL,
    sim_label = NULL
) {
  
  # ----------------------------------------------------------
  # Remove invalid values
  # ----------------------------------------------------------
  
  obs_valid <- is.finite(obs_x) & is.finite(obs_y)
  sim_valid <- is.finite(sim_x) & is.finite(sim_y)
  
  obs_x <- obs_x[obs_valid]
  obs_y <- obs_y[obs_valid]
  
  sim_x <- sim_x[sim_valid]
  sim_y <- sim_y[sim_valid]
  
  
  # ----------------------------------------------------------
  # Labels
  # ----------------------------------------------------------
  
  if (is.null(obs_label)) {
    obs_label <- paste0("Observed (n = ", length(obs_x), ")")
  }
  
  if (is.null(sim_label)) {
    sim_label <- paste0("Simulated (n = ", length(sim_x), ")")
  }
  
  
  # ----------------------------------------------------------
  # Data frames
  # ----------------------------------------------------------
  
  df_obs <- data.frame(
    x = obs_x,
    y = obs_y,
    Type = obs_label
  )
  
  df_sim <- data.frame(
    x = sim_x,
    y = sim_y,
    Type = sim_label
  )
  
  df_all <- bind_rows(df_sim, df_obs)
  
  
  # ----------------------------------------------------------
  # Common x/y limits
  # ----------------------------------------------------------
  
  x_range <- range(c(obs_x, sim_x), na.rm = TRUE)
  y_range <- range(c(obs_y, sim_y), na.rm = TRUE)
  
  x_pad <- 0.05 * diff(x_range)
  y_pad <- 0.05 * diff(y_range)
  
  if (x_pad == 0) x_pad <- 1
  if (y_pad == 0) y_pad <- 1
  
  x_limits <- c(
    x_range[1],
    x_range[2] + x_pad
  )
  
  y_limits <- c(
    y_range[1],
    y_range[2] + y_pad
  )
  
  
  # ----------------------------------------------------------
  # Scatter plot
  # ----------------------------------------------------------
  
  p_scatter <- ggplot() +
    
    geom_point(
      data = df_sim,
      aes(x = x, y = y, color = Type),
      alpha = alpha_sim,
      size = size_sim
    ) +
    
    geom_point(
      data = df_obs,
      aes(x = x, y = y, color = Type),
      alpha = alpha_obs,
      size = size_obs
    ) +
    
    scale_color_manual(
      values = setNames(
        c(sim_color, obs_color),
        c(sim_label, obs_label)
      )
    ) +
    
    coord_cartesian(
      xlim = x_limits,
      ylim = y_limits
    ) +
    
    labs(
      x = xlab,
      y = ylab,
      color = NULL
    ) +
    
    theme_classic(base_size = 11) +
    
    theme(
      axis.title = element_text(
        face = "bold",
        size = 12
      ),
      
      axis.text = element_text(
        size = 10
      ),
      
      legend.position = "bottom"
    )
  
  
  # ----------------------------------------------------------
  # Top histogram
  # ----------------------------------------------------------
  
  p_hist_x <- ggplot() +
    
    geom_histogram(
      aes(x = obs_x, y = after_stat(density)),
      bins = bins,
      fill = NA,
      color = obs_color,
      linewidth = 0.7
    ) +
    
    geom_histogram(
      aes(x = sim_x, y = after_stat(density)),
      bins = bins,
      fill = NA,
      color = sim_color,
      linewidth = 0.7
    ) +
    
    coord_cartesian(
      xlim = x_limits
    ) +
    
    theme_void() +
    
    theme(
      plot.margin = margin(
        t = 0,
        r = 0,
        b = 0,
        l = 0
      )
    )
  
  
  # ----------------------------------------------------------
  # Right histogram
  # ----------------------------------------------------------
  
  p_hist_y <- ggplot() +
    
    geom_histogram(
      aes(y = obs_y, x = after_stat(density)),
      bins = bins,
      fill = NA,
      color = obs_color,
      linewidth = 0.7
    ) +
    
    geom_histogram(
      aes(y = sim_y, x = after_stat(density)),
      bins = bins,
      fill = NA,
      color = sim_color,
      linewidth = 0.7
    ) +
    
    coord_cartesian(
      ylim = y_limits
    ) +
    
    theme_void() +
    
    theme(
      plot.margin = margin(
        t = 0,
        r = 0,
        b = 0,
        l = 0
      )
    )
  
  
  # ----------------------------------------------------------
  # Blank upper-right corner
  # ----------------------------------------------------------
  
  p_blank <- plot_spacer()
  
  
  # ----------------------------------------------------------
  # Combine
  #
  # Similar proportions to:
  # width_ratios = [4,1]
  # height_ratios = [1,4]
  # ----------------------------------------------------------
  
  joint <- (
    
    (p_hist_x | p_blank) /
      (p_scatter | p_hist_y)
    
  ) +
    
    plot_layout(
      widths = c(4, 1),
      heights = c(1, 4)
    )
  
  
  return(joint)
}

plot_copula_simulated_observed <- function(
    base_dir,
    target_events = 5,
    bins = 35,
    output_file = "Copula_Simulated_Observed_InSitu_All.png",
    width = 9,
    height = 12,
    dpi = 300
) {
  library(dplyr)
  
  # ============================================================
  # 1. RF-centered data
  # ============================================================
  
  df_rf <- read.csv(
    file.path(
      base_dir,
      paste0(
        "POT_RF_Functions.csv"
      )
    )
  )
  
  
  df_rf <- dplyr::filter(
    df_rf,
      complete.cases(df_rf),
      POT_RF > 0,
      Max_NTR != 0,
      Max_RD != 0
    )
  
  
  df_sim_rf <- read.csv(
    file.path(
      base_dir,
      "Simulated_Backtransformed_POT_RF.csv"
    )
  )
  
  
  # ============================================================
  # 2. RD-centered data
  # ============================================================
  
  df_rd <- read.csv(
    file.path(
      base_dir,
      paste0(
        "POT_RD_Functions.csv"
      )
    )
  )
  
  
  df_rd <- dplyr::filter(
    df_rd,
      complete.cases(df_rd),
      POT_RD > 0,
      Max_NTR != 0,
      Max_RF != 0
    )
  
  
  df_sim_rd <- read.csv(
    file.path(
      base_dir,
      "Simulated_Backtransformed_POT_RD.csv"
    )
  )
  
  
  # ============================================================
  # 3. NTR-centered data
  # ============================================================
  
  df_ntr <- read.csv(
    file.path(
      base_dir,
      paste0(
        "POT_NTR_Functions.csv"
      )
    )
  )
  
  
  df_ntr <- dplyr::filter(
    df_ntr,
      complete.cases(df_ntr),
      POT_NTR > 0,
      Max_RF != 0,
      Max_RD != 0
    )
  
  
  df_sim_ntr <- read.csv(
    file.path(
      base_dir,
      "Simulated_Backtransformed_POT_NTR.csv"
    )
  )
  
  
  # ============================================================
  # Sample sizes
  # ============================================================
  
  n_obs_rf <- nrow(df_rf)
  n_sim_rf <- nrow(df_sim_rf)
  
  n_obs_rd <- nrow(df_rd)
  n_sim_rd <- nrow(df_sim_rd)
  
  n_obs_ntr <- nrow(df_ntr)
  n_sim_ntr <- nrow(df_sim_ntr)
  
  
  # ============================================================
  # ROW 1 — RF-centered
  # ============================================================
  
  p1 <- joint_plot_r(
    obs_x = df_rf$POT_RF,
    obs_y = df_rf$Max_NTR,
    
    sim_x = df_sim_rf$Sim_RF,
    sim_y = df_sim_rf$Sim_NTR,
    
    xlab = "Rainfall (mm)",
    ylab = "Non-tidal Residual (m)",
    
    bins = bins,
    
    obs_label = paste0(
      "Observed (n = ",
      n_obs_rf,
      ")"
    ),
    
    sim_label = paste0(
      "Simulated (n = ",
      n_sim_rf,
      ")"
    )
  )
  
  
  p2 <- joint_plot_r(
    obs_x = df_rf$POT_RF,
    obs_y = df_rf$Max_RD,
    
    sim_x = df_sim_rf$Sim_RF,
    sim_y = df_sim_rf$Sim_RD,
    
    xlab = "Rainfall (mm)",
    ylab = expression(
      "River Discharge (m"^3*"/s)"
    ),
    
    bins = bins,
    
    obs_label = paste0(
      "Observed (n = ",
      n_obs_rf,
      ")"
    ),
    
    sim_label = paste0(
      "Simulated (n = ",
      n_sim_rf,
      ")"
    )
  )
  
  
  row1 <- (
    p1 | p2
  ) +
    
    plot_annotation(
      title = "Peak-over-threshold Rainfall",
      
      theme = theme(
        plot.title = element_text(
          size = 15,
          face = "bold",
          hjust = 0.5
        )
      )
    )
  
  
  # ============================================================
  # ROW 2 — RD-centered
  # ============================================================
  
  p3 <- joint_plot_r(
    obs_x = df_rd$POT_RD,
    obs_y = df_rd$Max_NTR,
    
    sim_x = df_sim_rd$Sim_RD,
    sim_y = df_sim_rd$Sim_NTR,
    
    xlab = expression(
      "River Discharge (m"^3*"/s)"
    ),
    
    ylab = "Non-tidal Residual (m)",
    
    bins = bins,
    
    obs_label = paste0(
      "Observed (n = ",
      n_obs_rd,
      ")"
    ),
    
    sim_label = paste0(
      "Simulated (n = ",
      n_sim_rd,
      ")"
    )
  )
  
  
  p4 <- joint_plot_r(
    obs_x = df_rd$POT_RD,
    obs_y = df_rd$Max_RF,
    
    sim_x = df_sim_rd$Sim_RD,
    sim_y = df_sim_rd$Sim_RF,
    
    xlab = expression(
      "River Discharge (m"^3*"/s)"
    ),
    
    ylab = "Rainfall (mm)",
    
    bins = bins,
    
    obs_label = paste0(
      "Observed (n = ",
      n_obs_rd,
      ")"
    ),
    
    sim_label = paste0(
      "Simulated (n = ",
      n_sim_rd,
      ")"
    )
  )
  
  
  row2 <- (
    p3 | p4
  ) +
    
    plot_annotation(
      title = "Peak-over-threshold River Discharge",
      
      theme = theme(
        plot.title = element_text(
          size = 15,
          face = "bold",
          hjust = 0.5
        )
      )
    )
  
  
  # ============================================================
  # ROW 3 — NTR-centered
  # ============================================================
  
  p5 <- joint_plot_r(
    obs_x = df_ntr$POT_NTR,
    obs_y = df_ntr$Max_RF,
    
    sim_x = df_sim_ntr$Sim_NTR,
    sim_y = df_sim_ntr$Sim_RF,
    
    xlab = "Non-tidal Residual (m)",
    ylab = "Rainfall (mm)",
    
    bins = bins,
    
    obs_label = paste0(
      "Observed (n = ",
      n_obs_ntr,
      ")"
    ),
    
    sim_label = paste0(
      "Simulated (n = ",
      n_sim_ntr,
      ")"
    )
  )
  
  
  p6 <- joint_plot_r(
    obs_x = df_ntr$POT_NTR,
    obs_y = df_ntr$Max_RD,
    
    sim_x = df_sim_ntr$Sim_NTR,
    sim_y = df_sim_ntr$Sim_RD,
    
    xlab = "Non-tidal Residual (m)",
    
    ylab = expression(
      "River Discharge (m"^3*"/s)"
    ),
    
    bins = bins,
    
    obs_label = paste0(
      "Observed (n = ",
      n_obs_ntr,
      ")"
    ),
    
    sim_label = paste0(
      "Simulated (n = ",
      n_sim_ntr,
      ")"
    )
  )
  
  
  row3 <- (
    p5 | p6
  ) +
    
    plot_annotation(
      title = "Peak-over-threshold Non-tidal Residual",
      
      theme = theme(
        plot.title = element_text(
          size = 15,
          face = "bold",
          hjust = 0.5
        )
      )
    )
  
  
  # ============================================================
  # Combine all rows
  # ============================================================
  
  final_plot <- (
    row1 /
      row2 /
      row3
  ) +
    
    plot_layout(
      heights = c(1, 1, 1),
      guides = "collect"
    ) &
    
    theme(
      legend.position = "top",
      
      legend.text = element_text(
        size = 13
      )
    )
  
  
  # ============================================================
  # Save
  # ============================================================
  
  output_path <- file.path(
    base_dir,
    output_file
  )
  
  
  ggsave(
    filename = output_path,
    plot = final_plot,
    width = width,
    height = height,
    dpi = dpi,
    units = "in",
    bg = "white"
  )
  
  
  # ------------------------------------------------------------
  # Return useful objects
  # ------------------------------------------------------------
  
  return(
    list(
      plot = final_plot,
      
      RF_observed = df_rf,
      RF_simulated = df_sim_rf,
      
      RD_observed = df_rd,
      RD_simulated = df_sim_rd,
      
      NTR_observed = df_ntr,
      NTR_simulated = df_sim_ntr,
      
      output_path = output_path
    )
  )
}
