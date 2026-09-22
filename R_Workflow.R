setwd("C:/Users/sadaf/Desktop/Generating_Boundary_Conditions/Multivariate_Codes/Github")

output_folder <- "outputs"
# ============================================================
# 1. Load the functions
# ============================================================

source("pipelines/Marginals_and_Copulas.R")


# ============================================================
# 2. Define inputs
# ============================================================

path_to_threshold_df <- file.path(output_folder, "Thresholds_RD.csv")
path_to_pot_df <- file.path(output_folder, "POT_RD_Functions.csv")

center_var_col_name_pot_df <- "POT_RD"
associated_var1 <- "Max_NTR"
associated_var2 <- "Max_RF"

time_col_pot_df <- "time"
time_col_threshold_df <- "Year"
threshold_col_name_threshold_df <- 'Threshold'

n_events_year <- 5

center_var_u_name <- "U_RD"
associated_var1_u_name <- "U_NTR"
associated_var2_u_name <- "U_RF"

center_var_sim_name <- "Sim_RD"
associated_var1_sim_name <- "Sim_NTR"
associated_var2_sim_name <- "Sim_RF"

# ============================================================
# 3. Load and prepare data
# ============================================================

data <- load_data(
  path_to_pot_df = path_to_pot_df,
  path_to_threshold_df = path_to_threshold_df,
  center_var_col_name_pot_df = center_var_col_name_pot_df,
  associated_var1_col_name_pot_df = associated_var1,
  associated_var2_col_name_pot_df = associated_var2,
  time_col_pot_df = time_col_pot_df,
  time_col_threshold_df = time_col_threshold_df,
  threshold_col_name_threshold_df = threshold_col_name_threshold_df
)

# ============================================================
# 4. Fit nonstationary GPD for center variable
# ============================================================

gpd_result <- gpd_dist(
  data = data,
  center_var_col_name_pot_df = center_var_col_name_pot_df
)

# ============================================================
# 5. Fit associated-variable marginals
# ============================================================

result_var1 <- fit_marginal(
  df = data$df,
  mask = gpd_result$mask,
  associated_var = associated_var1
)

result_var2 <- fit_marginal(
  df = data$df,
  mask = gpd_result$mask,
  associated_var = associated_var2
)

# ============================================================
# 6. Create copula dataframe
# ============================================================

copula_df <- data.frame(
  
  Datetime = data$df[[time_col_pot_df]][gpd_result$mask],
  
  Year = lubridate::year(
    data$df[[time_col_pot_df]][gpd_result$mask]
  ),
  
  U_Var1 = result_var1$u,
  U_Var2 = result_var2$u,
  U_Center = gpd_result$u_rd
)

head(copula_df)
summary(copula_df)

# ============================================================
# 7. Fit and simulate nonstationary vine copula
# ============================================================

copula_result <- fit_simulate_ns_vine(
  copula_df = copula_df,
  n_events_year = n_events_year,
  N = 100,
  u_var1 = "U_Var1",
  u_var2 = "U_Var2",
  u_center = "U_Center",
  year_col = "Year",
  bivariate_var1 = "U_Var1",
  bivariate_var2 = "U_Var2"
)

df_simulated <- copula_result$simulated
df_simulated <- df_simulated %>%
  dplyr::rename(
    !!associated_var1_u_name := U_Var1,
    !!associated_var2_u_name := U_Var2,
    !!center_var_u_name := U_Center
  )

# ============================================================
# 8. Back-transform center variable
# ============================================================

df_simulated <- backtransform_center_gpd(
  df_simulated = df_simulated,
  gpd_result = gpd_result,
  data = data,
  uniform_col = center_var_u_name,
  year_col = "Year",
  output_col = center_var_sim_name
)

# ============================================================
# 9. Back-transform associated variables
# ============================================================

df_simulated <- backtransform_marginal(
  df_simulated = df_simulated,
  marginal_result = result_var1,
  uniform_col = associated_var1_u_name,
  output_col = associated_var1_sim_name
)

df_simulated <- backtransform_marginal(
  df_simulated = df_simulated,
  marginal_result = result_var2,
  uniform_col = associated_var2_u_name,
  output_col = associated_var2_sim_name
)

# ============================================================
# 10. Save results
# ============================================================

write.csv(
  df_simulated,
  paste0("outputs/Simulated_Backtransformed_",center_var_col_name_pot_df,".csv"),
  row.names = FALSE
)


# ============================================================
# 2. Define inputs
# ============================================================

path_to_threshold_df <- file.path(output_folder, "Thresholds_RF.csv")
path_to_pot_df <- file.path(output_folder, "POT_RF_Functions.csv")

center_var_col_name_pot_df <- "POT_RF"
associated_var1 <- "Max_NTR"
associated_var2 <- "Max_RD"

time_col_pot_df <- "time"
time_col_threshold_df <- "Year"
threshold_col_name_threshold_df <- 'Threshold'

n_events_year <- 5

center_var_u_name <- "U_RF"
associated_var1_u_name <- "U_NTR"
associated_var2_u_name <- "U_RD"

center_var_sim_name <- "Sim_RF"
associated_var1_sim_name <- "Sim_NTR"
associated_var2_sim_name <- "Sim_RD"

# ============================================================
# 3. Load and prepare data
# ============================================================

data <- load_data(
  path_to_pot_df = path_to_pot_df,
  path_to_threshold_df = path_to_threshold_df,
  center_var_col_name_pot_df = center_var_col_name_pot_df,
  associated_var1_col_name_pot_df = associated_var1,
  associated_var2_col_name_pot_df = associated_var2,
  time_col_pot_df = time_col_pot_df,
  time_col_threshold_df = time_col_threshold_df,
  threshold_col_name_threshold_df = threshold_col_name_threshold_df
)

# ============================================================
# 4. Fit nonstationary GPD for center variable
# ============================================================

gpd_result <- gpd_dist(
  data = data,
  center_var_col_name_pot_df = center_var_col_name_pot_df
)

# ============================================================
# 5. Fit associated-variable marginals
# ============================================================

result_var1 <- fit_marginal(
  df = data$df,
  mask = gpd_result$mask,
  associated_var = associated_var1
)

result_var2 <- fit_marginal(
  df = data$df,
  mask = gpd_result$mask,
  associated_var = associated_var2
)

# ============================================================
# 6. Create copula dataframe
# ============================================================

copula_df <- data.frame(
  
  Datetime = data$df[[time_col_pot_df]][gpd_result$mask],
  
  Year = lubridate::year(
    data$df[[time_col_pot_df]][gpd_result$mask]
  ),
  
  U_Var1 = result_var1$u,
  U_Var2 = result_var2$u,
  U_Center = gpd_result$u_rd
)

head(copula_df)
summary(copula_df)

# ============================================================
# 7. Fit and simulate nonstationary vine copula
# ============================================================

copula_result <- fit_simulate_ns_vine(
  copula_df = copula_df,
  n_events_year = n_events_year,
  N = 100,
  u_var1 = "U_Var1",
  u_var2 = "U_Var2",
  u_center = "U_Center",
  year_col = "Year",
  bivariate_var1 = "U_Center",
  bivariate_var2 = "U_Var1"
)

df_simulated <- copula_result$simulated
df_simulated <- df_simulated %>%
  dplyr::rename(
    !!associated_var1_u_name := U_Var1,
    !!associated_var2_u_name := U_Var2,
    !!center_var_u_name := U_Center
  )

# ============================================================
# 8. Back-transform center variable
# ============================================================

df_simulated <- backtransform_center_gpd(
  df_simulated = df_simulated,
  gpd_result = gpd_result,
  data = data,
  uniform_col = center_var_u_name,
  year_col = "Year",
  output_col = center_var_sim_name
)

# ============================================================
# 9. Back-transform associated variables
# ============================================================

df_simulated <- backtransform_marginal(
  df_simulated = df_simulated,
  marginal_result = result_var1,
  uniform_col = associated_var1_u_name,
  output_col = associated_var1_sim_name
)

df_simulated <- backtransform_marginal(
  df_simulated = df_simulated,
  marginal_result = result_var2,
  uniform_col = associated_var2_u_name,
  output_col = associated_var2_sim_name
)

# ============================================================
# 10. Save results
# ============================================================

write.csv(
  df_simulated,
  paste0("outputs/Simulated_Backtransformed_",center_var_col_name_pot_df,".csv"),
  row.names = FALSE
)


# ============================================================
# 2. Define inputs
# ============================================================

path_to_threshold_df <- file.path(output_folder, "Thresholds_NTR.csv")
path_to_pot_df <- file.path(output_folder, "POT_NTR_Functions.csv")

center_var_col_name_pot_df <- "POT_NTR"
associated_var1 <- "Max_RF"
associated_var2 <- "Max_RD"

time_col_pot_df <- "time"
time_col_threshold_df <- "Year"
threshold_col_name_threshold_df <- 'Threshold'

n_events_year <- 5

center_var_u_name <- "U_NTR"
associated_var1_u_name <- "U_RF"
associated_var2_u_name <- "U_RD"

center_var_sim_name <- "Sim_NTR"
associated_var1_sim_name <- "Sim_RF"
associated_var2_sim_name <- "Sim_RD"

# ============================================================
# 3. Load and prepare data
# ============================================================

data <- load_data(
  path_to_pot_df = path_to_pot_df,
  path_to_threshold_df = path_to_threshold_df,
  center_var_col_name_pot_df = center_var_col_name_pot_df,
  associated_var1_col_name_pot_df = associated_var1,
  associated_var2_col_name_pot_df = associated_var2,
  time_col_pot_df = time_col_pot_df,
  time_col_threshold_df = time_col_threshold_df,
  threshold_col_name_threshold_df = threshold_col_name_threshold_df
)

# ============================================================
# 4. Fit nonstationary GPD for center variable
# ============================================================

gpd_result <- gpd_dist(
  data = data,
  center_var_col_name_pot_df = center_var_col_name_pot_df
)

# ============================================================
# 5. Fit associated-variable marginals
# ============================================================

result_var1 <- fit_marginal(
  df = data$df,
  mask = gpd_result$mask,
  associated_var = associated_var1
)

result_var2 <- fit_marginal(
  df = data$df,
  mask = gpd_result$mask,
  associated_var = associated_var2
)

# ============================================================
# 6. Create copula dataframe
# ============================================================

copula_df <- data.frame(
  
  Datetime = data$df[[time_col_pot_df]][gpd_result$mask],
  
  Year = lubridate::year(
    data$df[[time_col_pot_df]][gpd_result$mask]
  ),
  
  U_Var1 = result_var1$u,
  U_Var2 = result_var2$u,
  U_Center = gpd_result$u_rd
)

head(copula_df)
summary(copula_df)

# ============================================================
# 7. Fit and simulate nonstationary vine copula
# ============================================================

copula_result <- fit_simulate_ns_vine(
  copula_df = copula_df,
  n_events_year = n_events_year,
  N = 100,
  u_var1 = "U_Var1",
  u_var2 = "U_Var2",
  u_center = "U_Center",
  year_col = "Year",
  bivariate_var1 = "U_Center",
  bivariate_var2 = "U_Var1"
)

df_simulated <- copula_result$simulated
df_simulated <- df_simulated %>%
  dplyr::rename(
    !!associated_var1_u_name := U_Var1,
    !!associated_var2_u_name := U_Var2,
    !!center_var_u_name := U_Center
  )

# ============================================================
# 8. Back-transform center variable
# ============================================================

df_simulated <- backtransform_center_gpd(
  df_simulated = df_simulated,
  gpd_result = gpd_result,
  data = data,
  uniform_col = center_var_u_name,
  year_col = "Year",
  output_col = center_var_sim_name
)

# ============================================================
# 9. Back-transform associated variables
# ============================================================

df_simulated <- backtransform_marginal(
  df_simulated = df_simulated,
  marginal_result = result_var1,
  uniform_col = associated_var1_u_name,
  output_col = associated_var1_sim_name
)

df_simulated <- backtransform_marginal(
  df_simulated = df_simulated,
  marginal_result = result_var2,
  uniform_col = associated_var2_u_name,
  output_col = associated_var2_sim_name
)

# ============================================================
# 10. Save results
# ============================================================

write.csv(
  df_simulated,
  paste0("outputs/Simulated_Backtransformed_",center_var_col_name_pot_df,".csv"),
  row.names = FALSE
)



# ============================================================
# 11. Visualization
# ============================================================

plot_result <- plot_copula_simulated_observed(
  base_dir = output_folder,
  target_events = 5,
  bins = 35,
  output_file = "Copula_Simulated_Observed_All.png"
)

plot_result$plot














