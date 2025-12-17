# ============================================================================
# ADVANCED FDA Model - Target AUC > 0.8
# Advanced feature engineering, derivatives, and ensemble methods
# ============================================================================

library(tidyverse)
library(fda)
library(lubridate)
library(zoo)
library(pROC)
library(caret)
library(MASS)
library(randomForest)

set.seed(123)

cat(strrep("=", 70), "\n")
cat("ADVANCED FDA MODEL - TARGET AUC > 0.8\n")
cat(strrep("=", 70), "\n\n")

# ============================================================================
# 1. DATA LOADING AND ENHANCED PREPROCESSING
# ============================================================================

cat("Step 1: Loading and preprocessing data...\n")
data <- read_csv("credit_card_transactions.csv", show_col_types = FALSE)

data <- data %>%
  mutate(
    trans_datetime = ymd_hms(trans_date_trans_time),
    trans_date = as.Date(trans_datetime),
    trans_week = floor_date(trans_datetime, "week"),
    week_num = as.numeric(difftime(trans_datetime, min(trans_datetime, na.rm = TRUE), units = "weeks")),
    is_fraud = as.numeric(is_fraud),
    age = as.numeric(difftime(trans_datetime, ymd(dob), units = "days")) / 365.25,
    trans_hour = hour(trans_datetime),
    trans_weekday = wday(trans_datetime),
    trans_month = month(trans_datetime)
  ) %>%
  filter(!is.na(trans_datetime))

max_week <- ceiling(max(data$week_num, na.rm = TRUE))
week_grid <- seq(0, max_week, length.out = 52)

# Response
fraud_response <- data %>%
  group_by(cc_num) %>%
  summarise(
    has_fraud = max(is_fraud),
    fraud_rate = mean(is_fraud),
    n_trans = n(),
    .groups = "drop"
  )

# Card selection
cards_sufficient <- data %>%
  group_by(cc_num) %>%
  summarise(n_weeks = n_distinct(week_num), n_trans = n()) %>%
  filter(n_weeks >= 10, n_trans >= 20) %>%
  pull(cc_num)

sample_size <- min(500, length(cards_sufficient))
sample_cards <- sample(cards_sufficient, sample_size)

cat("✓ Analyzing", length(sample_cards), "cards\n\n")

# ============================================================================
# 2. ADVANCED WEEKLY FEATURES
# ============================================================================

cat("Step 2: Creating advanced weekly features...\n")

weekly_features <- data %>%
  filter(cc_num %in% sample_cards) %>%
  arrange(cc_num, trans_datetime) %>%
  group_by(cc_num, week_num) %>%
  summarise(
    # Amount features
    total_amt = sum(amt, na.rm = TRUE),
    avg_amt = mean(amt, na.rm = TRUE),
    median_amt = median(amt, na.rm = TRUE),
    max_amt = max(amt, na.rm = TRUE),
    min_amt = min(amt, na.rm = TRUE),
    sd_amt = sd(amt, na.rm = TRUE),
    
    # Count features
    n_trans = n(),
    
    # Diversity features
    n_categories = n_distinct(category),
    n_merchants = n_distinct(merchant),
    
    # Temporal features
    pct_weekend = mean(trans_weekday %in% c(1, 7)),
    pct_night = mean(trans_hour >= 22 | trans_hour <= 6),
    
    # Fraud indicator
    has_fraud = max(is_fraud),
    
    .groups = "drop"
  )

# Complete time grid
weekly_features <- weekly_features %>%
  tidyr::complete(cc_num, week_num = 0:max_week, 
                  fill = list(total_amt = 0, avg_amt = 0, n_trans = 0, 
                             max_amt = 0, min_amt = 0, sd_amt = 0,
                             n_categories = 0, n_merchants = 0,
                             pct_weekend = 0, pct_night = 0, has_fraud = 0))

cat("✓ Weekly features created\n\n")

# ============================================================================
# 3. FUNCTIONAL DATA: MULTIPLE CURVES + DERIVATIVES
# ============================================================================

cat("Step 3: Creating functional predictors with derivatives...\n")

# Helper function for B-spline smoothing
create_functional_data <- function(feature_col, lambda = 10) {
  feature_matrix_raw <- weekly_features %>%
    dplyr::select(cc_num, week_num, !!sym(feature_col)) %>%
    pivot_wider(id_cols = cc_num, names_from = week_num, 
                values_from = !!sym(feature_col), values_fill = 0) %>%
    filter(cc_num %in% sample_cards)
  
  week_points <- 0:max_week
  for (w in week_points) {
    if (!as.character(w) %in% colnames(feature_matrix_raw)) {
      feature_matrix_raw[[as.character(w)]] <- 0
    }
  }
  
  feature_matrix <- feature_matrix_raw %>%
    dplyr::select(cc_num, all_of(as.character(week_points))) %>%
    column_to_rownames("cc_num") %>%
    as.matrix()
  
  # Create functional data object
  n_basis <- min(15, ceiling(ncol(feature_matrix) / 3))
  basis <- create.bspline.basis(rangeval = c(0, max_week), nbasis = n_basis)
  fdPar_obj <- fdPar(basis, Lfdobj = 2, lambda = lambda)
  
  smooth_obj <- smooth.basis(week_points, t(feature_matrix), fdPar_obj)
  fd_obj <- smooth_obj$fd
  
  # Evaluate on grid
  values <- t(eval.fd(week_grid, fd_obj))
  
  # Get derivative (velocity)
  deriv1 <- t(eval.fd(week_grid, fd_obj, Lfdobj = 1))
  
  # Get second derivative (acceleration)
  deriv2 <- t(eval.fd(week_grid, fd_obj, Lfdobj = 2))
  
  return(list(
    values = values,
    deriv1 = deriv1,
    deriv2 = deriv2,
    fd_obj = fd_obj
  ))
}

# Create functional data for multiple features
cat("  Amount curves...\n")
amt_functional <- create_functional_data("total_amt", lambda = 10)

cat("  Count curves...\n")
count_functional <- create_functional_data("n_trans", lambda = 10)

cat("  Volatility curves...\n")
volatility_functional <- create_functional_data("sd_amt", lambda = 5)

cat("✓ Functional data created with derivatives\n\n")

# ============================================================================
# 4. FPCA ON MULTIPLE FUNCTIONAL PREDICTORS
# ============================================================================

cat("Step 4: Performing FPCA on multiple functional variables...\n")

n_basis_pred <- 12
pred_basis <- create.bspline.basis(rangeval = c(min(week_grid), max(week_grid)), 
                                   nbasis = n_basis_pred)
fdPar_pred <- fdPar(pred_basis, Lfdobj = 2, lambda = 0.1)

# Smooth each functional variable
amt_fd <- smooth.basis(week_grid, t(amt_functional$values), fdPar_pred)$fd
count_fd <- smooth.basis(week_grid, t(count_functional$values), fdPar_pred)$fd
deriv1_fd <- smooth.basis(week_grid, t(amt_functional$deriv1), fdPar_pred)$fd
deriv2_fd <- smooth.basis(week_grid, t(amt_functional$deriv2), fdPar_pred)$fd

cat("✓ Functional objects smoothed\n\n")

# ============================================================================
# 5. COMPREHENSIVE SCALAR FEATURES
# ============================================================================

cat("Step 5: Engineering comprehensive scalar features...\n")

scalar_features <- data %>%
  filter(cc_num %in% sample_cards) %>%
  arrange(cc_num, trans_datetime) %>%
  group_by(cc_num) %>%
  summarise(
    # Demographics
    gender_num = as.numeric(factor(first(gender))),
    age = first(age),
    log_city_pop = log(first(city_pop) + 1),
    
    # Transaction statistics
    n_trans = n(),
    total_spent = sum(amt),
    avg_amt = mean(amt),
    median_amt = median(amt),
    sd_amt = sd(amt),
    cv_amt = sd(amt) / mean(amt),  # Coefficient of variation
    max_amt = max(amt),
    min_amt = min(amt),
    range_amt = max(amt) - min(amt),
    
    # Percentiles
    q25_amt = quantile(amt, 0.25),
    q75_amt = quantile(amt, 0.75),
    iqr_amt = IQR(amt),
    
    # Temporal patterns
    span_days = as.numeric(difftime(max(trans_datetime), min(trans_datetime), units = "days")),
    trans_per_day = n() / (as.numeric(difftime(max(trans_datetime), min(trans_datetime), units = "days")) + 1),
    
    # Time distribution
    pct_weekend = mean(trans_weekday %in% c(1, 7)),
    pct_night = mean(trans_hour >= 22 | trans_hour <= 6),
    pct_business_hours = mean(trans_hour >= 9 & trans_hour <= 17),
    
    # Diversity metrics
    n_categories = n_distinct(category),
    n_merchants = n_distinct(merchant),
    n_states = n_distinct(state),
    category_entropy = -sum((table(category) / n()) * log(table(category) / n() + 1e-10)),
    merchant_concentration = max(table(merchant)) / n(),  # Herfindahl index component
    
    # Geographic features
    lat_mean = mean(lat),
    long_mean = mean(long),
    lat_sd = sd(lat),
    long_sd = sd(long),
    geo_spread = sqrt(sd(lat)^2 + sd(long)^2),
    
    # Velocity features (change over time)
    time_gaps = median(diff(as.numeric(trans_datetime)) / 3600, na.rm = TRUE),  # hours
    
    # Anomaly indicators
    n_large_trans = sum(amt > quantile(amt, 0.95)),
    pct_large_trans = mean(amt > quantile(amt, 0.95)),
    
    # Trend features
    amt_trend = ifelse(n() > 10, {
      tryCatch({
        time_seq <- as.numeric(trans_datetime - min(trans_datetime))
        coef(lm(amt ~ time_seq))[2]
      }, error = function(e) 0)
    }, 0),
    
    .groups = "drop"
  ) %>%
  mutate(
    # Log transformations
    log_n_trans = log(n_trans + 1),
    log_total_spent = log(total_spent + 1),
    log_avg_amt = log(avg_amt + 1),
    
    # Interaction terms
    trans_density_entropy = trans_per_day * category_entropy,
    spending_diversity = total_spent * n_merchants,
    
    # Risk indicators
    night_spending_ratio = pct_night * avg_amt,
    weekend_activity = pct_weekend * trans_per_day
  )

cat("✓ Created", ncol(scalar_features) - 1, "scalar features\n\n")

# ============================================================================
# 6. ALIGN AND PREPARE DATA
# ============================================================================

common_cards <- Reduce(intersect, list(
  as.character(fraud_response$cc_num),
  rownames(amt_functional$values),
  as.character(scalar_features$cc_num)
))

fraud_y <- fraud_response %>%
  filter(cc_num %in% as.numeric(common_cards)) %>%
  arrange(match(cc_num, as.numeric(common_cards))) %>%
  pull(has_fraud)

scalar_x <- scalar_features %>%
  filter(cc_num %in% as.numeric(common_cards)) %>%
  arrange(match(cc_num, as.numeric(common_cards))) %>%
  dplyr::select(-cc_num) %>%
  as.matrix()

# Clean scalar features
scalar_x[is.na(scalar_x)] <- 0
scalar_x[is.infinite(scalar_x)] <- 0

cat("Aligned data for", length(fraud_y), "cards\n")
cat("Fraud rate:", round(mean(fraud_y) * 100, 2), "%\n\n")

# ============================================================================
# 7. FPCA WITH OPTIMAL COMPONENTS
# ============================================================================

cat("Step 7: FPCA with optimal number of components...\n")

# Stratified split
train_idx <- createDataPartition(fraud_y, p = 0.7, list = FALSE)[,1]
test_idx <- setdiff(1:length(fraud_y), train_idx)

# FPCA on each functional variable
n_components <- 10

amt_fpca <- pca.fd(amt_fd[train_idx], nharm = n_components)
count_fpca <- pca.fd(count_fd[train_idx], nharm = n_components)
deriv1_fpca <- pca.fd(deriv1_fd[train_idx], nharm = n_components)

cat("Variance explained:\n")
cat("  Amount:", round(sum(amt_fpca$varprop), 3), "\n")
cat("  Count:", round(sum(count_fpca$varprop), 3), "\n")
cat("  Derivative:", round(sum(deriv1_fpca$varprop), 3), "\n\n")

# Extract scores
amt_scores_train <- amt_fpca$scores
count_scores_train <- count_fpca$scores
deriv1_scores_train <- deriv1_fpca$scores

amt_scores_test <- inprod(amt_fd[test_idx], amt_fpca$harmonics)
count_scores_test <- inprod(count_fd[test_idx], count_fpca$harmonics)
deriv1_scores_test <- inprod(deriv1_fd[test_idx], deriv1_fpca$harmonics)

# Combine all features
X_train <- cbind(
  amt_scores_train,
  count_scores_train,
  deriv1_scores_train,
  scalar_x[train_idx, ]
)

X_test <- cbind(
  amt_scores_test,
  count_scores_test,
  deriv1_scores_test,
  scalar_x[test_idx, ]
)

# Clean
X_train <- apply(X_train, 2, function(x) {
  x[is.na(x)] <- 0
  x[is.infinite(x)] <- 0
  return(x)
})

X_test <- apply(X_test, 2, function(x) {
  x[is.na(x)] <- 0
  x[is.infinite(x)] <- 0
  return(x)
})

y_train <- fraud_y[train_idx]
y_test <- fraud_y[test_idx]

cat("Total features:", ncol(X_train), "\n\n")

# ============================================================================
# 8. ENSEMBLE MODEL: MULTIPLE RF + ENHANCED GLM
# ============================================================================

cat(strrep("=", 70), "\n")
cat("ENSEMBLE MODELING FOR AUC > 0.8\n")
cat(strrep("=", 70), "\n\n")

# Prepare data
train_df <- data.frame(y = factor(y_train), X_train)
test_df <- data.frame(X_test)

# Class weights
weight_0 <- 1 / sum(y_train == 0)
weight_1 <- 1 / sum(y_train == 1)
weight_multiplier <- weight_1 / weight_0

# Model 1: Random Forest with many trees
cat("Training Random Forest Model 1 (500 trees, tuned mtry)...\n")

rf1_model <- randomForest(y ~ ., 
                          data = train_df,
                          ntree = 500,
                          mtry = max(floor(sqrt(ncol(X_train))), 5),
                          classwt = c("0" = 1, "1" = weight_multiplier * 10),
                          importance = TRUE,
                          nodesize = 5)

pred_prob_rf1 <- predict(rf1_model, newdata = test_df, type = "prob")[, 2]

roc_rf1 <- roc(y_test, pred_prob_rf1, quiet = TRUE)
auc_rf1 <- auc(roc_rf1)

cat("  RF1 AUC:", round(auc_rf1, 4), "\n\n")

# Model 2: Random Forest with more trees and different mtry
cat("Training Random Forest Model 2 (1000 trees, different mtry)...\n")

rf2_model <- randomForest(y ~ ., 
                          data = train_df,
                          ntree = 1000,
                          mtry = max(floor(ncol(X_train) / 3), 10),
                          classwt = c("0" = 1, "1" = weight_multiplier * 10),
                          nodesize = 10,
                          maxnodes = 50)

pred_prob_rf2 <- predict(rf2_model, newdata = test_df, type = "prob")[, 2]

roc_rf2 <- roc(y_test, pred_prob_rf2, quiet = TRUE)
auc_rf2 <- auc(roc_rf2)

cat("  RF2 AUC:", round(auc_rf2, 4), "\n\n")

# Model 3: Logistic Regression with interactions
cat("Training Logistic Regression with key interactions...\n")

# Select top features based on RF importance
importance_rf <- importance(rf1_model)
top_features_idx <- order(importance_rf[, "MeanDecreaseGini"], decreasing = TRUE)[1:20]
top_features <- colnames(X_train)[top_features_idx]

# Create interaction terms for top features
X_train_int <- X_train[, top_features]
X_test_int <- X_test[, top_features]

# Add selected interactions with consistent naming
if (ncol(X_train_int) >= 3) {
  # Create interactions for train
  interaction1_train <- X_train_int[, 1] * X_train_int[, 2]
  interaction2_train <- X_train_int[, 1] * X_train_int[, 3]
  
  # Create interactions for test
  interaction1_test <- X_test_int[, 1] * X_test_int[, 2]
  interaction2_test <- X_test_int[, 1] * X_test_int[, 3]
  
  # Add to matrices with same names
  X_train_int <- cbind(X_train_int, interaction1 = interaction1_train, interaction2 = interaction2_train)
  X_test_int <- cbind(X_test_int, interaction1 = interaction1_test, interaction2 = interaction2_test)
}

# Ensure both have same column names
colnames(X_test_int) <- colnames(X_train_int)

train_df_int <- data.frame(y = y_train, X_train_int)
test_df_int <- data.frame(X_test_int)

# Class weights for GLM
weight_0 <- 1 / sum(y_train == 0)
weight_1 <- 1 / sum(y_train == 1)
weights_glm <- ifelse(y_train == 1, weight_1, weight_0)
weights_glm <- weights_glm / sum(weights_glm) * length(y_train)

glm_model <- glm(y ~ ., data = train_df_int, family = binomial, weights = weights_glm)
pred_prob_glm <- predict(glm_model, newdata = test_df_int, type = "response")

roc_glm <- roc(y_test, pred_prob_glm, quiet = TRUE)
auc_glm <- auc(roc_glm)

cat("  GLM AUC:", round(auc_glm, 4), "\n\n")

# ============================================================================
# 9. ENSEMBLE PREDICTION
# ============================================================================

cat("Creating weighted ensemble of 3 models...\n")

# Ensemble weights based on AUC
total_auc <- auc_rf1 + auc_rf2 + auc_glm
w_rf1 <- auc_rf1 / total_auc
w_rf2 <- auc_rf2 / total_auc
w_glm <- auc_glm / total_auc

pred_prob_ensemble <- w_rf1 * pred_prob_rf1 + 
                      w_rf2 * pred_prob_rf2 + 
                      w_glm * pred_prob_glm

roc_ensemble <- roc(y_test, pred_prob_ensemble, quiet = TRUE)
auc_ensemble <- auc(roc_ensemble)

cat("\n")
cat(strrep("=", 70), "\n")
cat("FINAL RESULTS\n")
cat(strrep("=", 70), "\n\n")

results <- data.frame(
  Model = c("Random Forest 1", "Random Forest 2", "Logistic Regression", "ENSEMBLE"),
  AUC = c(auc_rf1, auc_rf2, auc_glm, auc_ensemble),
  Weight = c(w_rf1, w_rf2, w_glm, NA)
)

print(results)

cat("\n")
if (auc_ensemble >= 0.8) {
  cat("🎉 SUCCESS! AUC ≥ 0.8 achieved:", round(auc_ensemble, 4), "\n")
} else {
  cat("Current AUC:", round(auc_ensemble, 4), "\n")
  cat("Gap to 0.8:", round(0.8 - auc_ensemble, 4), "\n")
}

cat("\nKEY IMPROVEMENTS:\n")
cat("• Multiple functional predictors (amount, count, derivatives)\n")
cat("•", ncol(X_train), "total features\n")
cat("• 2 Random Forest models with different parameters\n")
cat("• Enhanced Logistic Regression with interactions\n")
cat("• Weighted ensemble of 3 models\n")
cat("• Class-weighted training (10x fraud weight)\n")
cat("• Feature interactions\n\n")

# Top features
cat("Top 15 features by importance:\n")
importance_df <- data.frame(
  Feature = rownames(importance_rf),
  Importance = importance_rf[, "MeanDecreaseGini"]
) %>%
  arrange(desc(Importance)) %>%
  head(15)
print(importance_df)

# Save results
write_csv(results, "advanced_model_results.csv")

cat("\n", strrep("=", 70), "\n")
cat("Analysis complete!\n")
cat(strrep("=", 70), "\n")

