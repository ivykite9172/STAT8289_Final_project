# ============================================================================
# IMPROVED Scalar-on-Function FDA Model with Enhanced Features
# Goal: Increase AUC through better feature engineering and modeling
# ============================================================================

library(tidyverse)
library(fda)
library(lubridate)
library(zoo)
library(pROC)
library(caret)
library(MASS)  # For stepwise selection

set.seed(123)

cat(strrep("=", 70), "\n")
cat("IMPROVED FDA MODEL FOR FRAUD DETECTION\n")
cat(strrep("=", 70), "\n\n")

# Load data (reusing previous preprocessing)
cat("Loading data...\n")
data <- read_csv("credit_card_transactions.csv", show_col_types = FALSE)

data <- data %>%
  mutate(
    trans_datetime = ymd_hms(trans_date_trans_time),
    trans_week = floor_date(trans_datetime, "week"),
    week_num = as.numeric(difftime(trans_datetime, min(trans_datetime, na.rm = TRUE), units = "weeks")),
    is_fraud = as.numeric(is_fraud),
    age = as.numeric(difftime(trans_datetime, ymd(dob), units = "days")) / 365.25,
    trans_hour = hour(trans_datetime),
    trans_weekday = wday(trans_datetime)
  ) %>%
  filter(!is.na(trans_datetime))

max_week <- ceiling(max(data$week_num, na.rm = TRUE))
week_grid <- seq(0, max_week, length.out = 52)

# Create scalar response
fraud_response <- data %>%
  group_by(cc_num) %>%
  summarise(
    has_fraud = max(is_fraud),
    fraud_rate = mean(is_fraud),
    n_trans = n(),
    .groups = "drop"
  )

cat("✓ Data loaded\n\n")

# Select cards with sufficient data
cards_sufficient <- data %>%
  group_by(cc_num) %>%
  summarise(n_weeks = n_distinct(week_num), n_trans = n()) %>%
  filter(n_weeks >= 10, n_trans >= 20) %>%
  pull(cc_num)

sample_size <- min(500, length(cards_sufficient))
sample_cards <- sample(cards_sufficient, sample_size)

# Weekly features
weekly_features <- data %>%
  filter(cc_num %in% sample_cards) %>%
  group_by(cc_num, week_num) %>%
  summarise(
    total_amt = sum(amt, na.rm = TRUE),
    avg_amt = mean(amt, na.rm = TRUE),
    n_trans = n(),
    max_amt = max(amt, na.rm = TRUE),
    min_amt = min(amt, na.rm = TRUE),
    sd_amt = sd(amt, na.rm = TRUE),
    n_categories = n_distinct(category),
    .groups = "drop"
  ) %>%
  tidyr::complete(cc_num, week_num = 0:max_week, 
                  fill = list(total_amt = 0, avg_amt = 0, n_trans = 0, 
                             max_amt = 0, min_amt = 0, sd_amt = 0, n_categories = 0))

cat("Creating functional predictors with optimal settings...\n")

# ============================================================================
# KDE with optimal bandwidth
# ============================================================================

perform_kde <- function(card_data, bw_method = "SJ") {
  if (nrow(card_data) < 3 || sum(card_data$total_amt) == 0) {
    return(rep(0, length(week_grid)))
  }
  tryCatch({
    weights <- card_data$total_amt / sum(card_data$total_amt)
    kde_result <- density(card_data$week_num, weights = weights, bw = bw_method, n = 512)
    approx(kde_result$x, kde_result$y, xout = week_grid, rule = 2)$y
  }, error = function(e) rep(0, length(week_grid)))
}

kde_amt_results <- weekly_features %>%
  group_by(cc_num) %>%
  group_modify(~ tibble(week_grid = week_grid, kde_density = perform_kde(.x))) %>%
  ungroup()

kde_amt_matrix <- kde_amt_results %>%
  pivot_wider(id_cols = cc_num, names_from = week_grid, values_from = kde_density) %>%
  filter(cc_num %in% sample_cards) %>%
  column_to_rownames("cc_num") %>%
  as.matrix()

# ============================================================================
# KME with optimal lambda
# ============================================================================

week_points <- 0:max_week
amt_matrix_raw <- weekly_features %>%
  dplyr::select(cc_num, week_num, total_amt) %>%
  pivot_wider(id_cols = cc_num, names_from = week_num, values_from = total_amt, values_fill = 0) %>%
  filter(cc_num %in% sample_cards)

for (w in week_points) {
  if (!as.character(w) %in% colnames(amt_matrix_raw)) {
    amt_matrix_raw[[as.character(w)]] <- 0
  }
}

amt_matrix <- amt_matrix_raw %>%
  dplyr::select(cc_num, all_of(as.character(week_points))) %>%
  column_to_rownames("cc_num") %>%
  as.matrix()

n_basis <- min(15, ceiling(ncol(amt_matrix) / 3))
amt_basis <- create.bspline.basis(rangeval = c(0, max_week), nbasis = n_basis)

# Use higher lambda for more smoothing (previous 0.01 was too low)
optimal_lambda <- 10  # Increased from 0.01
amt_fdPar <- fdPar(amt_basis, Lfdobj = 2, lambda = optimal_lambda)
amt_smooth <- smooth.basis(week_points, t(amt_matrix), amt_fdPar)
amt_fd <- amt_smooth$fd
kme_amt_matrix <- t(eval.fd(week_grid, amt_fd))

cat("✓ Functional predictors created\n\n")

# ============================================================================
# ENHANCED SCALAR COVARIATES
# ============================================================================

cat("Creating enhanced scalar features...\n")

scalar_enhanced <- data %>%
  filter(cc_num %in% sample_cards) %>%
  group_by(cc_num) %>%
  summarise(
    # Demographics
    gender = first(gender),
    age = first(age),
    city_pop = first(city_pop),
    lat = mean(lat, na.rm = TRUE),
    long = mean(long, na.rm = TRUE),
    
    # Transaction patterns
    total_trans = n(),
    avg_amt = mean(amt, na.rm = TRUE),
    median_amt = median(amt, na.rm = TRUE),
    max_amt = max(amt, na.rm = TRUE),
    min_amt = min(amt, na.rm = TRUE),
    sd_amt = sd(amt, na.rm = TRUE),
    cv_amt = sd(amt, na.rm = TRUE) / mean(amt, na.rm = TRUE),  # Coefficient of variation
    
    # Temporal features
    span_days = as.numeric(difftime(max(trans_datetime), min(trans_datetime), units = "days")),
    trans_per_day = n() / (as.numeric(difftime(max(trans_datetime), min(trans_datetime), units = "days")) + 1),
    n_weekends = sum(trans_weekday %in% c(1, 7)),
    pct_weekend = mean(trans_weekday %in% c(1, 7)),
    n_night_trans = sum(trans_hour >= 22 | trans_hour <= 6),
    pct_night = mean(trans_hour >= 22 | trans_hour <= 6),
    
    # Category diversity
    n_categories = n_distinct(category),
    category_entropy = -sum((table(category) / n()) * log(table(category) / n() + 1e-10)),
    
    # Merchant diversity
    n_merchants = n_distinct(merchant),
    merchant_entropy = -sum((table(merchant) / n()) * log(table(merchant) / n() + 1e-10)),
    
    # Geographic spread
    lat_range = max(lat) - min(lat),
    long_range = max(long) - min(long),
    
    .groups = "drop"
  ) %>%
  mutate(
    gender_numeric = as.numeric(factor(gender)),
    log_city_pop = log(city_pop + 1),
    log_total_trans = log(total_trans + 1),
    log_avg_amt = log(avg_amt + 1),
    amt_variability = cv_amt  # Renamed for clarity
  ) %>%
  dplyr::select(-gender, -city_pop, -total_trans, -avg_amt, -cv_amt)  # Remove originals, keep transformed

cat("✓ Enhanced features created:", ncol(scalar_enhanced) - 1, "features\n\n")

# ============================================================================
# ALIGN DATA
# ============================================================================

common_cards <- Reduce(intersect, list(
  as.character(fraud_response$cc_num),
  rownames(kde_amt_matrix),
  rownames(kme_amt_matrix),
  as.character(scalar_enhanced$cc_num)
))

fraud_y <- fraud_response %>%
  filter(cc_num %in% as.numeric(common_cards)) %>%
  arrange(match(cc_num, as.numeric(common_cards))) %>%
  pull(has_fraud)

kde_amt_x <- kde_amt_matrix[common_cards, ]
kme_amt_x <- kme_amt_matrix[common_cards, ]

scalar_x <- scalar_enhanced %>%
  filter(cc_num %in% as.numeric(common_cards)) %>%
  arrange(match(cc_num, as.numeric(common_cards))) %>%
  dplyr::select(-cc_num) %>%
  as.matrix()

# Handle any NaN/Inf values
scalar_x[is.na(scalar_x)] <- 0
scalar_x[is.infinite(scalar_x)] <- 0

cat("Data aligned:", length(fraud_y), "cards\n")
cat("Fraud rate:", round(mean(fraud_y) * 100, 2), "%\n\n")

# ============================================================================
# CREATE FD OBJECTS AND PERFORM FPCA WITH MORE COMPONENTS
# ============================================================================

cat("Performing FPCA with more components for better representation...\n")

n_basis_pred <- 12
pred_basis <- create.bspline.basis(rangeval = c(min(week_grid), max(week_grid)), nbasis = n_basis_pred)
fdPar_pred <- fdPar(pred_basis, Lfdobj = 2, lambda = 0.1)

kde_amt_fd <- smooth.basis(week_grid, t(kde_amt_x), fdPar_pred)$fd
kme_amt_fd <- smooth.basis(week_grid, t(kme_amt_x), fdPar_pred)$fd

# ============================================================================
# TRAIN/TEST SPLIT WITH STRATIFICATION
# ============================================================================

train_prop <- 0.7
train_idx <- createDataPartition(fraud_y, p = train_prop, list = FALSE)[,1]
test_idx <- setdiff(1:length(fraud_y), train_idx)

cat("\nTrain/test split:\n")
cat("  Train:", length(train_idx), "cards, fraud rate:", 
    round(mean(fraud_y[train_idx]) * 100, 2), "%\n")
cat("  Test:", length(test_idx), "cards, fraud rate:", 
    round(mean(fraud_y[test_idx]) * 100, 2), "%\n\n")

# ============================================================================
# IMPROVED MODEL: MORE FPCA COMPONENTS + REGULARIZATION
# ============================================================================

cat(strrep("=", 70), "\n")
cat("IMPROVED MODEL WITH ENHANCED FEATURES\n")
cat(strrep("=", 70), "\n\n")

# Use more FPCA components (10 instead of 5)
n_components <- 10

cat("Using", n_components, "FPCA components for better feature representation\n")

kde_amt_fpca <- pca.fd(kde_amt_fd[train_idx], nharm = n_components)
kme_amt_fpca <- pca.fd(kme_amt_fd[train_idx], nharm = n_components)

cat("Variance explained:\n")
cat("  KDE:", round(sum(kde_amt_fpca$varprop[1:n_components]), 3), "\n")
cat("  KME:", round(sum(kme_amt_fpca$varprop[1:n_components]), 3), "\n\n")

# Get FPC scores
kde_scores_train <- kde_amt_fpca$scores
kde_scores_test <- inprod(kde_amt_fd[test_idx], kde_amt_fpca$harmonics)

kme_scores_train <- kme_amt_fpca$scores
kme_scores_test <- inprod(kme_amt_fd[test_idx], kme_amt_fpca$harmonics)

# Derive additional features from functional data
cat("Deriving additional features from functional curves...\n")

# Extract functional characteristics
extract_curve_features <- function(fd_obj) {
  curves <- eval.fd(week_grid, fd_obj)
  data.frame(
    curve_mean = colMeans(curves),
    curve_sd = apply(curves, 2, sd),
    curve_max = apply(curves, 2, max),
    curve_min = apply(curves, 2, min),
    curve_range = apply(curves, 2, max) - apply(curves, 2, min),
    curve_skew = apply(curves, 2, function(x) {
      n <- length(x); m <- mean(x); s <- sd(x)
      n * sum((x - m)^3) / ((n - 1) * (n - 2) * s^3)
    })
  )
}

curve_features_train <- extract_curve_features(kde_amt_fd[train_idx])
curve_features_test <- extract_curve_features(kde_amt_fd[test_idx])

# Combine all features - ensure all are matrices
X_train <- cbind(
  as.matrix(kde_scores_train),
  as.matrix(kme_scores_train),
  as.matrix(scalar_x[train_idx, ]),
  as.matrix(curve_features_train)
)

X_test <- cbind(
  as.matrix(kde_scores_test),
  as.matrix(kme_scores_test),
  as.matrix(scalar_x[test_idx, ]),
  as.matrix(curve_features_test)
)

# Handle any remaining NA/Inf
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

cat("✓ Total features:", ncol(X_train), "\n\n")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION WITH STEPWISE SELECTION
# ============================================================================

cat("Fitting Logistic Regression with stepwise feature selection...\n")

# Calculate class weights to handle imbalance
weight_0 <- 1 / sum(y_train == 0)
weight_1 <- 1 / sum(y_train == 1)
weights <- ifelse(y_train == 1, weight_1, weight_0)
weights <- weights / sum(weights) * length(y_train)

# Create data frame
train_df_full <- data.frame(y = y_train, X_train)
test_df_full <- data.frame(X_test)

# Fit full logistic regression model
full_model <- glm(y ~ ., data = train_df_full, family = binomial, weights = weights)

# Stepwise selection based on AIC
cat("  Performing stepwise selection...\n")
step_model <- step(full_model, direction = "both", trace = 0)

cat("  Features selected:", length(coef(step_model)) - 1, "out of", ncol(X_train), "\n")

# Predictions
pred_prob_glm <- predict(step_model, newdata = test_df_full, type = "response")
pred_class_glm <- ifelse(pred_prob_glm > 0.5, 1, 0)

# Evaluate
roc_glm <- roc(y_test, pred_prob_glm, quiet = TRUE)
auc_glm <- auc(roc_glm)

cat("\n✓ Stepwise Logistic Regression Model:\n")
cat("  AUC:", round(auc_glm, 4), "\n")
cat("  Accuracy:", round(mean(pred_class_glm == y_test), 4), "\n")
conf_mat <- confusionMatrix(factor(pred_class_glm), factor(y_test))
cat("  Sensitivity:", round(conf_mat$byClass["Sensitivity"], 4), "\n")
cat("  Specificity:", round(conf_mat$byClass["Specificity"], 4), "\n\n")

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

cat("Fitting Random Forest for comparison...\n")

library(randomForest)

# Convert to data frames
train_df <- data.frame(y = factor(y_train), X_train)
test_df <- data.frame(y = factor(y_test), X_test)

# Fit RF with balanced class weights
rf_model <- randomForest(y ~ ., data = train_df,
                         ntree = 500,
                         mtry = floor(sqrt(ncol(X_train))),
                         classwt = c("0" = weight_0 * 100, "1" = weight_1 * 100),
                         importance = TRUE)

# Predictions
pred_prob_rf <- predict(rf_model, newdata = test_df, type = "prob")[, 2]
pred_class_rf <- predict(rf_model, newdata = test_df, type = "class")

# Evaluate
roc_rf <- roc(y_test, pred_prob_rf, quiet = TRUE)
auc_rf <- auc(roc_rf)

cat("\n✓ Random Forest Model:\n")
cat("  AUC:", round(auc_rf, 4), "\n")
cat("  Accuracy:", round(mean(as.numeric(as.character(pred_class_rf)) == y_test), 4), "\n\n")

# Feature importance
cat("Top 10 most important features:\n")
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
) %>%
  arrange(desc(Importance)) %>%
  head(10)
print(importance_df)

# ============================================================================
# COMPARISON
# ============================================================================

cat("\n\n")
cat(strrep("=", 70), "\n")
cat("MODEL COMPARISON\n")
cat(strrep("=", 70), "\n\n")

comparison <- data.frame(
  Model = c("Stepwise GLM", "Random Forest"),
  AUC = c(auc_glm, auc_rf),
  Features_Used = c(length(coef(step_model)) - 1, ncol(X_train)),
  Features_Available = c(ncol(X_train), ncol(X_train))
)

print(comparison)

cat("\n✓ IMPROVEMENT SUMMARY:\n")
cat("  • Increased FPCA components: 5 → 10\n")
cat("  • Enhanced scalar features: +", (ncol(scalar_enhanced) - 1) - ncol(scalar_x), "\n")
cat("  • Added curve-derived features: +6\n")
cat("  • Total features:", ncol(X_train), "\n")
cat("  • Applied class weighting for imbalance\n")
cat("  • Used stepwise feature selection (AIC)\n")
cat("  • Compared with Random Forest\n")
cat("  • Better smoothing parameter (λ: 0.01 → 10)\n\n")

# Save results
write_csv(comparison, "improved_model_comparison.csv")

cat(strrep("=", 70), "\n")
cat("Analysis complete!\n")
cat(strrep("=", 70), "\n")

