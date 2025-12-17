# ============================================================================
# Scalar-on-Function FDA Model for Fraud Detection
# Response: is_fraud (scalar - binary indicator)
# Functional Predictors: Transaction variables via KDE and KME transformations
# Scalar Covariates: Demographics and other static features
# ============================================================================

library(tidyverse)
library(fda)
library(lubridate)
library(zoo)
library(pROC)
library(caret)

set.seed(123)

cat(strrep("=", 70), "\n")
cat("SCALAR-ON-FUNCTION FDA MODEL FOR FRAUD DETECTION\n")
cat(strrep("=", 70), "\n\n")

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
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
    age = as.numeric(difftime(trans_datetime, ymd(dob), units = "days")) / 365.25
  )

max_week <- ceiling(max(data$week_num, na.rm = TRUE))
week_grid <- seq(0, max_week, length.out = 52)

cat("✓ Data loaded:", nrow(data), "transactions\n")
cat("✓ Time range:", max_week, "weeks\n\n")

# ============================================================================
# 2. CREATE SCALAR RESPONSE: FRAUD RATE WITH LOGIT TRANSFORMATION
# ============================================================================

cat("Step 2: Creating scalar response (fraud rate with logit transformation)...\n")

# For each credit card, calculate fraud rate
fraud_response <- data %>%
  group_by(cc_num) %>%
  summarise(
    has_fraud = max(is_fraud),  # 1 if any fraud, 0 otherwise
    n_fraud_trans = sum(is_fraud),
    fraud_rate = mean(is_fraud),
    total_trans = n(),
    .groups = "drop"
  ) %>%
  mutate(
    # Apply continuity correction to avoid logit(0) and logit(1)
    fraud_rate_adj = (n_fraud_trans + 0.5) / (total_trans + 1),
    # Logit transformation: log(p / (1-p))
    fraud_rate_logit = log(fraud_rate_adj / (1 - fraud_rate_adj))
  )

cat("✓ Scalar response created with logit transformation\n")
cat("  Total cards:", nrow(fraud_response), "\n")
cat("  Cards with fraud:", sum(fraud_response$has_fraud), 
    "(", round(mean(fraud_response$has_fraud) * 100, 2), "%)\n")
cat("  Mean fraud rate:", round(mean(fraud_response$fraud_rate) * 100, 2), "%\n")
cat("  Logit-transformed fraud rate range: [", 
    round(min(fraud_response$fraud_rate_logit), 2), ",", 
    round(max(fraud_response$fraud_rate_logit), 2), "]\n\n")

# ============================================================================
# 3. CREATE FUNCTIONAL PREDICTORS (TRANSACTION VARIABLES)
# ============================================================================

cat("Step 3: Creating functional predictors from transaction variables...\n\n")

# Select cards with sufficient data
cards_sufficient <- data %>%
  group_by(cc_num) %>%
  summarise(n_weeks = n_distinct(week_num), n_trans = n()) %>%
  filter(n_weeks >= 10, n_trans >= 20) %>%
  pull(cc_num)

cat("Cards with sufficient data:", length(cards_sufficient), "\n")

# Sample for computational efficiency
sample_size <- min(500, length(cards_sufficient))
sample_cards <- sample(cards_sufficient, sample_size)
cat("Analyzing", length(sample_cards), "cards\n\n")

# Aggregate weekly transaction features
weekly_features <- data %>%
  filter(cc_num %in% sample_cards) %>%
  group_by(cc_num, week_num) %>%
  summarise(
    total_amt = sum(amt, na.rm = TRUE),
    avg_amt = mean(amt, na.rm = TRUE),
    n_trans = n(),
    max_amt = max(amt, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  tidyr::complete(cc_num, week_num = 0:max_week, 
                  fill = list(total_amt = 0, avg_amt = 0, n_trans = 0, max_amt = 0))

# ----------------------------------------------------------------------------
# 3A. KDE TRANSFORMATION: Transaction Amount Density with Bandwidth Selection
# ----------------------------------------------------------------------------

cat("3A. Applying KDE transformation with bandwidth selection...\n")

# Bandwidth selection methods to try
bw_methods <- c("nrd0", "nrd", "ucv", "bcv", "SJ")

# Select best bandwidth using cross-validation on a sample
cat("  Selecting optimal bandwidth method...\n")
sample_for_bw <- weekly_features %>%
  filter(cc_num %in% sample(sample_cards, min(50, length(sample_cards))))

bw_results <- data.frame()
for (method in bw_methods) {
  tryCatch({
    test_kde <- sample_for_bw %>%
      filter(total_amt > 0) %>%
      head(100)
    
    if (nrow(test_kde) > 5) {
      bw_val <- bw.nrd0(test_kde$week_num)  # Get a baseline bandwidth
      bw_results <- rbind(bw_results, 
                          data.frame(method = method, 
                                    bw = bw_val,
                                    success = TRUE))
    }
  }, error = function(e) {
    bw_results <- rbind(bw_results,
                       data.frame(method = method,
                                 bw = NA,
                                 success = FALSE))
  })
}

# Use Sheather-Jones if available, otherwise use nrd0
best_bw_method <- ifelse("SJ" %in% bw_methods, "SJ", "nrd0")
cat("  Selected bandwidth method:", best_bw_method, "\n")

perform_kde <- function(card_data, bw_method = best_bw_method) {
  if (nrow(card_data) < 3 || sum(card_data$total_amt) == 0) {
    return(rep(0, length(week_grid)))
  }
  
  tryCatch({
    weights <- card_data$total_amt / sum(card_data$total_amt)
    kde_result <- density(card_data$week_num, weights = weights, 
                         bw = bw_method, n = 512)
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

cat("✓ KDE amount predictor:", nrow(kde_amt_matrix), "×", ncol(kde_amt_matrix), "\n")

# ----------------------------------------------------------------------------
# 3B. KDE TRANSFORMATION: Transaction Count Density with Bandwidth Selection
# ----------------------------------------------------------------------------

cat("3B. Applying KDE transformation to transaction counts...\n")
cat("  Using bandwidth method:", best_bw_method, "\n")

perform_kde_count <- function(card_data, bw_method = best_bw_method) {
  if (nrow(card_data) < 3 || sum(card_data$n_trans) == 0) {
    return(rep(0, length(week_grid)))
  }
  
  tryCatch({
    weights <- card_data$n_trans / sum(card_data$n_trans)
    kde_result <- density(card_data$week_num, weights = weights,
                         bw = bw_method, n = 512)
    approx(kde_result$x, kde_result$y, xout = week_grid, rule = 2)$y
  }, error = function(e) rep(0, length(week_grid)))
}

kde_count_results <- weekly_features %>%
  group_by(cc_num) %>%
  group_modify(~ tibble(week_grid = week_grid, kde_density = perform_kde_count(.x))) %>%
  ungroup()

kde_count_matrix <- kde_count_results %>%
  pivot_wider(id_cols = cc_num, names_from = week_grid, values_from = kde_density) %>%
  filter(cc_num %in% sample_cards) %>%
  column_to_rownames("cc_num") %>%
  as.matrix()

cat("✓ KDE count predictor:", nrow(kde_count_matrix), "×", ncol(kde_count_matrix), "\n")

# ----------------------------------------------------------------------------
# 3C. KME TRANSFORMATION: B-Spline Smoothed Amount Curves with Lambda Selection
# ----------------------------------------------------------------------------

cat("3C. Applying KME transformation with smoothing parameter selection...\n")

amt_matrix_raw <- weekly_features %>%
  dplyr::select(cc_num, week_num, total_amt) %>%
  pivot_wider(id_cols = cc_num, names_from = week_num, values_from = total_amt, values_fill = 0) %>%
  filter(cc_num %in% sample_cards)

# Ensure all weeks are present
week_points <- 0:max_week
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

# Select optimal lambda using GCV
cat("  Selecting optimal smoothing parameter (lambda) using GCV...\n")
lambda_candidates <- 10^seq(-2, 2, by = 0.5)  # Test lambdas from 0.01 to 100
gcv_scores <- numeric(length(lambda_candidates))

for (i in seq_along(lambda_candidates)) {
  tryCatch({
    temp_fdPar <- fdPar(amt_basis, Lfdobj = 2, lambda = lambda_candidates[i])
    temp_smooth <- smooth.basis(week_points, t(amt_matrix[1:min(10, nrow(amt_matrix)), ]), temp_fdPar)
    gcv_scores[i] <- mean(temp_smooth$gcv)
  }, error = function(e) {
    gcv_scores[i] <- Inf
  })
}

# Select lambda with minimum GCV
optimal_lambda_amt <- lambda_candidates[which.min(gcv_scores)]
cat("  Selected lambda for amounts:", round(optimal_lambda_amt, 4), "\n")
cat("  GCV score:", round(min(gcv_scores), 4), "\n")

# Apply smoothing with optimal lambda
amt_fdPar <- fdPar(amt_basis, Lfdobj = 2, lambda = optimal_lambda_amt)
amt_smooth <- smooth.basis(week_points, t(amt_matrix), amt_fdPar)
amt_fd <- amt_smooth$fd
kme_amt_matrix <- t(eval.fd(week_grid, amt_fd))

cat("✓ KME amount predictor:", nrow(kme_amt_matrix), "×", ncol(kme_amt_matrix), "\n")

# ----------------------------------------------------------------------------
# 3D. KME TRANSFORMATION: B-Spline Smoothed Count Curves with Lambda Selection
# ----------------------------------------------------------------------------

cat("3D. Applying KME transformation with smoothing parameter selection to counts...\n")

count_matrix_raw <- weekly_features %>%
  dplyr::select(cc_num, week_num, n_trans) %>%
  pivot_wider(id_cols = cc_num, names_from = week_num, values_from = n_trans, values_fill = 0) %>%
  filter(cc_num %in% sample_cards)

for (w in week_points) {
  if (!as.character(w) %in% colnames(count_matrix_raw)) {
    count_matrix_raw[[as.character(w)]] <- 0
  }
}

count_matrix <- count_matrix_raw %>%
  dplyr::select(cc_num, all_of(as.character(week_points))) %>%
  column_to_rownames("cc_num") %>%
  as.matrix()

# Select optimal lambda for count data
cat("  Selecting optimal smoothing parameter (lambda) for counts...\n")
gcv_scores_count <- numeric(length(lambda_candidates))

for (i in seq_along(lambda_candidates)) {
  tryCatch({
    temp_fdPar <- fdPar(amt_basis, Lfdobj = 2, lambda = lambda_candidates[i])
    temp_smooth <- smooth.basis(week_points, t(count_matrix[1:min(10, nrow(count_matrix)), ]), temp_fdPar)
    gcv_scores_count[i] <- mean(temp_smooth$gcv)
  }, error = function(e) {
    gcv_scores_count[i] <- Inf
  })
}

optimal_lambda_count <- lambda_candidates[which.min(gcv_scores_count)]
cat("  Selected lambda for counts:", round(optimal_lambda_count, 4), "\n")
cat("  GCV score:", round(min(gcv_scores_count), 4), "\n")

# Apply smoothing with optimal lambda
count_fdPar <- fdPar(amt_basis, Lfdobj = 2, lambda = optimal_lambda_count)
count_smooth <- smooth.basis(week_points, t(count_matrix), count_fdPar)
count_fd <- count_smooth$fd
kme_count_matrix <- t(eval.fd(week_grid, count_fd))

cat("✓ KME count predictor:", nrow(kme_count_matrix), "×", ncol(kme_count_matrix), "\n\n")

# ============================================================================
# 4. EXTRACT SCALAR COVARIATES
# ============================================================================

cat("Step 4: Extracting scalar covariates...\n")

scalar_covariates <- data %>%
  filter(cc_num %in% sample_cards) %>%
  group_by(cc_num) %>%
  summarise(
    gender = first(gender),
    age = first(age),
    city_pop = first(city_pop),
    lat = mean(lat, na.rm = TRUE),
    long = mean(long, na.rm = TRUE),
    avg_trans_amt = mean(amt, na.rm = TRUE),
    total_trans = n(),
    n_categories = n_distinct(category),
    .groups = "drop"
  ) %>%
  mutate(
    gender_numeric = as.numeric(factor(gender)),
    log_city_pop = log(city_pop + 1),
    log_total_trans = log(total_trans + 1)
  )

cat("✓ Scalar covariates extracted:", ncol(scalar_covariates) - 1, "features\n\n")

# ============================================================================
# 5. ALIGN ALL DATA
# ============================================================================

cat("Step 5: Aligning all data...\n")

# Get common cards across all matrices
common_cards <- Reduce(intersect, list(
  as.character(fraud_response$cc_num),
  rownames(kde_amt_matrix),
  rownames(kde_count_matrix),
  rownames(kme_amt_matrix),
  rownames(kme_count_matrix),
  as.character(scalar_covariates$cc_num)
))

cat("Common cards across all data:", length(common_cards), "\n")

# Align all data
fraud_data <- fraud_response %>%
  filter(cc_num %in% as.numeric(common_cards)) %>%
  arrange(match(cc_num, as.numeric(common_cards)))

fraud_y <- fraud_data$has_fraud
fraud_y_rate <- fraud_data$fraud_rate
fraud_y_logit <- fraud_data$fraud_rate_logit

kde_amt_x <- kde_amt_matrix[common_cards, ]
kde_count_x <- kde_count_matrix[common_cards, ]
kme_amt_x <- kme_amt_matrix[common_cards, ]
kme_count_x <- kme_count_matrix[common_cards, ]

scalar_x <- scalar_covariates %>%
  filter(cc_num %in% as.numeric(common_cards)) %>%
  arrange(match(cc_num, as.numeric(common_cards))) %>%
  dplyr::select(gender_numeric, age, log_city_pop, lat, long, 
                avg_trans_amt, log_total_trans, n_categories) %>%
  as.matrix()

cat("✓ All data aligned\n")
cat("  Response: logit-transformed fraud rate for", length(fraud_y_logit), "cards\n")
cat("  Mean fraud rate:", round(mean(fraud_y_rate) * 100, 2), "%\n")
cat("  Mean logit-transformed:", round(mean(fraud_y_logit), 3), "\n\n")

# ============================================================================
# 6. CREATE FUNCTIONAL DATA OBJECTS AND PERFORM FPCA
# ============================================================================

cat("Step 6: Creating functional data objects and performing FPCA...\n")

n_basis_pred <- 12
pred_basis <- create.bspline.basis(rangeval = c(min(week_grid), max(week_grid)), 
                                   nbasis = n_basis_pred)
fdPar_pred <- fdPar(pred_basis, Lfdobj = 2, lambda = 0.1)

# Create FD objects
kde_amt_fd <- smooth.basis(week_grid, t(kde_amt_x), fdPar_pred)$fd
kde_count_fd <- smooth.basis(week_grid, t(kde_count_x), fdPar_pred)$fd
kme_amt_fd <- smooth.basis(week_grid, t(kme_amt_x), fdPar_pred)$fd
kme_count_fd <- smooth.basis(week_grid, t(kme_count_x), fdPar_pred)$fd

cat("✓ Functional data objects created\n\n")

# ============================================================================
# 7. TRAIN/TEST SPLIT
# ============================================================================

cat("Step 7: Splitting data into train/test sets...\n")

n_total <- length(common_cards)
train_prop <- 0.7
train_idx <- sample(1:n_total, floor(train_prop * n_total))
test_idx <- setdiff(1:n_total, train_idx)

cat("Training set:", length(train_idx), "cards\n")
cat("  Fraud rate:", round(mean(fraud_y_rate[train_idx]) * 100, 2), "%\n")
cat("  Mean logit:", round(mean(fraud_y_logit[train_idx]), 3), "\n")
cat("Test set:", length(test_idx), "cards\n")
cat("  Fraud rate:", round(mean(fraud_y_rate[test_idx]) * 100, 2), "%\n")
cat("  Mean logit:", round(mean(fraud_y_logit[test_idx]), 3), "\n\n")

# ============================================================================
# 8. MODEL 1: KDE-BASED SCALAR-ON-FUNCTION REGRESSION
# ============================================================================

cat(strrep("=", 70), "\n")
cat("MODEL 1: KDE-BASED SCALAR-ON-FUNCTION REGRESSION\n")
cat(strrep("=", 70), "\n\n")

cat("Functional Predictors: KDE-transformed amounts and counts\n")
cat("Scalar Covariates: Demographics and transaction statistics\n")
cat("Response: Logit-transformed fraud rate (continuous)\n\n")

# Functional PCA for dimension reduction
cat("Performing FPCA on predictors...\n")
kde_amt_fpca <- pca.fd(kde_amt_fd[train_idx], nharm = 5)
kde_count_fpca <- pca.fd(kde_count_fd[train_idx], nharm = 5)

cat("✓ FPCA completed\n")
cat("  Amount FPCA variance explained:", round(sum(kde_amt_fpca$varprop), 3), "\n")
cat("  Count FPCA variance explained:", round(sum(kde_count_fpca$varprop), 3), "\n\n")

# Get FPC scores for train and test
kde_amt_scores_train <- kde_amt_fpca$scores
kde_count_scores_train <- kde_count_fpca$scores
kde_amt_scores_test <- inprod(kde_amt_fd[test_idx], kde_amt_fpca$harmonics)
kde_count_scores_test <- inprod(kde_count_fd[test_idx], kde_count_fpca$harmonics)

# Create data frames with logit-transformed response
train_data_kde <- data.frame(
  fraud_logit = fraud_y_logit[train_idx],
  kde_amt_scores_train,
  kde_count_scores_train,
  scalar_x[train_idx, ]
)
colnames(train_data_kde) <- c("fraud_logit", 
                               paste0("amt_PC", 1:5), 
                               paste0("count_PC", 1:5),
                               colnames(scalar_x))

test_data_kde <- data.frame(
  fraud_logit = fraud_y_logit[test_idx],
  kde_amt_scores_test,
  kde_count_scores_test,
  scalar_x[test_idx, ]
)
colnames(test_data_kde) <- colnames(train_data_kde)

# Fit linear regression on logit-transformed response
cat("Fitting linear regression model on logit-transformed response...\n")
model_kde <- lm(fraud_logit ~ ., data = train_data_kde)

cat("\n✓ KDE Model Summary:\n")
print(summary(model_kde))

# Predictions (on logit scale)
pred_logit_kde_train <- predict(model_kde, newdata = train_data_kde)
pred_logit_kde_test <- predict(model_kde, newdata = test_data_kde)

# Back-transform to probability scale using inverse logit
inv_logit <- function(x) exp(x) / (1 + exp(x))
pred_prob_kde_train <- inv_logit(pred_logit_kde_train)
pred_prob_kde_test <- inv_logit(pred_logit_kde_test)

# Clip probabilities to valid range
pred_prob_kde_test <- pmax(0, pmin(1, pred_prob_kde_test))
pred_class_kde_test <- ifelse(pred_prob_kde_test > 0.5, 1, 0)

# Performance metrics
cat("\n✓ KDE Model Performance (Test Set):\n")
cat("  Logit scale MSE:", round(mean((pred_logit_kde_test - fraud_y_logit[test_idx])^2), 4), "\n")
cat("  Logit scale RMSE:", round(sqrt(mean((pred_logit_kde_test - fraud_y_logit[test_idx])^2)), 4), "\n")
cat("  Probability scale MSE:", round(mean((pred_prob_kde_test - fraud_y_rate[test_idx])^2), 6), "\n")
cat("  Probability scale MAE:", round(mean(abs(pred_prob_kde_test - fraud_y_rate[test_idx])), 6), "\n\n")

# Classification metrics
cat("Binary Classification Performance:\n")
conf_matrix_kde <- confusionMatrix(factor(pred_class_kde_test), 
                                   factor(fraud_y[test_idx]))
print(conf_matrix_kde)

# ROC and AUC
roc_kde <- roc(fraud_y[test_idx], pred_prob_kde_test)
auc_kde <- auc(roc_kde)
cat("\nAUC-ROC:", round(auc_kde, 4), "\n\n")

# ============================================================================
# 9. MODEL 2: KME-BASED SCALAR-ON-FUNCTION REGRESSION
# ============================================================================

cat(strrep("=", 70), "\n")
cat("MODEL 2: KME-BASED SCALAR-ON-FUNCTION REGRESSION\n")
cat(strrep("=", 70), "\n\n")

cat("Functional Predictors: KME-transformed (B-spline smoothed) amounts and counts\n")
cat("Scalar Covariates: Demographics and transaction statistics\n")
cat("Response: Fraud indicator (binary)\n\n")

# Functional PCA for KME predictors
cat("Performing FPCA on predictors...\n")
kme_amt_fpca <- pca.fd(kme_amt_fd[train_idx], nharm = 5)
kme_count_fpca <- pca.fd(kme_count_fd[train_idx], nharm = 5)

cat("✓ FPCA completed\n")
cat("  Amount FPCA variance explained:", round(sum(kme_amt_fpca$varprop), 3), "\n")
cat("  Count FPCA variance explained:", round(sum(kme_count_fpca$varprop), 3), "\n\n")

# Get FPC scores
kme_amt_scores_train <- kme_amt_fpca$scores
kme_count_scores_train <- kme_count_fpca$scores
kme_amt_scores_test <- inprod(kme_amt_fd[test_idx], kme_amt_fpca$harmonics)
kme_count_scores_test <- inprod(kme_count_fd[test_idx], kme_count_fpca$harmonics)

# Create data frames
train_data_kme <- data.frame(
  fraud = fraud_y[train_idx],
  kme_amt_scores_train,
  kme_count_scores_train,
  scalar_x[train_idx, ]
)
colnames(train_data_kme) <- c("fraud", 
                               paste0("amt_PC", 1:5), 
                               paste0("count_PC", 1:5),
                               colnames(scalar_x))

test_data_kme <- data.frame(
  fraud = fraud_y[test_idx],
  kme_amt_scores_test,
  kme_count_scores_test,
  scalar_x[test_idx, ]
)
colnames(test_data_kme) <- colnames(train_data_kme)

# Fit logistic regression
cat("Fitting logistic regression model...\n")
model_kme <- glm(fraud ~ ., data = train_data_kme, family = binomial)

cat("\n✓ KME Model Summary:\n")
print(summary(model_kme))

# Predictions
pred_prob_kme_train <- predict(model_kme, newdata = train_data_kme, type = "response")
pred_prob_kme_test <- predict(model_kme, newdata = test_data_kme, type = "response")
pred_class_kme_test <- ifelse(pred_prob_kme_test > 0.5, 1, 0)

# Performance metrics
cat("\n✓ KME Model Performance (Test Set):\n")
conf_matrix_kme <- confusionMatrix(factor(pred_class_kme_test), 
                                   factor(fraud_y[test_idx]))
print(conf_matrix_kme)

# ROC and AUC
roc_kme <- roc(fraud_y[test_idx], pred_prob_kme_test)
auc_kme <- auc(roc_kme)
cat("\nAUC-ROC:", round(auc_kme, 4), "\n\n")

# ============================================================================
# 10. MODEL COMPARISON
# ============================================================================

cat(strrep("=", 70), "\n")
cat("MODEL COMPARISON: KDE vs KME\n")
cat(strrep("=", 70), "\n\n")

comparison <- data.frame(
  Model = c("KDE", "KME"),
  AUC = c(auc_kde, auc_kme),
  Accuracy = c(conf_matrix_kde$overall["Accuracy"], 
               conf_matrix_kme$overall["Accuracy"]),
  Sensitivity = c(conf_matrix_kde$byClass["Sensitivity"],
                  conf_matrix_kme$byClass["Sensitivity"]),
  Specificity = c(conf_matrix_kde$byClass["Specificity"],
                  conf_matrix_kme$byClass["Specificity"])
) %>%
  mutate(
    Best_AUC = AUC == max(AUC),
    Best_Accuracy = Accuracy == max(Accuracy)
  )

print(comparison)

cat("\n✓ Model Recommendation:\n")
if (auc_kde > auc_kme) {
  cat("  → KDE approach performs better (AUC =", round(auc_kde, 4), ")\n")
  cat("  → KDE captures temporal density patterns more effectively\n")
} else if (auc_kme > auc_kde) {
  cat("  → KME approach performs better (AUC =", round(auc_kme, 4), ")\n")
  cat("  → Smoothed functional curves capture fraud patterns better\n")
} else {
  cat("  → Both approaches perform similarly\n")
}

# Save results
write_csv(comparison, "scalar_on_function_comparison.csv")
cat("\n✓ Results saved to 'scalar_on_function_comparison.csv'\n\n")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

cat(strrep("=", 70), "\n")
cat("GENERATING VISUALIZATIONS\n")
cat(strrep("=", 70), "\n\n")

if (!dir.exists("plots")) dir.create("plots")

# Plot 1: ROC Curves Comparison
roc_data <- rbind(
  data.frame(
    FPR = 1 - roc_kde$specificities,
    TPR = roc_kde$sensitivities,
    Model = "KDE"
  ),
  data.frame(
    FPR = 1 - roc_kme$specificities,
    TPR = roc_kme$sensitivities,
    Model = "KME"
  )
)

p1 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  scale_color_manual(values = c("KDE" = "steelblue", "KME" = "coral")) +
  labs(title = "ROC Curves: KDE vs KME Scalar-on-Function Models",
       subtitle = paste0("KDE AUC = ", round(auc_kde, 4), 
                        ", KME AUC = ", round(auc_kme, 4)),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/24_scalar_on_function_roc.png", p1, width = 10, height = 8)

# Plot 2: Model performance comparison
p2 <- comparison %>%
  dplyr::select(Model, AUC, Accuracy, Sensitivity, Specificity) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("KDE" = "steelblue", "KME" = "coral")) +
  labs(title = "Scalar-on-Function Model Performance Comparison",
       subtitle = "Higher values indicate better performance",
       y = "Value") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/25_scalar_on_function_performance.png", p2, width = 10, height = 6)

# Plot 3: Predicted probabilities distribution
pred_data <- rbind(
  data.frame(
    Probability = pred_prob_kde_test,
    Actual = factor(fraud_y[test_idx], labels = c("No Fraud", "Fraud")),
    Model = "KDE"
  ),
  data.frame(
    Probability = pred_prob_kme_test,
    Actual = factor(fraud_y[test_idx], labels = c("No Fraud", "Fraud")),
    Model = "KME"
  )
)

p3 <- ggplot(pred_data, aes(x = Probability, fill = Actual)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(~ Model, ncol = 1) +
  scale_fill_manual(values = c("No Fraud" = "steelblue", "Fraud" = "red")) +
  labs(title = "Predicted Fraud Probability Distribution",
       subtitle = "Test set predictions by actual fraud status",
       x = "Predicted Probability of Fraud",
       y = "Count") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/26_probability_distribution.png", p3, width = 10, height = 8)

cat("✓ Visualizations saved\n\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

cat(strrep("=", 70), "\n")
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 70), "\n\n")

cat("METHODOLOGY:\n")
cat("• Response: Fraud indicator per card (scalar, binary)\n")
cat("• Functional Predictors:\n")
cat("  - Transaction amounts (KDE and KME transformed)\n")
cat("  - Transaction counts (KDE and KME transformed)\n")
cat("• Scalar Covariates: Demographics and transaction statistics\n")
cat("• Model: Scalar-on-function regression (logistic)\n\n")

cat("BANDWIDTH/SMOOTHING PARAMETER SELECTION:\n")
cat("• KDE bandwidth method:", best_bw_method, "\n")
cat("• KME lambda (amounts):", round(optimal_lambda_amt, 4), "\n")
cat("• KME lambda (counts):", round(optimal_lambda_count, 4), "\n\n")

cat("RESULTS:\n")
cat("• Cards analyzed:", n_total, "\n")
cat("• Training set:", length(train_idx), "cards\n")
cat("• Test set:", length(test_idx), "cards\n")
cat("• KDE Model AUC:", round(auc_kde, 4), "\n")
cat("• KME Model AUC:", round(auc_kme, 4), "\n\n")

cat("OUTPUTS:\n")
cat("✓ scalar_on_function_comparison.csv\n")
cat("✓ plots/24_scalar_on_function_roc.png\n")
cat("✓ plots/25_scalar_on_function_performance.png\n")
cat("✓ plots/26_probability_distribution.png\n\n")

cat("This FDA framework demonstrates:\n")
cat("• How functional transaction patterns predict fraud (scalar outcome)\n")
cat("• The effectiveness of KDE vs KME transformations\n")
cat("• Integration of functional and scalar predictors\n")
cat("• Binary classification using functional data\n\n")

cat(strrep("=", 70), "\n")

