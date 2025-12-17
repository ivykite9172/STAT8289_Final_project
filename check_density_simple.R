# ============================================================================
# Simple Transaction Density Analysis
# ============================================================================

library(tidyverse)
library(lubridate)

cat("Loading and analyzing transaction density...\n\n")

data <- read_csv("credit_card_transactions.csv", show_col_types = FALSE)

# Clean data
data <- data %>%
  mutate(
    trans_datetime = ymd_hms(trans_date_trans_time),
    is_fraud = as.numeric(is_fraud)
  ) %>%
  filter(!is.na(trans_datetime))  # Remove failed parses

cat("✓ Data loaded:", nrow(data), "transactions\n")
cat("✓ Cards:", n_distinct(data$cc_num), "\n\n")

# ==========================
# DENSITY ANALYSIS
# ==========================

cat(strrep("=", 70), "\n")
cat("TRANSACTION DENSITY PER CARD\n")
cat(strrep("=", 70), "\n\n")

density_stats <- data %>%
  group_by(cc_num) %>%
  summarise(
    n_trans = n(),
    span_days = as.numeric(difftime(max(trans_datetime), min(trans_datetime), units = "days")),
    has_fraud = max(is_fraud)
  ) %>%
  mutate(
    trans_per_day = n_trans / (span_days + 1),
    trans_per_week = trans_per_day * 7
  )

cat("TRANSACTIONS PER CARD:\n")
print(summary(density_stats$n_trans))

cat("\nTRANSACTIONS PER WEEK:\n")
print(summary(density_stats$trans_per_week))

cat("\nRECOMMENDATION:\n")
median_tpw <- median(density_stats$trans_per_week, na.rm = TRUE)
cat("Median transactions/week:", round(median_tpw, 1), "\n")

if (median_tpw >= 10) {
  cat("→ WEEKLY GRID recommended (52 points)\n")
  cat("  Sufficient density for weekly analysis\n")
} else if (median_tpw >= 5) {
  cat("→ WEEKLY or BI-WEEKLY GRID (26-52 points)\n")
  cat("  Moderate density\n")
} else {
  cat("→ BI-WEEKLY or MONTHLY GRID (12-26 points)\n")
  cat("  Low density - use coarser grid\n")
}

cat("\n✓ Analysis complete\n")

