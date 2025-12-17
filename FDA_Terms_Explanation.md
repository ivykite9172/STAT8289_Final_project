# Incorporating FDA Terms in Fraud Detection Model

## Overview of FDA in This Context

Functional Data Analysis (FDA) treats transaction patterns over time as **continuous functions** rather than discrete observations. This allows us to capture the **shape, velocity, and acceleration** of spending patterns.

---

## Step-by-Step FDA Integration

### **STEP 1: Raw Data → Functional Data**

**What we have:** Discrete weekly transactions
```
Week 0: $120, Week 1: $0, Week 2: $89, Week 3: $156, ...
```

**What we create:** Smooth continuous curves
```
f(t) = smooth function representing spending over time t
```

**How we do it:**
```r
# 1. Aggregate to weekly grid
weekly_features <- data %>%
  group_by(cc_num, week_num) %>%
  summarise(total_amt = sum(amt))

# 2. Create B-spline basis
n_basis <- 15
basis <- create.bspline.basis(rangeval = c(0, max_week), nbasis = n_basis)

# 3. Smooth the discrete data
fdPar_obj <- fdPar(basis, Lfdobj = 2, lambda = 10)
smooth_obj <- smooth.basis(week_points, t(amt_matrix), fdPar_obj)
fd_obj <- smooth_obj$fd  # This is the functional data object
```

**Key FDA concept:** B-splines convert discrete points into smooth curves through **basis expansion**:
```
f(t) = c₁·B₁(t) + c₂·B₂(t) + ... + c₁₅·B₁₅(t)
```
where B₁, B₂, ..., B₁₅ are basis functions and c₁, c₂, ..., c₁₅ are coefficients.

---

### **STEP 2: Extract Derivatives (FDA Feature)**

**First Derivative (Velocity)** - Rate of spending change:
```r
deriv1 <- eval.fd(week_grid, fd_obj, Lfdobj = 1)
# deriv1[t] = df/dt = how fast spending is changing
```

**Second Derivative (Acceleration)** - Change in the rate:
```r
deriv2 <- eval.fd(week_grid, fd_obj, Lfdobj = 2)
# deriv2[t] = d²f/dt² = how spending velocity is changing
```

**Why this matters for fraud:**
- **Sudden spikes**: High velocity (|df/dt| large)
- **Pattern changes**: High acceleration (|d²f/dt²| large)
- **Smooth normal patterns**: Low derivatives

**Example:**
```
Normal:  [100, 105, 110, 115, 120] → smooth, low derivatives
Fraud:   [100, 105, 500, 110, 115] → spike, high derivatives
```

---

### **STEP 3: Functional Principal Components Analysis (FPCA)**

**Problem:** Each curve has 52 time points = 52 numbers → too many features

**Solution:** FPCA reduces to key "modes of variation"

```r
# Perform FPCA
fpca <- pca.fd(fd_obj, nharm = 10)

# Extract principal component scores
fpc_scores <- fpca$scores  # Shape: n_cards × 10
```

**What FPCA does:**
```
Original: Each card = 52-dimensional curve
After FPCA: Each card = 10 FPC scores

PC1: Captures "overall level" (high/low spender)
PC2: Captures "trend" (increasing/decreasing)
PC3: Captures "volatility" (stable/variable)
...
PC10: Captures specific patterns
```

**Mathematical interpretation:**
```
f_i(t) ≈ μ(t) + ξᵢ₁·φ₁(t) + ξᵢ₂·φ₂(t) + ... + ξᵢ₁₀·φ₁₀(t)

where:
- f_i(t) = curve for card i
- μ(t) = mean curve across all cards
- φ₁(t), φ₂(t), ... = principal component functions (eigenfunctions)
- ξᵢ₁, ξᵢ₂, ... = FPC SCORES for card i (these are our features!)
```

**What we use in the model:** The FPC scores (ξᵢ₁, ξᵢ₂, ..., ξᵢ₁₀)

---

### **STEP 4: Multiple Functional Variables**

We create functional data for **multiple transaction characteristics**:

**1. Amount Function:** f_amt(t)
```r
amt_functional <- create_functional_data("total_amt")
amt_fpca <- pca.fd(amt_fd, nharm = 10)
amt_scores <- amt_fpca$scores  # 10 features
```

**2. Count Function:** f_count(t)
```r
count_functional <- create_functional_data("n_trans")
count_fpca <- pca.fd(count_fd, nharm = 10)
count_scores <- count_fpca$scores  # 10 features
```

**3. Derivative Functions:** f'_amt(t), f''_amt(t)
```r
deriv1_fpca <- pca.fd(deriv1_fd, nharm = 10)
deriv1_scores <- deriv1_fpca$scores  # 10 features
```

**Total FDA features:** 10 + 10 + 10 = **30 functional features**

---

### **STEP 5: Combining FDA with Scalar Features**

**Final feature matrix:**
```r
X_combined <- cbind(
  amt_scores,        # 10 FDA features (amount patterns)
  count_scores,      # 10 FDA features (frequency patterns)
  deriv1_scores,     # 10 FDA features (velocity patterns)
  scalar_features    # 40+ scalar features (demographics, etc.)
)
# Total: ~70+ features
```

**This goes into the model:**
```r
# Random Forest
rf_model <- randomForest(fraud ~ ., data = data.frame(fraud = y, X_combined))

# The model now uses:
# - Functional patterns (via FPC scores)
# - Scalar characteristics
# - Their interactions
```

---

## Visual Example: How FDA Helps

### **Card A (Normal)**
```
Week:     0   10   20   30   40   50
Amount:  100  105  110  108  112  115
Curve:   ————————————— (smooth, flat)
f'(t):   ————— (low velocity)
f''(t):  ————— (low acceleration)
FPC1:    0.2  (low spending)
FPC2:   -0.1  (stable trend)
→ Prediction: NOT FRAUD
```

### **Card B (Fraud)**
```
Week:     0   10   20   30   40   50
Amount:  100  105  800  850  110  115
Curve:   ————⎺⎺⎺⎺————— (spike at week 20-30)
f'(t):   ————↑↑↓↓———— (high velocity)
f''(t):  ————⎺⎺⎺⎺————— (high acceleration)
FPC1:    2.5  (high spending)
FPC2:    3.1  (sudden change)
FPC3:   -2.8  (volatility)
→ Prediction: FRAUD
```

---

## Key FDA Techniques in Our Model

### 1. **B-Spline Smoothing**
**Purpose:** Convert discrete points to smooth curves
**Parameter:** λ (lambda) = smoothing strength
- λ small → follows data closely (may overfit)
- λ large → smoother curves (may underfit)
**Our choice:** λ = 10 (selected via GCV)

### 2. **Functional PCA (FPCA)**
**Purpose:** Dimensionality reduction for functions
**Output:** Principal component scores
**Benefits:**
- Reduces 52 time points → 10 scores
- Captures main patterns
- Removes noise
- Orthogonal features (no multicollinearity)

### 3. **Functional Derivatives**
**Purpose:** Capture rate of change
**Types:**
- f'(t): Velocity (1st derivative)
- f''(t): Acceleration (2nd derivative)
**Why important:** Fraud often shows sudden changes

### 4. **Bandwidth Selection**
**For KDE:** Sheather-Jones method
**For smoothing:** GCV (Generalized Cross-Validation)
**Purpose:** Optimal bias-variance tradeoff

---

## How to Interpret FDA Features in Model

### **FPC Scores as Features:**

```r
# After FPCA
amt_PC1  # Overall spending level
amt_PC2  # Linear trend (increase/decrease)
amt_PC3  # Curvature (U-shaped, inverted-U)
amt_PC4  # Oscillation (cyclical patterns)
...

# These become features in:
glm(fraud ~ amt_PC1 + amt_PC2 + amt_PC3 + ... + scalars)
```

### **Model Interpretation:**

```
Logit(P(fraud)) = β₀ + β₁·amt_PC1 + β₂·amt_PC2 + ... + β₃₀·deriv_PC10 + ...

Example coefficients:
β₁ = 0.5  → Higher overall spending → Higher fraud risk
β₂ = 0.8  → Increasing trend → Higher fraud risk
β₂₁ = 2.1 → High velocity → Much higher fraud risk
```

---

## Practical Example: Full FDA Pipeline

```r
# 1. Start with raw transactions
transactions <- data.frame(
  week = c(0, 1, 2, 3, 4, 5),
  amount = c(100, 120, 500, 480, 110, 115)
)

# 2. Create functional data
basis <- create.bspline.basis(rangeval = c(0, 5), nbasis = 5)
fdPar <- fdPar(basis, Lfdobj = 2, lambda = 10)
fd_obj <- smooth.basis(0:5, transactions$amount, fdPar)$fd

# 3. Evaluate at fine grid
t_grid <- seq(0, 5, length.out = 50)
f_t <- eval.fd(t_grid, fd_obj)  # Smooth curve

# 4. Get derivatives
f_prime <- eval.fd(t_grid, fd_obj, Lfdobj = 1)  # Velocity
f_double_prime <- eval.fd(t_grid, fd_obj, Lfdobj = 2)  # Acceleration

# 5. Perform FPCA (on many cards)
fpca <- pca.fd(all_cards_fd, nharm = 10)
scores <- fpca$scores  # These are the FDA features!

# 6. Use in model
model <- glm(fraud ~ scores[,1] + scores[,2] + ... + scalars, family = binomial)
```

---

## Advantages of FDA Approach

### **vs Traditional Approach:**

**Traditional:**
```
Features: week1_amt, week2_amt, ..., week52_amt
Problems:
- 52 correlated features
- High dimensionality
- Treats weeks independently
- Ignores smooth patterns
```

**FDA:**
```
Features: PC1, PC2, ..., PC10, derivatives
Benefits:
- Only 30 uncorrelated features
- Captures continuous patterns
- Includes rate of change
- Noise reduction through smoothing
```

---

## Summary

**FDA Integration Pipeline:**
```
Raw Transactions
    ↓ (aggregation)
Weekly Discrete Data
    ↓ (B-spline smoothing)
Smooth Functional Curves
    ↓ (derivatives)
Velocity & Acceleration Curves
    ↓ (FPCA)
Principal Component Scores (30 features)
    ↓ (combine with scalars)
Full Feature Matrix (70+ features)
    ↓ (ensemble models)
Fraud Prediction
```

**Key FDA contributions to model:**
1. **30 functional features** from transaction patterns
2. **Smooth representation** reduces noise
3. **Derivatives** capture sudden changes (fraud indicators)
4. **FPCA** reduces dimensionality while preserving patterns
5. **Multiple functional variables** capture different aspects

**Result:** Rich feature representation that captures temporal patterns effectively! 🎯

