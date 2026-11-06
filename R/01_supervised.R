# ==============================================================================
# Predicting Mental Wellness: A Regression Analysis
# Supervised learning: OLS, Ridge, Lasso, Random Forest, XGBoost
#
# Author : Mohamed Amine Laabidi
# Course : Statistical Learning — MSc Data Science for Economics and Health,
#          Università degli Studi di Milano (April 2026)
# Data   : "Screen Time vs Mental Wellness Survey – 2025" (Kaggle, n = 400)
#
# Run from the repository root:  source("R/01_supervised.R")
# Figures used in the README are written to figures/.
# ==============================================================================


# ── 1. LOAD LIBRARIES & DATA ──────────────────────────────────────────────────
library(tidyverse)
library(corrplot)
library(car)          # leveneTest
library(lmtest)       # resettest
library(caret)
library(glmnet)
library(randomForest) # tuneRF
library(ranger)       # random forest engine used by caret
library(xgboost)
library(effectsize)   # eta_squared
library(e1071)        # skewness
library(gridExtra)

dir.create("figures", showWarnings = FALSE)

df_raw <- read.csv("data/ScreenTime_vs_MentalWellness.csv")
df_raw$X <- NULL      # empty column created by a trailing comma in the CSV

dim(df_raw)
str(df_raw)
summary(df_raw)


# ── 2. EXPLORATORY DATA ANALYSIS (EDA) ───────────────────────────────────────

# ── 2.1 Univariate Distributions ─────────────────────────────────────────────
df_raw %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Univariate Distributions — Numeric Variables")

ggplot(df_raw, aes(x = gender))     + geom_bar(fill = "steelblue")  + labs(title = "Gender Distribution")
ggplot(df_raw, aes(x = work_mode))  + geom_bar(fill = "coral")      + labs(title = "Work Mode Distribution")
ggplot(df_raw, aes(x = occupation)) + geom_bar(fill = "seagreen")   + labs(title = "Occupation Distribution")


# ── 2.2 Target Variable Distribution ─────────────────────────────────────────
target <- df_raw$mental_wellness_index_0_100
cat("=== mental_wellness_index_0_100 ===\n")
cat("Min:     ", min(target), "\n")
cat("Max:     ", max(target), "\n")
cat("Mean:    ", round(mean(target), 2), "\n")
cat("Median:  ", median(target), "\n")
cat("SD:      ", round(sd(target), 2), "\n")
cat("Skewness:", round(skewness(target), 3), "\n")
cat("Share of exact zeros:", round(mean(target == 0) * 100, 1), "%\n")

ggplot(df_raw, aes(x = mental_wellness_index_0_100)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_density(color = "red", linewidth = 1.2) +
  geom_vline(xintercept = mean(target),   color = "orange",    linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = median(target), color = "darkgreen", linetype = "dashed", linewidth = 1) +
  annotate("text", x = mean(target) + 3,   y = 0.03, label = "Mean",   color = "orange",    size = 3.5) +
  annotate("text", x = median(target) - 3, y = 0.03, label = "Median", color = "darkgreen", size = 3.5) +
  labs(title = "Distribution of Mental Wellness Index",
       subtitle = "Target variable — continuous scale 0 to 100",
       x = "Mental Wellness Index (0–100)", y = "Density") +
  theme_minimal(base_size = 12)

ggplot(df_raw, aes(y = mental_wellness_index_0_100)) +
  geom_boxplot(fill = "steelblue", alpha = 0.7, width = 0.3) +
  labs(title = "Boxplot — Mental Wellness Index", y = "Mental Wellness Index (0–100)") +
  theme_minimal(base_size = 12)

shapiro.test(target)


# ── 2.3 Sleep Quality Distribution ───────────────────────────────────────────
table(df_raw$sleep_quality_1_5)
pct_q1 <- round(mean(df_raw$sleep_quality_1_5 == 1) * 100, 1)

ggplot(df_raw, aes(x = factor(sleep_quality_1_5))) +
  geom_bar(fill = "steelblue", color = "white", width = 0.6) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 4) +
  labs(title = "Distribution of Sleep Quality Ratings",
       subtitle = paste0(pct_q1, "% of participants rated sleep quality as 1 out of 5"),
       x = "Sleep Quality (1 = Poor, 5 = Excellent)", y = "Number of Participants",
       caption = "Recoded to a binary indicator (1 = poor sleep) before modelling") +
  theme_minimal(base_size = 12) +
  theme(plot.subtitle = element_text(color = "grey50"))


# ── 2.4 Correlation Matrix ────────────────────────────────────────────────────
num_vars   <- df_raw[, sapply(df_raw, is.numeric)]
cor_matrix <- cor(num_vars, use = "complete.obs")

corrplot(cor_matrix,
         method      = "color",
         type        = "upper",
         addCoef.col = "black",
         number.cex  = 0.65,
         tl.cex      = 0.75,
         tl.col      = "black",
         tl.srt      = 45,
         diag        = FALSE,
         mar         = c(0, 0, 2, 0))

round(cor_matrix[, "mental_wellness_index_0_100"], 2)


# ── 2.5 Predictors vs Target ──────────────────────────────────────────────────
vars <- c("work_screen_hours", "leisure_screen_hours", "sleep_hours",
          "stress_level_0_10", "exercise_minutes_per_week", "social_hours_per_week")

for (v in vars) {
  p <- ggplot(df_raw, aes(x = .data[[v]], y = mental_wellness_index_0_100)) +
    geom_point(alpha = 0.3, color = "steelblue") +
    geom_smooth(method = "lm", formula = y ~ x, color = "red", se = TRUE) +
    labs(title = paste(v, "vs Mental Wellness"), x = v)
  print(p)
}

ggplot(df_raw, aes(x = work_mode,  y = mental_wellness_index_0_100, fill = work_mode))  + geom_boxplot() + labs(title = "Mental Wellness by Work Mode")
ggplot(df_raw, aes(x = gender,     y = mental_wellness_index_0_100, fill = gender))     + geom_boxplot() + labs(title = "Mental Wellness by Gender")
ggplot(df_raw, aes(x = occupation, y = mental_wellness_index_0_100, fill = occupation)) + geom_boxplot() + labs(title = "Mental Wellness by Occupation")


# ── 2.6 Occupation & Work Mode Analysis ──────────────────────────────────────
ggplot(df_raw, aes(x = occupation, y = work_screen_hours, fill = occupation)) +
  geom_boxplot() + labs(title = "Work Screen Hours by Occupation")

ggplot(df_raw, aes(x = occupation, y = leisure_screen_hours, fill = occupation)) +
  geom_boxplot() + labs(title = "Leisure Screen Hours by Occupation")

summary(aov(work_screen_hours    ~ occupation, data = df_raw))
summary(aov(leisure_screen_hours ~ occupation, data = df_raw))
TukeyHSD(aov(work_screen_hours    ~ occupation, data = df_raw))
TukeyHSD(aov(leisure_screen_hours ~ occupation, data = df_raw))

df_raw %>%
  group_by(occupation) %>%
  summarise(
    shapiro_leisure = shapiro.test(leisure_screen_hours)$p.value,
    shapiro_work    = shapiro.test(work_screen_hours)$p.value
  )

leveneTest(leisure_screen_hours ~ occupation, data = df_raw)
leveneTest(work_screen_hours    ~ occupation, data = df_raw)

df_raw %>%
  group_by(work_mode) %>%
  summarise(shapiro_p = shapiro.test(screen_time_hours)$p.value)

leveneTest(screen_time_hours ~ work_mode, data = df_raw)
summary(aov(screen_time_hours ~ work_mode, data = df_raw))
oneway.test(screen_time_hours ~ work_mode, data = df_raw, var.equal = FALSE)
TukeyHSD(aov(screen_time_hours ~ work_mode, data = df_raw))

ggplot(df_raw, aes(x = work_mode, y = screen_time_hours, fill = work_mode)) +
  geom_boxplot() +
  labs(title = "Total Screen Time by Work Mode", x = "Work Mode", y = "Total Screen Time (hours)")

df_raw %>%
  group_by(work_mode) %>%
  summarise(shapiro_p = shapiro.test(productivity_0_100)$p.value)

leveneTest(productivity_0_100 ~ work_mode, data = df_raw)

aov_prod <- aov(productivity_0_100 ~ work_mode, data = df_raw)
summary(aov_prod)
TukeyHSD(aov_prod)
eta_squared(aov_prod)

ggplot(df_raw, aes(x = work_mode, y = productivity_0_100, fill = work_mode)) +
  geom_boxplot() +
  labs(title = "Productivity by Work Mode", x = "Work Mode", y = "Productivity Score (0–100)")


# ── 3. NON-LINEARITY ASSESSMENT ───────────────────────────────────────────────
predictors <- c("age", "sleep_hours", "work_screen_hours",
                "social_hours_per_week", "leisure_screen_hours",
                "stress_level_0_10", "exercise_minutes_per_week", "sleep_quality_1_5")

resid_plot <- function(var, df) {
  formula   <- as.formula(paste("mental_wellness_index_0_100 ~", var))
  model     <- lm(formula, data = df)
  plot_data <- data.frame(fitted = fitted(model), residuals = residuals(model))

  ggplot(plot_data, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.4, color = "steelblue", size = 1.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE, color = "orange", linewidth = 1) +
    labs(title = paste("Residuals vs Fitted —", var), x = "Fitted Values", y = "Residuals") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(size = 9, face = "bold"))
}

plots <- lapply(predictors, resid_plot, df = df_raw)
grid.arrange(grobs = plots, ncol = 3)

cat_resid_plot <- function(var, df) {
  formula   <- as.formula(paste("mental_wellness_index_0_100 ~", var))
  model     <- lm(formula, data = df)
  plot_data <- data.frame(category = df[[var]], residuals = residuals(model))

  ggplot(plot_data, aes(x = category, y = residuals, fill = category)) +
    geom_boxplot(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(title = paste("Residuals by", var), x = var, y = "Residuals") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none")
}

cat_plots <- lapply(c("gender", "occupation", "work_mode"), cat_resid_plot, df = df_raw)
grid.arrange(grobs = cat_plots, ncol = 3)

# Ramsey RESET test for the leisure-screen relationship
model_leisure <- lm(mental_wellness_index_0_100 ~ leisure_screen_hours, data = df_raw)
resettest(model_leisure, power = 2:3, type = "fitted")


# ── 4. DATA CLEANING & PREPROCESSING ─────────────────────────────────────────
colSums(is.na(df_raw))        # no missing values
sum(duplicated(df_raw))       # no duplicate rows

# Outlier screening (|z| > 3) — reported, not removed
num_vars    <- df_raw[, sapply(df_raw, is.numeric)]
z_scores    <- scale(num_vars)
outlier_idx <- which(abs(z_scores) > 3, arr.ind = TRUE)

outlier_report <- data.frame(
  row      = outlier_idx[, 1],
  variable = colnames(num_vars)[outlier_idx[, 2]],
  value    = num_vars[outlier_idx],
  z_score  = z_scores[outlier_idx]
)
outlier_report <- outlier_report[order(abs(outlier_report$z_score), decreasing = TRUE), ]
print(outlier_report)

# Sleep quality: 69% of respondents answered 1, only 3 answered 4 and none 5.
# Collapsed into a binary indicator: 1 = poor sleep (score 1), 0 = score >= 2.
df_raw$sleep_quality_binary <- ifelse(df_raw$sleep_quality_1_5 == 1, 1, 0)
table(df_raw$sleep_quality_binary)

# Dropped: ID, productivity (r = 0.90 with target -> near-leakage),
# total screen time (= work + leisure) and the original sleep-quality score.
df_clean <- df_raw %>%
  dplyr::select(-user_id, -productivity_0_100, -screen_time_hours, -sleep_quality_1_5)

df_clean$gender     <- as.factor(df_clean$gender)
df_clean$work_mode  <- as.factor(df_clean$work_mode)
df_clean$occupation <- as.factor(df_clean$occupation)

cat("Remaining variables:", ncol(df_clean), "\n")
names(df_clean)

set.seed(42)
train_idx <- createDataPartition(df_clean$mental_wellness_index_0_100, p = 0.8, list = FALSE)
train <- df_clean[train_idx, ]
test  <- df_clean[-train_idx, ]

cat("Train size:", nrow(train), "\nTest size:", nrow(test), "\n")   # 321 / 79

# Dummy-coded design matrices (used by XGBoost; caret builds its own for the others)
X_train <- model.matrix(mental_wellness_index_0_100 ~ . - 1, data = train)
X_test  <- model.matrix(mental_wellness_index_0_100 ~ . - 1, data = test)
y_train <- train$mental_wellness_index_0_100
y_test  <- test$mental_wellness_index_0_100


# ── 5. CROSS-VALIDATION CONTROL ───────────────────────────────────────────────
# Note: models are fitted in sequence after set.seed(42), so the whole script is
# reproducible when run top-to-bottom in a fresh session.
set.seed(42)
cv_control <- trainControl(
  method          = "repeatedcv",
  number          = 10,
  repeats         = 3,
  verboseIter     = FALSE,
  savePredictions = "final"
)


# ── 6. MODELS ─────────────────────────────────────────────────────────────────

# ── 6.1 OLS ──────────────────────────────────────────────────────────────────
model_ols <- train(mental_wellness_index_0_100 ~ ., data = train,
                   method = "lm", trControl = cv_control)
print(model_ols)

# Coefficients in original units (e.g. effect of +1 point of stress)
summary(model_ols$finalModel)


# ── 6.2 Ridge ────────────────────────────────────────────────────────────────
ridge_grid <- expand.grid(alpha = 0, lambda = 10^seq(-4, 2, length = 100))

model_ridge <- train(mental_wellness_index_0_100 ~ ., data = train,
                     method     = "glmnet",
                     trControl  = cv_control,
                     tuneGrid   = ridge_grid,
                     preProcess = c("center", "scale"))

print(model_ridge$bestTune)


# ── 6.3 Lasso ────────────────────────────────────────────────────────────────
# caret may warn about "missing values in resampled performance measures":
# for the largest lambdas Lasso drops every predictor, predictions are constant
# and R-squared is undefined. This does not affect the RMSE-based selection.
lasso_grid <- expand.grid(alpha = 1, lambda = 10^seq(-4, 2, length = 100))

model_lasso <- train(mental_wellness_index_0_100 ~ ., data = train,
                     method     = "glmnet",
                     trControl  = cv_control,
                     tuneGrid   = lasso_grid,
                     preProcess = c("center", "scale"))

print(model_lasso$bestTune)

# Predictors were standardised, so these are effects of a 1-SD change
coef(model_lasso$finalModel, s = model_lasso$bestTune$lambda)

pred_lasso_test <- predict(model_lasso, newdata = test)
ss_res <- sum((y_test - pred_lasso_test)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
cat("Lasso Test R²:", round(1 - ss_res / ss_tot, 4), "\n")


# ── 6.4 Random Forest ────────────────────────────────────────────────────────
set.seed(42)
tuneRF_result <- tuneRF(
  x          = train[, names(train) != "mental_wellness_index_0_100"],
  y          = train$mental_wellness_index_0_100,
  stepFactor = 1.5,
  improve    = 0.01,
  ntreeTry   = 300,
  trace      = TRUE,
  plot       = TRUE
)
best_mtry <- tuneRF_result[which.min(tuneRF_result[, "OOBError"]), "mtry"]

rf_grid <- expand.grid(
  mtry          = unique(c(2, 3, best_mtry, best_mtry + 1)),
  splitrule     = "variance",
  min.node.size = c(1, 3, 5, 10)
)

model_rf <- train(mental_wellness_index_0_100 ~ ., data = train,
                  method     = "ranger",
                  trControl  = cv_control,
                  tuneGrid   = rf_grid,
                  num.trees  = 500,
                  importance = "impurity")

print(model_rf$bestTune)


# ── 6.5 XGBoost ──────────────────────────────────────────────────────────────
# nrounds is chosen with xgb.cv (plain 10-fold CV). Exact results can differ
# slightly between xgboost package versions.
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

params <- list(
  objective        = "reg:squarederror",
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)

set.seed(42)
cv_xgb <- xgb.cv(params = params, data = dtrain,
                 nrounds = 300, nfold = 10, metrics = "rmse", verbose = 0)

best_nrounds <- which.min(cv_xgb$evaluation_log$test_rmse_mean)
best_cv_rmse <- min(cv_xgb$evaluation_log$test_rmse_mean)
cat("Best nrounds:", best_nrounds, "\nBest CV RMSE:", best_cv_rmse, "\n")

model_xgb <- xgb.train(params = params, data = dtrain,
                       nrounds = best_nrounds, verbose = 0)

pred_xgb <- predict(model_xgb, dtest)
cat("XGBoost Test RMSE:", round(RMSE(pred_xgb, y_test), 4), "\n")


# ── 7. FINAL EVALUATION ───────────────────────────────────────────────────────
pred_ols   <- predict(model_ols,   newdata = test)
pred_ridge <- predict(model_ridge, newdata = test)
pred_lasso <- predict(model_lasso, newdata = test)
pred_rf    <- predict(model_rf,    newdata = test)

evaluate_model <- function(actual, predicted, model_name) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  data.frame(
    Model = model_name,
    R2    = round(1 - ss_res / ss_tot, 4),
    RMSE  = round(sqrt(mean((actual - predicted)^2)), 4),
    MAE   = round(mean(abs(actual - predicted)), 4)
  )
}

results <- rbind(
  evaluate_model(y_test, pred_ols,              "OLS"),
  evaluate_model(y_test, as.vector(pred_ridge), "Ridge"),
  evaluate_model(y_test, as.vector(pred_lasso), "Lasso"),
  evaluate_model(y_test, pred_rf,               "Random Forest"),
  evaluate_model(y_test, pred_xgb,              "XGBoost")
)

# Delta = Test RMSE - CV RMSE. Positive: test error higher than CV predicted.
# With only 79 test observations, deltas of this size are within sampling noise.
results$CV_RMSE <- round(c(
  min(model_ols$results$RMSE),
  min(model_ridge$results$RMSE),
  min(model_lasso$results$RMSE),
  min(model_rf$results$RMSE),
  best_cv_rmse
), 4)
results$Delta <- round(results$RMSE - results$CV_RMSE, 4)
print(results)
write.csv(results, "figures/model_results.csv", row.names = FALSE)

# Actual vs predicted (saved for the README)
plot_avp <- function() {
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
  preds  <- list(OLS = pred_ols, Ridge = pred_ridge, Lasso = pred_lasso,
                 `Random Forest` = pred_rf, XGBoost = pred_xgb)
  cols   <- c("steelblue", "seagreen", "coral", "purple", "orange")
  for (i in seq_along(preds)) {
    plot(y_test, preds[[i]], main = names(preds)[i], xlab = "Actual", ylab = "Predicted",
         pch = 19, col = cols[i], xlim = c(0, 100), ylim = c(-10, 100))
    abline(0, 1, col = "red", lwd = 2)
  }
}
plot_avp()
png("figures/actual_vs_predicted.png", width = 1500, height = 950, res = 150)
plot_avp()
dev.off()
par(mfrow = c(1, 1))
