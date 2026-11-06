# ==============================================================================
# Unsupervised Analysis of Screen Time and Mental Wellness Behaviours
# PCA + Hierarchical clustering (Ward) + K-Means
#
# Author : Mohamed Amine Laabidi
# Course : Statistical Learning — MSc Data Science for Economics and Health,
#          Università degli Studi di Milano (April 2026)
# Data   : "Screen Time vs Mental Wellness Survey – 2025" (Kaggle, n = 400)
#
# Run from the repository root:  source("R/02_unsupervised.R")
# ==============================================================================


# ── 1. SETUP ──────────────────────────────────────────────────────────────────
library(tidyverse)
library(factoextra)   # fviz_* plots
library(cluster)      # silhouette
library(clue)         # solve_LSAP (label matching)
library(gridExtra)

dir.create("figures", showWarnings = FALSE)


# ── 2. LOAD & SELECT NUMERIC BEHAVIOURAL VARIABLES ───────────────────────────
df_raw <- read.csv("data/ScreenTime_vs_MentalWellness.csv")
df_raw$X <- NULL
df_raw$sleep_quality_binary <- ifelse(df_raw$sleep_quality_1_5 == 1, 1, 0)

# Same exclusions as the supervised analysis
df_clean <- df_raw %>%
  dplyr::select(-user_id, -productivity_0_100,
                -screen_time_hours, -sleep_quality_1_5)

# Clustering uses numeric behaviours only: the target and the categorical
# variables are kept aside and used afterwards to profile the clusters.
df_num <- df_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-mental_wellness_index_0_100)

wellness <- df_clean$mental_wellness_index_0_100

cat("Numeric variables used:\n"); print(names(df_num))
# age, work_screen_hours, leisure_screen_hours, sleep_hours,
# stress_level_0_10, exercise_minutes_per_week,
# social_hours_per_week, sleep_quality_binary


# ── 3. SCALE ──────────────────────────────────────────────────────────────────
df_scaled <- scale(df_num)
cat("Mean of scaled data (should be ~0):\n", round(colMeans(df_scaled), 3), "\n")
cat("SD of scaled data (should be ~1):\n",   round(apply(df_scaled, 2, sd), 3), "\n")


# ── 4. PCA ────────────────────────────────────────────────────────────────────
pca_result <- prcomp(df_scaled, center = FALSE, scale. = FALSE)  # already scaled

fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50),
         main = "PCA — Variance Explained per Component")

fviz_pca_biplot(pca_result, repel = TRUE, geom.ind = "point",
                alpha.ind = 0.4, col.ind = "steelblue",
                col.var = "red", label = "var",
                title = "PCA — Biplot (Variables + Participants)")

fviz_contrib(pca_result, choice = "var", axes = 1, title = "Variable Contributions to PC1")
fviz_contrib(pca_result, choice = "var", axes = 2, title = "Variable Contributions to PC2")
fviz_contrib(pca_result, choice = "var", axes = 3, title = "Variable Contributions to PC3")

summary(pca_result)
# PC1 30.2%, PC2 16.0%, PC3 13.4% -> first 3 PCs keep 59.7% of the variance.
# From PC4 onwards each component explains about as much variance as a single
# original variable (eigenvalue ~1 or less), so it mostly adds noise to distances.

pca_scores  <- as.data.frame(pca_result$x)
pca_2d      <- pca_scores[, 1:2]   # for plotting
pca_cluster <- pca_scores[, 1:3]   # for clustering


# ── 5. HIERARCHICAL CLUSTERING ────────────────────────────────────────────────
dist_pca <- dist(pca_cluster, method = "euclidean")
hc_fit   <- hclust(dist_pca, method = "ward.D2")

# Silhouette for K = 2 to 8
hc_sil <- sapply(2:8, function(k) {
  mean(silhouette(cutree(hc_fit, k = k), dist_pca)[, 3])
})

ggplot(data.frame(K = 2:8, Silhouette = hc_sil), aes(x = K, y = Silhouette)) +
  geom_line(color = "seagreen", linewidth = 1) +
  geom_point(size = 3, color = "seagreen") +
  labs(title = "Hierarchical — Silhouette Scores", x = "K", y = "Mean Silhouette") +
  theme_minimal()

cat("Best K from Hierarchical silhouette:", which.max(hc_sil) + 1, "\n")   # 3

# Merge heights of the last steps: big jumps at K = 2 and K = 3, flat after
rev(tail(hc_fit$height, 5))


# ── 6. K-MEANS ────────────────────────────────────────────────────────────────
set.seed(42)

km_wss <- sapply(2:8, function(k) {
  kmeans(pca_cluster, centers = k, nstart = 25, iter.max = 100)$tot.withinss
})

km_sil <- sapply(2:8, function(k) {
  km <- kmeans(pca_cluster, centers = k, nstart = 25, iter.max = 100)
  mean(silhouette(km$cluster, dist_pca)[, 3])
})

p1 <- ggplot(data.frame(K = 2:8, WSS = km_wss), aes(x = K, y = WSS)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "K-Means — Elbow Plot", x = "K", y = "Total WSS") +
  theme_minimal()

p2 <- ggplot(data.frame(K = 2:8, Sil = km_sil), aes(x = K, y = Sil)) +
  geom_line(color = "coral", linewidth = 1) +
  geom_point(size = 3, color = "coral") +
  labs(title = "K-Means — Silhouette", x = "K", y = "Mean Silhouette") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

cat("Best K from K-Means silhouette:", which.max(km_sil) + 1, "\n")      # 2
round(km_sil, 3)  # K = 2: 0.349, K = 3: 0.340 -> nearly tied


# ── 7. CHOICE OF K ────────────────────────────────────────────────────────────
# The hierarchical silhouette peaks at K = 3; the K-Means silhouette is highest
# at K = 2 but K = 3 is almost tied (0.340 vs 0.349). K = 3 is retained because
# it is supported by the hierarchical solution and separates two distinct
# high-stress groups that K = 2 merges. All silhouettes are below 0.5, so the
# cluster structure is weak-to-moderate rather than sharply separated.
K <- 3

hc_labels <- cutree(hc_fit, k = K)

fviz_dend(hc_fit, k = K, cex = 0.4,
          k_colors = c("steelblue", "coral", "seagreen"),
          color_labels_by_k = TRUE, rect = TRUE,
          main = "Hierarchical Clustering — Dendrogram (Ward, Euclidean on PCA)")

p_clusters <- fviz_cluster(list(data = pca_2d, cluster = hc_labels),
                           palette = c("steelblue", "coral", "seagreen"),
                           geom = "point", ellipse.type = "convex",
                           main = "Hierarchical Clusters in PCA Space (K = 3)")
print(p_clusters)
ggsave("figures/clusters_pca.png", p_clusters, width = 7, height = 5, dpi = 150)

set.seed(42)
km3 <- kmeans(pca_cluster, centers = K, nstart = 25, iter.max = 100)

sil_km3 <- silhouette(km3$cluster, dist_pca)
fviz_silhouette(sil_km3, palette = "jco", main = "K-Means (K = 3) — Silhouette Plot")
cat("Hierarchical (K = 3) mean silhouette:", round(mean(silhouette(hc_labels, dist_pca)[, 3]), 3), "\n")
cat("K-Means      (K = 3) mean silhouette:", round(mean(sil_km3[, 3]), 3), "\n")


# ── 8. AGREEMENT BETWEEN METHODS ──────────────────────────────────────────────
# Cluster numbers are arbitrary, so labels are matched optimally (Hungarian
# algorithm) before computing the share of identical assignments.
conf_mat <- table(KMeans = km3$cluster, HC = hc_labels)
print(conf_mat)
matched   <- solve_LSAP(conf_mat, maximum = TRUE)
agreement <- sum(conf_mat[cbind(seq_len(K), matched)]) / nrow(pca_cluster)
cat("K-Means vs Hierarchical agreement (label-matched):", round(agreement * 100, 1), "%\n")


# ── 9. CLUSTER PROFILING (Hierarchical, K = 3) ───────────────────────────────
df_profile <- df_num %>%
  mutate(wellness   = wellness,
         cluster    = factor(hc_labels),
         work_mode  = df_raw$work_mode,
         occupation = df_raw$occupation)

profile <- df_profile %>%
  group_by(cluster) %>%
  summarise(
    N                  = n(),
    Pct                = round(n() / nrow(df_profile) * 100, 1),
    Avg_Wellness       = round(mean(wellness), 1),
    Avg_Stress         = round(mean(stress_level_0_10), 2),
    Avg_Sleep_Hours    = round(mean(sleep_hours), 2),
    Avg_Leisure_Screen = round(mean(leisure_screen_hours), 2),
    Avg_Work_Screen    = round(mean(work_screen_hours), 2),
    Avg_Exercise       = round(mean(exercise_minutes_per_week), 1),
    Pct_Poor_Sleep     = round(mean(sleep_quality_binary) * 100, 1)
  )
print(as.data.frame(profile))
write.csv(profile, "figures/cluster_profile.csv", row.names = FALSE)

# Categorical profile (not used to build the clusters)
round(prop.table(table(df_profile$cluster, df_profile$work_mode), 1) * 100, 1)
round(prop.table(table(df_profile$cluster, df_profile$occupation), 1) * 100, 1)
chisq.test(table(df_profile$cluster, df_profile$work_mode))

# Do the clusters differ in wellness? (wellness was NOT used for clustering)
kruskal.test(wellness ~ cluster, data = df_profile)

p_well <- ggplot(df_profile, aes(x = cluster, y = wellness, fill = cluster)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "coral", "seagreen")) +
  scale_x_discrete(labels = c("1\nWork-screen stressed", "2\nBalanced", "3\nLeisure-screen stressed")) +
  labs(title = "Mental Wellness by Cluster (K = 3, Hierarchical)",
       x = NULL, y = "Mental Wellness Index") +
  theme_minimal() + theme(legend.position = "none")
print(p_well)
ggsave("figures/wellness_by_cluster.png", p_well, width = 7, height = 5, dpi = 150)
