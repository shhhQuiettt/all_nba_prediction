library(tidyverse)
library(jsonlite)
library(magrittr)

save_width <- 12
save_height <- 7

raw_data <- read_json("results/experiments_results.json")
df <- tibble(
  min_train_season = map_int(raw_data, "min_train_season"),
  classifier = map_chr(raw_data, "classifier"),
  regressor = map_chr(raw_data, "regressor"),
  score_rookie_2026 = map_dbl(
    raw_data,
    ~ pluck(.x, "scores_all_rookie", "2026")
  ),
  score_all_nba_2026 = map_dbl(raw_data, ~ pluck(.x, "scores_all_nba", "2026"))
) %>%
  mutate(total_score_2026 = score_rookie_2026 + score_all_nba_2026) %>%
  mutate(
    classifier = if_else(
      classifier == "DummyClassifier",
      "No Classifier",
      classifier
    )
  )


score_by_training_recency_boxplot = ggplot(
  df,
  aes(x = factor(min_train_season), y = total_score_2026)
) +
  geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.4, color = "darkblue") +
  theme_minimal(base_size = 14) +
  labs(
    title = "Effect of Training Data Recency on 2026 Validation",
    subtitle = "Does starting the training data later improve modern predictions?",
    x = "Minimum Training Season (Start Year)",
    y = "Total Score 2026 (Rookie + All-NBA)"
  ) +
  theme(panel.grid.minor = element_blank())
score_by_training_recency_boxplot

ggsave(
  "report/plots/score_by_training_recency_boxplot.png",
  score_by_training_recency_boxplot,
  width = save_width,
  height = save_height,
  dpi = 300,
)


ggplot(df, aes(x = factor(min_train_season), y = total_score_2026)) +
  geom_violin(fill = "steelblue", alpha = 0.6, trim = FALSE, color = NA) +
  geom_jitter(width = 0.15, alpha = 0.4, color = "darkblue", size = 1.2) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, alpha = 0.7) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Effect of Training Data Recency on 2026 Validation",
    subtitle = "Distribution of scores by training start year",
    x = "Minimum Training Season (Start Year)",
    y = "Total Score 2026 (Rookie + All-NBA)"
  ) +
  theme(panel.grid.minor = element_blank())

####### Classifier

# Violin
classifier_distributions <- ggplot(
  df,
  aes(
    x = reorder(classifier, total_score_2026, FUN = median),
    y = total_score_2026
  )
) +
  geom_violin(
    fill = "lightcoral",
    alpha = 0.6,
    trim = FALSE,
    color = "darkred"
  ) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, alpha = 0.7) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Performance Density by Classifier Model",
    x = "Classifier",
    y = "Total Score 2026"
  )
classifier_distributions

ggsave(
  "report/plots/classifier_distributions.png",
  classifier_distributions,
  width = save_width - 2,
  height = save_height,
  dpi = 300,
)


ggplot(
  df,
  aes(
    x = reorder(classifier, total_score_2026, FUN = median),
    y = total_score_2026
  )
) +
  geom_boxplot(fill = "lightcoral", alpha = 0.7) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Performance by Classifier Model",
    x = "Classifier",
    y = "Total Score 2026"
  )


#### Regressor

# Box
ggplot(
  df,
  aes(
    x = reorder(regressor, total_score_2026, FUN = median),
    y = total_score_2026
  )
) +
  geom_boxplot(fill = "mediumseagreen", alpha = 0.7) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Performance by Regression Model",
    x = "Regressor",
    y = "Total Score 2026"
  )

# Violin
regressor_distributions <- ggplot(
  df,
  aes(
    x = reorder(regressor, total_score_2026, FUN = median),
    y = total_score_2026
  )
) +
  geom_violin(
    fill = "mediumseagreen",
    alpha = 0.6,
    trim = FALSE,
    color = "darkgreen"
  ) +
  geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA, alpha = 0.7) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    title = "Performance Density by Regression Model",
    x = "Regressor",
    y = "Total Score 2026"
  )
regressor_distributions

ggsave(
  "report/plots/regressor_distributions.png",
  regressor_distributions,
  width = save_width - 2,
  height = save_height,
  dpi = 300,
)

#### Combo

combo_summary <- df %>%
  group_by(classifier, regressor) %>%
  summarize(
    median_score = median(total_score_2026),
    max_score = max(total_score_2026),
    .groups = "drop"
  )

combo_median = ggplot(
  combo_summary,
  aes(x = regressor, y = classifier, fill = median_score)
) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = round(median_score, 1)), color = "black", size = 4) +
  scale_fill_viridis_c(direction = 1) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Hurdle Model Synergy: Classifier vs. Regressor",
    subtitle = "Median 2026 Total Score for each combination",
    x = "Regressor Used",
    y = "Classifier Used",
    fill = "Median\nScore"
  )
combo_median

ggsave(
  "report/plots/combo_median.png",
  combo_median,
  width = 10,
  height = 8,
  dpi = 300,
)

combo_max = ggplot(
  combo_summary,
  aes(x = regressor, y = classifier, fill = max_score)
) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = round(max_score, 1)), color = "black", size = 4) +
  scale_fill_viridis_c(direction = 1) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Hurdle Model Synergy: Classifier vs. Regressor",
    subtitle = "Max 2026 Total Score for each combination",
    x = "Regressor Used",
    y = "Classifier Used",
    fill = "Max\nScore"
  )
combo_max

ggsave(
  "report/plots/combo_max.png",
  combo_max,
  width = 10,
  height = 8,
  dpi = 300,
)


df %>%
  mutate(max_score = max(total_score_2026)) %>%
  filter(total_score_2026 == max_score) %>%
  select(classifier, regressor, total_score_2026, min_train_season)
