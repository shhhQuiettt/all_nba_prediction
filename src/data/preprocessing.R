library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)
library(scales)

player_stats <- read.csv("./dataset/player_stats.csv")
player_stats = player_stats %>%
  mutate(YEAR = as.numeric(substr(SEASON, 1, 4)) + 1) %>%
  mutate(PLAYER_SEASON_ID = paste(PLAYER_ID, YEAR, sep = "_"))


awards = read.csv("./dataset/awards.csv")
awards_cleaned = awards %>%
  mutate(YEAR = as.numeric(YEAR)) %>%
  mutate(PLAYER_SEASON_ID = paste(PLAYER_ID, YEAR, sep = "_")) %>%
  select(PLAYER_SEASON_ID, YEAR, PLAYER_ID, AWARD, VOTES_PERCENTAGE)


awards_wide = awards_cleaned %>%
  pivot_wider(
    names_from = AWARD,
    values_from = VOTES_PERCENTAGE,
    names_prefix = "VOTES_PERCENTAGE_",
  ) %>%
  rename(
    VOTES_PERCENTAGE_ALL_NBA = "VOTES_PERCENTAGE_ALL-NBA",
    VOTES_PERCENTAGE_ALL_ROOKIE = "VOTES_PERCENTAGE_ALL-ROOKIE"
  ) %>%
  mutate(
    VOTES_PERCENTAGE_ALL_NBA = ifelse(
      is.na(VOTES_PERCENTAGE_ALL_NBA),
      0.0,
      VOTES_PERCENTAGE_ALL_NBA
    ),
    VOTES_PERCENTAGE_ALL_ROOKIE = ifelse(
      is.na(VOTES_PERCENTAGE_ALL_ROOKIE),
      0.0,
      VOTES_PERCENTAGE_ALL_ROOKIE
    )
  )


final_ml_dataset = player_stats %>%
  left_join(awards_wide, by = "PLAYER_SEASON_ID") %>%
  mutate(
    VOTES_PERCENTAGE_ALL_NBA = ifelse(
      is.na(VOTES_PERCENTAGE_ALL_NBA),
      0.0,
      VOTES_PERCENTAGE_ALL_NBA
    )
  ) %>%
  mutate(
    VOTES_PERCENTAGE_ALL_ROOKIE = ifelse(
      is.na(VOTES_PERCENTAGE_ALL_ROOKIE),
      0.0,
      VOTES_PERCENTAGE_ALL_ROOKIE
    )
  ) %>%
  rename(PLAYER_ID = PLAYER_ID.x, YEAR = YEAR.x) %>%
  group_by(PLAYER_ID) %>%
  mutate(IS_ROOKIE = (SEASON == min(SEASON))) %>%
  ungroup()

final_ml_dataset %>%
  group_by(PLAYER_NAME, YEAR) %>%
  summarise(
    count = n(),
    max_all_nba = max(VOTES_PERCENTAGE_ALL_NBA),
    max_all_rookie = max(VOTES_PERCENTAGE_ALL_ROOKIE)
  ) %>%
  filter(count > 1) %>%
  arrange(desc(count))

final_ml_dataset %>% count()
final_ml_dataset = final_ml_dataset %>%
  distinct(PLAYER_NAME, YEAR, .keep_all = TRUE)
final_ml_dataset %>% count()


identifiers = c(
  "PLAYER_SEASON_ID",
  "PLAYER_ID",
  "SEASON",
  "YEAR",
  "PLAYER_NAME",
  "TEAM_ID",
  "TEAM_ABBREVIATION"
)
features <- c(
  "PTS_RANK",
  "PIE_RANK",
  "PLUS_MINUS_RANK",
  "W_PCT_RANK",
  "PTS",
  "REB",
  "AST",
  "STL",
  "BLK",
  "USG_PCT",
  "TS_PCT",
  "PIE",
  "PlayoffRank",
  "WinPCT",
  "TD3",
  "GP",
  "IS_ROOKIE"
)
targets <- c("VOTES_PERCENTAGE_ALL_NBA", "VOTES_PERCENTAGE_ALL_ROOKIE")

final_ml_dataset = final_ml_dataset %>%
  select(all_of(identifiers), all_of(features), all_of(targets))

write.csv(final_ml_dataset, "./dataset/final_ml_dataset.csv", row.names = FALSE)


custom_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "gray40", size = 12),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray85", linetype = "dashed"),
    axis.title = element_text(face = "bold", color = "gray30"),
    axis.text = element_text(color = "gray20")
  )

p1 <- final_ml_dataset %>%
  filter(!IS_ROOKIE) %>%
  ggplot(aes(x = VOTES_PERCENTAGE_ALL_NBA)) +
  geom_histogram(
    binwidth = 0.05,
    fill = "#2C3E50",
    color = "white",
    alpha = 0.9,
    boundary = 0
  ) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.25)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "ALL-NBA Voting",
    subtitle = "Non-Rookie Players",
    x = "Share of Total Votes",
    y = "Number of Players"
  ) +
  custom_theme

p2 <- final_ml_dataset %>%
  filter(IS_ROOKIE) %>%
  ggplot(aes(x = VOTES_PERCENTAGE_ALL_ROOKIE)) +
  geom_histogram(
    binwidth = 0.05,
    fill = "#E67E22",
    color = "white",
    alpha = 0.9,
    boundary = 0
  ) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.25)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "ALL-ROOKIE Voting",
    subtitle = "Rookie Players",
    x = "Share of Total Votes",
    y = NULL
  ) +
  custom_theme

p1 +
  p2 +
  plot_annotation(
    title = "NBA Award Voting Distribution",
    caption = "Bin width: 5%",
    theme = theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5), # Centered master title
      plot.caption = element_text(color = "gray50", size = 10, face = "italic")
    )
  )

ggsave("report/plots/votes_distribution.png", p1 + p2, width = 12, height = 6)
