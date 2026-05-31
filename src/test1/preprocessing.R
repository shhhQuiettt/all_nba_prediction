library(dplyr)
library(ggplot2)
library(tidyr)

player_stats <- read.csv("./dataset/player_stats.csv")
player_stats = player_stats %>% 
    mutate(YEAR = as.numeric(substr(SEASON, 1, 4))+1) %>%
    mutate(PLAYER_SEASON_ID = paste(PLAYER_ID, YEAR, sep="_"))

player_stats %>% colnames()

awards = read.csv("./dataset/awards.csv")
awards_cleaned = awards %>% 
    mutate(YEAR = as.numeric(YEAR)) %>%
    mutate(PLAYER_SEASON_ID = paste(PLAYER_ID, YEAR, sep="_")) %>%
    select(PLAYER_SEASON_ID, PLAYER_ID, AWARD, VOTES_PERCENTAGE)
    

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
           VOTES_PERCENTAGE_ALL_NBA = ifelse(is.na(VOTES_PERCENTAGE_ALL_NBA), 0.0, VOTES_PERCENTAGE_ALL_NBA),
           VOTES_PERCENTAGE_ALL_ROOKIE = ifelse(is.na(VOTES_PERCENTAGE_ALL_ROOKIE), 0.0, VOTES_PERCENTAGE_ALL_ROOKIE)
    )


final_ml_dataset = player_stats %>% left_join(awards_wide, by = "PLAYER_SEASON_ID") %>%
    mutate(VOTES_PERCENTAGE_ALL_NBA = ifelse(is.na(VOTES_PERCENTAGE_ALL_NBA), 0.0, VOTES_PERCENTAGE_ALL_NBA)) %>%
    mutate(VOTES_PERCENTAGE_ALL_ROOKIE = ifelse(is.na(VOTES_PERCENTAGE_ALL_ROOKIE), 0.0, VOTES_PERCENTAGE_ALL_ROOKIE)) %>%
    rename(PLAYER_ID = PLAYER_ID.x) %>%
    group_by(PLAYER_ID) %>%
    mutate(IS_ROOKIE = (SEASON == min(SEASON))) %>%
    ungroup()



identifiers = c("PLAYER_SEASON_ID", "PLAYER_ID", "SEASON", "YEAR", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION")
features <- c(
  "PTS_RANK", "PIE_RANK", "PLUS_MINUS_RANK", "W_PCT_RANK",
  "PTS", "REB", "AST", "STL", "BLK",
  "USG_PCT", "TS_PCT", "PIE",
  "PlayoffRank", "WinPCT",
  "TD3", "GP", 
  "IS_ROOKIE"
)
targets <- c("VOTES_PERCENTAGE_ALL_NBA", "VOTES_PERCENTAGE_ALL_ROOKIE")



final_ml_dataset = final_ml_dataset %>% select(all_of(identifiers), all_of(features), all_of(targets))  

# export to csv
write.csv(final_ml_dataset, "./dataset/final_ml_dataset.csv", row.names = FALSE)



awards %>% filter(YEAR == 2026) %>% count()


