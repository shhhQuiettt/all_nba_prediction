library(dplyr)
library(ggplot2)
library(tidyr)

data <- read.csv("./dataset/players_stats.csv")

data
players <- read.csv("./dataset/players.csv")
players %>% count()

data %>% filter(PLAYER_ID==1824)

all = players %>% group_by(PERSON_ID) %>% select(PERSON_ID)
all  %>% count()

gathered = data %>% group_by(PLAYER_ID) %>% summarise(count = n()) %>% select(PLAYER_ID)
gathered %>% count()


missing = setdiff(all$PERSON_ID, gathered$PLAYER_ID)
missing

players %>% filter(PERSON_ID == 1824)

players %>% filter(FROM_YEAR == 1998) %>% select(DISPLAY_FIRST_LAST)

gathered %>% group_by(PLAYER_ID)

players %>% filter(DISPLAY_FIRST_LAST == "Precious Achiuwa") 

data %>% head()

players %>% colnames()
players %>% select(FROM_YEAR) %>% min()

######
players %>% colnames()

players%>% head() %>% select(DISPLAY_FIRST_LAST, FROM_YEAR, TO_YEAR)

data_raw <- read.csv("./dataset/players_stats.csv")
data_raw %>% colnames()


award_data <- read.csv("./dataset/tables/all_nba_2001.csv")
award_data %>% colnames()
award_data %>% filter(PLACE == "2nd")
                      

award_data %>% select("Unnamed..0_level_0")
award_data %>% select(Voting.2)

award_data %>% head()
award_data

# filter if the name contains "Lever"
players %>% filter(grepl("Lever", DISPLAY_FIRST_LAST)) 

awards = read.csv("./dataset/awards.csv")
awards %>% filter(PLACE %in% c("1st", "2nd", "3rd")) %>% group_by(YEAR, AWARD) %>% summarise(count = n())

awards %>% filter(AWARD == "ALL-ROOKIE", PLACE %in% c("1st", "2nd", "3rd")) %>% group_by(YEAR, AWARD) %>% summarise(count = n()) %>% ggplot(aes(x=YEAR, y=count)) + geom_bar(stat="identity") + ggtitle("Number of ALL-NBA awards per year")



### JOINGING

player_stats <- read.csv("./dataset/players_stats.csv")
player_stats <- player_stats %>%
    mutate(YEAR = as.numeric(substr(SEASON_ID, 1, 4))+1) %>% 
    mutate(player_year_id = paste(PLAYER_ID, YEAR, sep="_")) 

awards <- read.csv("./dataset/awards.csv")
awards = awards %>% mutate(YEAR = as.numeric(YEAR)) %>%
    mutate(player_year_id = paste(PLAYER_ID, YEAR, sep="_")) 


duplicated_stats = player_stats %>% 
    group_by(player_year_id) %>%
    summarise(count = n()) %>% 
    filter(count > 1) %>%
    mutate(YEAR = as.numeric(substr(player_year_id, nchar(player_year_id)-3, nchar(player_year_id)))) 

duplicated_stats %>% count()

duplicated_stats %>% filter(YEAR > 2020) %>% left_join(awards, by = "player_year_id") %>% filter(!is.na(AWARD)) %>% count()

# remove duplicated stats
player_stats %>% group_by(player_year_id) %>% summarise(count = n()) %>% filter(count == 1) %>% select(player_year_id) %>% left_join(player_stats, by = "player_year_id")  %>% count()


player_stats %>% filter(PLAYER_ID == 3) %>% select(player_year_id, PLAYER_ID, SEASON_ID, YEAR, AST, FGA)



joined_data <- player_stats %>% left_join(awards, by = "PLAYER_ID")
joined_data <- joined_data %>% mutate(VOTES_PERCENTAGE = ifelse(is.na(VOTES_PERCENTAGE), 0.0, VOTES_PERCENTAGE))

joined_data %>% filter(YEAR.x != YEAR.y) %>% select(PLAYER_ID, YEAR.x, YEAR.y)
joined_data %>% colnames()

joined_data %>% filter(PLACE %in% c("1st", "2nd", "3rd")) %>% group_by(YEAR, AWARD) %>% summarise(count = n())


#

player_stats %>
