# NBA Awards Prediction
Krzysztof Skrobała 156039 

## Link to repository
Code and rendered report can be founct on [GitHub](https://github.com/shhhQuiettt/all_nba_prediction) 

## Introduction
This report details the implementation, and results of a predictive modeling project aimed at forecasting the end-of-season All-NBA and All-Rookie teams.

## All-NBA and All-Rookie Teams
The All-NBA Team is an annual award for the best players in the National Basketball Association (NBA) following every regular season. Voted on by a global panel of sportswriters and broadcasters, the honor is divided into three distinct five-man lineups: the First, Second, and Third teams.

Similarly, the All-Rookie Team honors the top performing first-year players in the league. Voted on by NBA head coaches, this distinction is divided into two five-man lineups: the First and Second teams.

## Metric
To evaluate the model's prediction performance, a custom, point-based scoring system is utilized. 

Points are awarded per player based on the following criteria:

- Exact Match: **10 points** (The player is placed in the correct team).
- Off by One: **8 points** (The player is placed one team above or below their actual result).
- Off by Two: **6 points** (The player is placed two teams above or below their actual result).
- Un-nominated: **0 points**

To further reward highly accurate models, cumulative bonus points are awarded for correctly predicting multiple players within the same specific exact lineup:
- 2 correct players: +5 points
- 3 correct players: +10 points
- 4 correct players: +20 points
- 5 correct players: +40 points

### Maximum Possible Score & Normalization
A perfect prediction for a single team (5 players at 10 points each, plus the 40-point completion bonus) yields exactly 90 points. Across the five target teams, a perfect results in 450 points.

## Approach
The predictive modeling approach is formulated as a _regression_ problem, where the model is trained to predict a _vote share_ distinctively for _ALL-NBA_ and _ALL-ROOKIE-NBA_. This is a `0-1` percentage value, which is used to create a ranking of players. The top 5 groups of players are then selected as the predicted team for each category (All-NBA and All-Rookie). 

### Tackling Sparsity
One of the problems with the data is the sparsity of votes. Almost _all_ players receive `0` votes, and only a small fraction of players receive any votes at all. This creates a highly imbalanced dataset, which can be challenging for traditional regression models to learn from effectively.

To tackle this issue, a __hurdle regression__ approach is implemented. This involves first training a __binary classifier__ to predict probabilities whether a player receives any votes (i.e., whether they are nominated or not). Then, for the a separate regression model is trained to predict their vote share. This two-step process helps to mitigate the effects of sparsity and allows the model to focus on learning from the relevant subset of data.

The comparison to a standard regression model is included in _Experiments_ section below.


## Data Acquisition
Despite the official NBA stat API being available, the task of gathering data was very painstaking.

To gather league statistics, the `nba_api` package was used [link](https://github.com/swar/nba_api). 

Both basic and advanced statistics were collected, ensuring they are taken from regular season, and not from the playoffs, as the voting is based on regular season performance.

However, the package do not provide access to the voting data. It had to be scraped from the basketball reference website [link](https://www.basketball-reference.com/awards/awards_2023.html). This process involved writing custom web scraping scripts to extract the relevant information from the HTML pages

Additionally, a column regarding team position in the final season ranking was added as an additional feature

Finally the data was cleaned and preprocessed in `R` as it is much more convenient and superior than `python` for tabular data manipulation and cleaning. The cleaned data was then exported to `dataset/final_ml_dataset.csv` file

### Code
1. Obtaining league statistics: `src/data/get_league_stats.py`
2. Scraping voting data: `src/data/scrape_voting_data.py`
3. Processing html tables and merging them: `src/data/process_html_tables.py`
3. Data cleaning and preprocessing: `src/data/preprocessing.R`

## Feature selection
The following features were selected for the model:

| Feature | Definition |
| :--- | :--- |
| **PTS_RANK** | A team or player's league-wide ranking based on their average points scored per game. |
| **PIE_RANK** | A ranking based on the Player Impact Estimate (PIE), indicating an entity's overall statistical contribution compared to the rest of the league. |
| **PLUS_MINUS_RANK** | The league ranking of a player's or team's average point differential (Plus/Minus) while they are on the court. |
| **W_PCT_RANK** | A team's rank based on their win percentage during the regular season. |
| **PTS** | The average number of points scored per game. |
| **REB** | The average number of total rebounds (offensive plus defensive) collected per game. |
| **AST** | The average number of assists distributed per game, representing passes that directly lead to a scored basket. |
| **STL** | The average number of steals recorded per game by legally taking the ball away from an opponent. |
| **BLK** | The average number of blocked shots per game where a defensive player legally deflects a field goal attempt. |
| **USG_PCT** | Usage Percentage estimates the percentage of team plays used by a specific player while they are on the floor. |
| **TS_PCT** | True Shooting Percentage measures overall shooting efficiency by factoring in the value of two-point field goals, three-point field goals, and free throws. |
| **PlayoffRank** | A team's current standing or projected seed within their respective conference for playoff qualification. |
| **WinPCT** | The ratio of games won to total games played, representing the team's overall winning percentage. |
| **TD3** | The total number of triple-doubles achieved, which occurs when a player accumulates double digits in three of the five major statistical categories in a single game. |
| **GP** | The total number of games played by a player or team during the specified season. |
| **IS_ROOKIE** | A boolean or binary flag indicating whether a player is in their first active season in the NBA. |


### Feature standardization
Because absolute value features (e.g., points, rebounds) are not directly comparable across different seasons due to changes in the style of play and league dynamics, they were standardized using league-wide rankings. This approach allows for a more consistent comparison of player performance across different eras.


### Encountered problems

**1. In reality there is not always 5 players per team**
For example in 1995-1996, there was a tie in the all-rookie nba voting, which resulted in 6 players being selected for the first team and only 4 players being selected for the second team. 
![](report/images/tie_1996.png)
Similar situation happened in 2007
![](report/images/tie_2007.png)

**2. Mismatch in names between stats and voting data**
The scraped data do not contain `PLAYER_ID` so it had to be obtained by matching the player names in the voting data with the player names in the stats data. 

Sometimes, the names do not match exactly, so different heuristics were implemented including removing UTF-8 characters, matching only the last name etc. 

All heuristics matches were logged as warnings and validated manually.

**3. Duplicate names**
There are `4` occurances of duplicate names in the same season. Normally it should not be a problem, because the `PLAYER_ID` differentiates them, but I noticed it very late, and I verified these person have never received any votes, so I just dropped them from the dataset.

![](report/images/duplicates.png)

## Pipeline
The whole process was implemented in `sklearn` pipeline, which includes the following steps:
1. Selecting appropriate features
2. Regularization of the appropriate features
3. Predicting probabilities of being nominated using a binary classifier
4. Predicting vote share using a regression model
5. Scaling the predicted vote share by the predicted probabilities of being nominated
6. Ranking the players based on the scaled vote share and selecting the top 5 players for each team (All-NBA and All-Rookie)

## Validation
The validation was performed on the `2026` season, which is the most recent season for which the voting data is available. The model was trained on all previous seasons, and then used to predict the All-NBA and All-Rookie teams for the `2026` season. The predicted teams were then compared to the actual teams using the scoring system described in the _Metric_ section above.

Probably this is not the most reliable validation strategy, but unfortunately, the rules for selecting the All-NBA and All-Rookie teams changed in 2023, which makes it difficult to use older seasons for validation. The new rules eliminate the requirement for diverse positions across the 5 players in the same team, as well as the recent minimum games played requirement, which significantly changes the voting dynamics and criteria.

## Experiments

I wanted to check the following:

1. How does excluding old seasons affect the performance of the model? In the past the game were different, and the voting criteria change as well
2. Is the classification step necessary, if so how different classification models perform? 
3. How do different regression models perform?

To answer these questions, the pipeline was fitted and evaluated with the following configurations:
1. Training only from `min_train_season` from `1998` to `2024` seasons
2. Fitting the following classification models:  `Logistic Regression`, `XGBClassifier`, `HistGradient Boosting Classifier`,`Random Forest Classifier`, `Gradient Boosting Classifier`. Additionally balanced version were fitted, which means that the target classes were weighted inversely proportional to their frequency in the training data
3. Fitting the following regression models: `Linear Regression`, `Random Forest Regressor`, `Gradient Boosting Regressor`, `XGBRegressor`, `HistGradient Boosting Regressor`

### Code:
1. Pipeline implementation: `src/inference/pipeline.py`
2. Helpers: `src/inference/utils.py`
3. Building and running a pipeline: `src/inference/inference.py`
4. Experiments: `src/inference/experiments.py`

## Results

**The best score `375` for `2026` season was achieved by the Gradient Boosting Regressor with the classification by Logistic Regression.**

### Analysis

![](report/plots/score_by_training_recency_boxplot.png)
As can be seen, despite rule changes in 2023, the training benefits from including older seasons, which suggests that there are some underlying patterns in the data that are consistent across different eras of the NBA

---

![](/report/plots/classifier_distributions.png)

There are some minor differences between the classifier, suggesting that `Logistic Regression` worked the best.

What is very noticable, is that the `No classification` approach, performs significanly worse, suggesting that the `Hurdle Regression` approach is indeed necessary to tackle the sparsity of the data.

---

![](/report/plots/regressor_distributions.png)

The top regressors are very close to each other in median. What is suprising is that the `XGBRegressor` performs the worst, which may be due to the fact that it is more prone to overfitting, especially with a small dataset like this one, or because mistuned hyperparameters.

--- 

![](/report/plots/combo_median.png)
![](/report/plots/combo_max.png)

Again it is visible, that Logistic regression performed the best. What is interesting, is that both median and max score are the highest with the `Gradient Boosting Regressor`, which suggests that it is more consistent in its performance, while the `Random Forest Regressor` and has a higher variance, which may be due to its tendency to overfit on the training data.


---

### Code
1. Plots: `src/result_analysis/plots.R`

## Running final inference

### If using `uv`
```bash
uv sync
uv run python src/final_run.py --output_path /path/to/output.json
```

## If not using `uv`
```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
python src/final_run.py --output_path /path/to/output.json
```


## Possible improvements

### 1. Acknowledgin the rules change in 2023
The rules change in 2023 significantly affects the voting dynamics and criteria for selecting the All-NBA and All-Rookie teams. One of the changes is dropping the requirement for a concrete number of players from each position (e.g., guards, forwards, centers) in the All-NBA teams. Another change is the minimum games played requirement

### 2. Smarter feature selection
The current feature selection is based on domain knowledge and intuition. However, a more systematic approach could be implemented, such as using feature importance from a tree-based model or using a feature selection algorithm.

### 3. Hyperparameter tuning
The current model is trained with default hyperparameters. A more thorough approach would involve using techniques such as grid search or random search to find the optimal hyperparameters for both the classification and regression models.

### 4. Better validation strategy
The current validation is performed on a single season. Theoreticaly, a more robust approach would involve a cross-validation strategy, although because of the 2023 rule, which eliminates requirement for diverse positions across the 5 players in the same team, as well as the recent minimum games played requirement, it is hard to tell haw to acquire a real validation score.

### 5. Statistical significance testing
For the comparison results to be actually meaningful, it would be important to perform statistical significance testing to determine whether the differences in performance between different models and approaches are statistically significant or if they could be due to random chance.
