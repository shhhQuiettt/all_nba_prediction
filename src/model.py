import numpy as np
from utils import calculate_score, validate_season
from pipelines import create_pipeline

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from sklearn.pipeline import Pipeline
import logging
from pathlib import Path

SEED = 0xC0FFEE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    np.random.seed(0xC0FFEE)
    csv_path = Path("dataset/final_ml_dataset.csv")

    df = pd.read_csv(csv_path)
    df_train = df[(df.YEAR >= 2020) & (df.YEAR < 2026)]
    df_val = df[df.YEAR == 2026]
    del df

    features_to_normalize = ["PTS", "REB", "AST", "STL", "BLK", "TS_PCT", "TD3", "GP"]

    features = [
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
    ]

    base_classifier = HistGradientBoostingClassifier(random_state=SEED)
    base_regressor = HistGradientBoostingRegressor(random_state=SEED)

    nba_all_pipeline = create_pipeline(
        base_classifier=base_classifier,
        base_regressor=base_regressor,
        features=features,
        features_to_normalize=features_to_normalize,
    )

    nba_rookie_pipeline = create_pipeline(
        base_classifier=clone(base_classifier),
        base_regressor=clone(base_regressor),
        features=features,
        features_to_normalize=features_to_normalize,
    )

    df_train_all_nba = df_train[~df_train["IS_ROOKIE"]]
    df_train_all_rookie = df_train[df_train["IS_ROOKIE"]]
    del df_train
    y_train_all_nba = df_train_all_nba["VOTES_PERCENTAGE_ALL_NBA"]
    y_train_all_rookie = df_train_all_rookie["VOTES_PERCENTAGE_ALL_ROOKIE"]

    assert y_train_all_nba.notna().all(), (
        "Target variable for ALL NBA contains NaN values"
    )
    assert y_train_all_rookie.notna().all(), (
        "Target variable for ALL ROOKIE NBA contains NaN values"
    )
    assert (y_train_all_nba > 0).any(), (
        "There must be at least one positive sample in ALL NBA training data"
    )
    assert (y_train_all_nba <= 1).all(), (
        "Target variable values for ALL NBA must be in the range [0, 1]"
    )
    assert (y_train_all_rookie > 0).any(), (
        "There must be at least one positive sample in ALL ROOKIE NBA training data"
    )
    assert (y_train_all_rookie <= 1).all(), (
        "Target variable values for ALL ROOKIE NBA must be in the range [0, 1]"
    )

    nba_all_pipeline.fit(df_train_all_nba, y_train_all_nba)
    nba_rookie_pipeline.fit(df_train_all_rookie, y_train_all_rookie)
    print("Pipelines successfully trained!")

    for year in sorted(df_train_all_nba["SEASON"].unique()):
        logger.info(f"Validating season {year}...")
        total = 0
        for award, df, y in [
            ("ALL NBA", df_train_all_nba, y_train_all_nba),
            ("ALL ROOKIE NBA", df_train_all_rookie, y_train_all_rookie),
        ]:
            season_mask = df["SEASON"] == year
            X_season = df[season_mask]
            y_season = y[season_mask]

            df_season_results, score = validate_season(
                nba_all_pipeline, X_season, y_season
            )
            total += score
            logger.info(f"{award} score = {score}")
        logger.info(f"Total score for season {year} = {total}\n")

    del score
    df_val_all_nba = df_val[~df_val["IS_ROOKIE"]]
    df_val_all_rookie = df_val[df_val["IS_ROOKIE"]]
    y_val_all_nba = df_val_all_nba["VOTES_PERCENTAGE_ALL_NBA"]
    y_val_all_rookie = df_val_all_rookie["VOTES_PERCENTAGE_ALL_ROOKIE"]

    assert len(df_val_all_nba) == len(y_val_all_nba), (
        "Mismatch in number of samples for ALL NBA validation data"
    )
    assert len(df_val_all_rookie) == len(y_val_all_rookie), (
        "Mismatch in number of samples for ALL ROOKIE NBA validation data"
    )

    assert y_val_all_nba.notna().all(), (
        "Validation target variable for ALL NBA contains NaN values"
    )
    assert y_val_all_rookie.notna().all(), (
        "Validation target variable for ALL ROOKIE NBA contains NaN values"
    )
    assert (y_val_all_nba > 0).any(), (
        "There must be at least one positive sample in ALL NBA validation data"
    )
    assert (y_val_all_nba <= 1).all(), (
        "Validation target variable values for ALL NBA must be in the range [0, 1]"
    )
    assert (y_val_all_rookie > 0).any(), (
        "There must be at least one positive sample in ALL ROOKIE NBA validation data"
    )
    assert (y_val_all_rookie <= 1).all(), (
        "Validation target variable values for ALL ROOKIE NBA must be in the range [0, 1]"
    )

    df_val_results_all_nba, score_all_nba = validate_season(
        nba_all_pipeline, df_val_all_nba, y_val_all_nba
    )
    df_val_results_all_rookie, score_all_rookie = validate_season(
        nba_rookie_pipeline, df_val_all_rookie, y_val_all_rookie
    )

    assert df_val_results_all_rookie["actual_vote_share"].notna().all(), (
        "Validation results for ALL ROOKIE NBA contain NaN values in actual_vote_share"
    )

    logger.info("Validation Season 2026:")
    logger.info(f"ALL-NBA Score = {score_all_nba}")
    logger.info(f"ALL-ROOKIE NBA Score = {score_all_rookie}")
    logger.info(f"Total Score = {score_all_nba + score_all_rookie}")

    logger.info("Selected players for 2026 All-NBA Teams:")
    for place in (1, 2, 3):
        selected_players = df_val_results_all_nba[
            df_val_results_all_nba["predicted_place"] == place
        ]
        logger.info(f"Place {place}:")
        # [["PLAYER_NAME", "actual_vote_share", "predicted_vote_share", "predicted_place", "actual_place"]]
        # print in nice tabular format
        df_selected_players = selected_players[
            [
                "PLAYER_NAME",
                "actual_vote_share",
                "predicted_vote_share",
                "predicted_place",
                "actual_place",
            ]
        ]
        print(df_selected_players.to_string(index=False))

    for place in (1, 2):
        selected_players = df_val_results_all_rookie[
            df_val_results_all_rookie["predicted_place"] == place
        ]
        logger.info(f"Place {place} Rookie:")

        df_selected_players = selected_players[
            [
                "PLAYER_NAME",
                "actual_vote_share",
                "predicted_vote_share",
                "predicted_place",
                "actual_place",
            ]
        ]
        print(df_selected_players.to_string(index=False))


if __name__ == "__main__":
    main()
