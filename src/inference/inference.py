import numpy as np
from inference.utils import validate_season
from inference.pipeline import create_pipeline
from dataclasses import dataclass
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

import logging
from pathlib import Path


@dataclass
class ExperimentConfig:
    csv_path: Path
    seed: int
    features: list[str]
    features_to_normalize: list[str]
    base_classifier: BaseEstimator
    base_regressor: RegressorMixin
    min_train_season: int


@dataclass
class ExperimentResults:
    season_scores_all_nba: dict[int, int]
    season_scores_all_rookie: dict[int, int]
    selected_2026_players_all_nba: pd.DataFrame
    selected_2026_players_all_rookie: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "season_scores_all_nba": self.season_scores_all_nba,
            "season_scores_all_rookie": self.season_scores_all_rookie,
            "selected_2026_players_all_nba": self.selected_2026_players_all_nba.to_dict(
                orient="records"
            ),
            "selected_2026_players_all_rookie": self.selected_2026_players_all_rookie.to_dict(
                orient="records"
            ),
        }


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentResults:
    logger = logging.getLogger(__name__)

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    np.random.seed(config.seed)
    csv_path = config.csv_path

    df = pd.read_csv(csv_path)
    df_train = df[(df.YEAR >= config.min_train_season) & (df.YEAR < 2026)]
    df_val = df[df.YEAR == 2026]

    features_to_normalize = config.features_to_normalize
    features = config.features
    assert set(features_to_normalize).issubset(
        set(features)
    ), "Features to normalize must be a subset of the features used in the model"

    nba_all_pipeline = create_pipeline(
        base_classifier=config.base_classifier,
        base_regressor=config.base_regressor,
        features=features,
        features_to_normalize=features_to_normalize,
    )

    nba_rookie_pipeline = create_pipeline(
        base_classifier=clone(config.base_classifier),
        base_regressor=clone(config.base_regressor),
        features=features,
        features_to_normalize=features_to_normalize,
    )

    df_train_all_nba = df_train[~df_train["IS_ROOKIE"]]
    df_train_all_rookie = df_train[df_train["IS_ROOKIE"]]
    del df_train
    y_train_all_nba = df_train_all_nba["VOTES_PERCENTAGE_ALL_NBA"]
    y_train_all_rookie = df_train_all_rookie["VOTES_PERCENTAGE_ALL_ROOKIE"]

    assert (
        y_train_all_nba.notna().all()
    ), "Target variable for ALL NBA contains NaN values"
    assert (
        y_train_all_rookie.notna().all()
    ), "Target variable for ALL ROOKIE NBA contains NaN values"
    assert (
        y_train_all_nba > 0
    ).any(), "There must be at least one positive sample in ALL NBA training data"
    assert (
        y_train_all_nba <= 1
    ).all(), "Target variable values for ALL NBA must be in the range [0, 1]"
    assert (
        y_train_all_rookie > 0
    ).any(), (
        "There must be at least one positive sample in ALL ROOKIE NBA training data"
    )
    assert (
        y_train_all_rookie <= 1
    ).all(), "Target variable values for ALL ROOKIE NBA must be in the range [0, 1]"

    nba_all_pipeline.fit(df_train_all_nba, y_train_all_nba)
    nba_rookie_pipeline.fit(df_train_all_rookie, y_train_all_rookie)
    logger.info("Pipelines successfully trained!")

    season_scores_all_nba = {}
    season_scores_all_rookie = {}

    for year in sorted(df_train_all_nba["YEAR"].unique()):
        logger.info(f"Validating season {year}...")
        total = 0
        for award, df, y, pipeline in [
            ("ALL NBA", df_train_all_nba, y_train_all_nba, nba_all_pipeline),
            (
                "ALL ROOKIE NBA",
                df_train_all_rookie,
                y_train_all_rookie,
                nba_rookie_pipeline,
            ),
        ]:
            season_mask = df["YEAR"] == year
            X_season = df[season_mask]
            y_season = y[season_mask]

            df_season_results, score = validate_season(
                pipeline, X_season, y_season, is_rookie=(award == "ALL ROOKIE NBA")
            )
            if award == "ALL NBA":
                season_scores_all_nba[int(year)] = score
            else:
                season_scores_all_rookie[int(year)] = score

            total += score
            logger.info(f"{award} score = {score}")
        logger.info(f"Total score for season {year} = {total}\n")

    del score
    df_val_all_nba = df_val[~df_val["IS_ROOKIE"]]
    df_val_all_rookie = df_val[df_val["IS_ROOKIE"]]
    y_val_all_nba = df_val_all_nba["VOTES_PERCENTAGE_ALL_NBA"]
    y_val_all_rookie = df_val_all_rookie["VOTES_PERCENTAGE_ALL_ROOKIE"]

    assert len(df_val_all_nba) == len(
        y_val_all_nba
    ), "Mismatch in number of samples for ALL NBA validation data"
    assert len(df_val_all_rookie) == len(
        y_val_all_rookie
    ), "Mismatch in number of samples for ALL ROOKIE NBA validation data"

    assert (
        y_val_all_nba.notna().all()
    ), "Validation target variable for ALL NBA contains NaN values"
    assert (
        y_val_all_rookie.notna().all()
    ), "Validation target variable for ALL ROOKIE NBA contains NaN values"
    assert (
        y_val_all_nba > 0
    ).any(), "There must be at least one positive sample in ALL NBA validation data"
    assert (
        y_val_all_nba <= 1
    ).all(), "Validation target variable values for ALL NBA must be in the range [0, 1]"
    assert (
        y_val_all_rookie > 0
    ).any(), (
        "There must be at least one positive sample in ALL ROOKIE NBA validation data"
    )
    assert (
        y_val_all_rookie <= 1
    ).all(), "Validation target variable values for ALL ROOKIE NBA must be in the range [0, 1]"

    df_val_results_all_nba, score_all_nba = validate_season(
        nba_all_pipeline, df_val_all_nba, y_val_all_nba, is_rookie=False
    )
    season_scores_all_nba[2026] = score_all_nba

    df_val_results_all_rookie, score_all_rookie = validate_season(
        nba_rookie_pipeline, df_val_all_rookie, y_val_all_rookie, is_rookie=True
    )
    season_scores_all_rookie[2026] = score_all_rookie

    assert (
        df_val_results_all_rookie["actual_vote_share"].notna().all()
    ), "Validation results for ALL ROOKIE NBA contain NaN values in actual_vote_share"

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
        df_selected_players = selected_players[
            [
                "PLAYER_NAME",
                "actual_vote_share",
                "predicted_vote_share",
                "predicted_place",
                "actual_place",
            ]
        ]
        logger.info("\n" + df_selected_players.to_string(index=False))

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
        logger.info("\n" + df_selected_players.to_string(index=False))

    return ExperimentResults(
        season_scores_all_nba=season_scores_all_nba,
        season_scores_all_rookie=season_scores_all_rookie,
        selected_2026_players_all_nba=df_val_results_all_nba,
        selected_2026_players_all_rookie=df_val_results_all_rookie,
    )


if __name__ == "__main__":
    SEED = 0xC0FFEE

    config = ExperimentConfig(
        csv_path=Path("dataset/final_ml_dataset.csv"),
        seed=SEED,
        features_to_normalize=[
            "PTS",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TS_PCT",
            "TD3",
            "GP",
        ],
        features=[
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
        ],
        base_classifier=HistGradientBoostingClassifier(random_state=SEED),
        base_regressor=HistGradientBoostingRegressor(random_state=SEED),
        min_train_season=2020,
    )
    run_experiment(config)
