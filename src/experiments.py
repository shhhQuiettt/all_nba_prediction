from inference.inference import ExperimentConfig, run_experiment
import numpy as np
import json
from tqdm import tqdm
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from itertools import product

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor


class DummyClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return np.array([[0.0, 1.0] for _ in range(len(X))])


EXPERIMENTS_OUTPUT_PATH = Path("results/experiments_results.json")

experiments_setups = {
    "min_train_season": range(1998, 2026),
    "clasifiers": [
        XGBClassifier(random_state=0xC0FFEE),
        HistGradientBoostingClassifier(random_state=0xC0FFEE),
        HistGradientBoostingClassifier(random_state=0xC0FFEE, class_weight="balanced"),
        LogisticRegression(random_state=0xC0FFEE, max_iter=3000),
        LogisticRegression(
            random_state=0xC0FFEE, class_weight="balanced", max_iter=3000
        ),
        RandomForestClassifier(random_state=0xC0FFEE),
        RandomForestClassifier(random_state=0xC0FFEE, class_weight="balanced"),
        DummyClassifier(),
    ],
    "regressors": [
        XGBRegressor(random_state=0xC0FFEE),
        HistGradientBoostingRegressor(random_state=0xC0FFEE),
        RandomForestRegressor(random_state=0xC0FFEE),
        GradientBoostingRegressor(random_state=0xC0FFEE),
        LinearRegression(),
    ],
}

experiments_results = []


def classifier_name(classifier) -> str:
    if hasattr(classifier, "class_weight") and classifier.class_weight is not None:
        return f"{classifier.__class__.__name__} (balanced)"
    return classifier.__class__.__name__


def regressor_name(regressor) -> str:
    return regressor.__class__.__name__


def main():
    EXPERIMENTS_OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    config = ExperimentConfig(
        csv_path=Path("dataset/final_ml_dataset.csv"),
        seed=0xC0FFEE,
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
        base_classifier=HistGradientBoostingClassifier(random_state=0xC0FFEE),
        base_regressor=HistGradientBoostingRegressor(random_state=0xC0FFEE),
        min_train_season=2020,
    )
    experiments_n = (
        len(experiments_setups["min_train_season"])
        * len(experiments_setups["clasifiers"])
        * len(experiments_setups["regressors"])
    )

    for min_year, classifier, regressor in tqdm(
        product(
            experiments_setups["min_train_season"],
            experiments_setups["clasifiers"],
            experiments_setups["regressors"],
        ),
        total=experiments_n,
    ):
        config.min_train_season = min_year
        config.base_classifier = classifier
        config.base_regressor = regressor

        results = run_experiment(config, verbose=False)
        experiments_results.append(
            {
                "min_train_season": min_year,
                "classifier": classifier_name(classifier),
                "regressor": regressor_name(regressor),
                "scores_all_rookie": results.season_scores_all_rookie,
                "scores_all_nba": results.season_scores_all_nba,
            }
        )

    with open(EXPERIMENTS_OUTPUT_PATH, "w") as f:
        json.dump(experiments_results, f, indent=4)


if __name__ == "__main__":
    main()
