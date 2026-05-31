import numpy as np
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




def validate_season(pipeline, X, y) -> tuple[pd.DataFrame, int]:
    assert "YEAR" not in X.columns
    predicted_vote_shares = pipeline.predict(X)

    df_res = pd.DataFrame({
        "predicted_vote_share": predicted_vote_shares,
        "actual_vote_share": y
    })

    df_res["predicted_team"] = df_res["predicted_vote_share"].rank(ascending=False, method="first").apply(assign_team)
    df_res["actual_team"] = df_res["actual_vote_share"].rank(ascending=False, method="first").apply(assign_team)

    score = calculate_score(df_res["predicted_team"], df_res["actual_team"])

    return df_res, score

def calculate_score(predicted_team: pd.Series, actual_team: pd.Series) -> int:
    # Analizujemy tylko zawodników, których wytypowaliśmy do jakiejkolwiek piątki
    df = pd.DataFrame({"pred": predicted_team, "act": actual_team})

    assert not df["pred"].isna().any()

    df = df[df["pred"] > 0]

    total_score = 0
    bonus_map = {1: 0, 2: 5, 3: 10, 4: 20, 5: 40}

    for team_val, group in df.groupby("pred"):
        exact_matches = group["act"] == group["pred"]

        off_by_1 = (group["act"] > 0) & (abs(group["act"] - group["pred"]) == 1)
        off_by_2 = (group["act"] > 0) & (abs(group["act"] - group["pred"]) == 2)

        team_score = (
            (exact_matches.sum() * 10) + (off_by_1.sum() * 8) + (off_by_2.sum() * 6)
        )
        total_score += team_score

        correct_count = exact_matches.sum()

        if correct_count >= 5:
            total_score += 40
        elif correct_count in bonus_map:
            total_score += bonus_map[correct_count]

    return int(total_score)


class SeasonZScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_normalize, season_column="YEAR"):
        self.columns_to_normalize = columns_to_normalize
        self.season_column = season_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clean = X.copy()
        for col in self.columns_to_normalize:
            X_clean[col] = X_clean.groupby(self.season_column)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        return X_clean


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


class HurdleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        y_arr = np.asarray(y)

        y_binary = (y_arr > 0).astype(int)
        self.clf_ = clone(self.classifier)
        self.clf_.fit(X, y_binary)

        mask = y_arr > 0
        self.reg_ = clone(self.regressor)

        assert np.sum(mask) > 0, "No positive samples (no votes)"

        X_voted = X.iloc[mask]
        self.reg_.fit(X_voted, y_arr[mask])

        return self

    def predict(self, X):
        probabilities = self.clf_.predict_proba(X)[:, 1]

        raw_predictions = self.reg_.predict(X)

        final_shares = probabilities * raw_predictions
        # final_shares = raw_predictions

        return np.clip(final_shares, 0, 1)


def assign_team(rank):
    if rank <= 5:
        return 1

    if rank <= 10:
        return 2

    if rank <= 15:
        return 3

    return 0


if __name__ == "__main__":
    np.random.seed(0xC0FFEE)
    csv_path = Path("dataset/final_ml_dataset.csv")
    df = pd.read_csv(csv_path)
    df_train = df[(df.YEAR >= 2024) & (df.YEAR < 2026)]
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
    assert set(features_to_normalize).issubset(set(features)), (
        "Normalization features must be in the feature list"
    )

    X_train = df_train[features + ["SEASON"]]
    y_train = df_train["VOTES_PERCENTAGE_ALL_NBA"]

    base_classifier = HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced")
    base_regressor = HistGradientBoostingRegressor(random_state=SEED)

    nba_award_pipeline = Pipeline(
        steps=[
            (
                "season_normalization",
                SeasonZScoreTransformer(
                    columns_to_normalize=features_to_normalize,
                    season_column="SEASON",
                ),
            ),
            (
                "drop_metadata",
                ColumnDropper(columns_to_drop=["SEASON"]),
            ),
            (
                "hurdle_model",
                HurdleRegressor(classifier=base_classifier, regressor=base_regressor),
            ),
        ]
    )

    nba_award_pipeline.fit(X_train, y_train)
    print("Pipeline successfully trained!")


    for year in sorted(df_train["SEASON"].unique()):
        season_mask = df_train["SEASON"] == year
        X_season = X_train[season_mask]
        y_season = y_train[season_mask]

        df_season_results, score = validate_season(nba_award_pipeline, X_season, y_season)

        logger.info(f"Season {year}: Score = {score}")

    

    X_val = df_val[features + ["SEASON"]]
    y_val = df_val["VOTES_PERCENTAGE_ALL_NBA"]

    df_val_results, score = validate_season(nba_award_pipeline, X_val, y_val)

    logger.info(f"Validation Season 2026: Score = {score}")

