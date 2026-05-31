from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


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

        assert np.sum(mask) > 0, (
            "There must be at least one positive sample to fit the regressor"
        )

        self.reg_.fit(X[mask], y_arr[mask])

        X_voted = X.iloc[mask]
        self.reg_.fit(X_voted, y_arr[mask])

        return self

    def predict(self, X):
        probabilities = self.clf_.predict_proba(X)[:, 1]

        raw_predictions = self.reg_.predict(X)

        final_shares = probabilities * raw_predictions
        # final_shares = raw_predictions

        return np.clip(final_shares, 0, 1)


def create_pipeline(base_classifier, base_regressor, features, features_to_normalize):
    assert set(features_to_normalize).issubset(set(features)), (
        "Normalization features must be in the feature list"
    )

    return Pipeline(
        steps=[
            (
                "select_features",
                ColumnSelector(
                    columns=features + ["SEASON"],
                ),
            ),
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
