from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from inference.inference import ExperimentConfig, run_experiment
import logging
import json
import argparse

RANDOM_SEED = 0xC0FFEE
BEST_CLASSIFIER = LogisticRegression(random_state=RANDOM_SEED, max_iter=3000)
BEST_REGRESSOR = GradientBoostingRegressor(random_state=0xC0FFEE)
BEST_MIN_TRAIN_SEASON = 2011


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run the final experiment for ALL-NBA and ALL-ROOKIE predictions."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Absolute path to save the final results JSON file.",
    )
    output_json_path = parser.parse_args().output_path

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
        base_classifier=BEST_CLASSIFIER,
        base_regressor=BEST_REGRESSOR,
        min_train_season=BEST_MIN_TRAIN_SEASON,
    )

    results = run_experiment(config, verbose=True)
    logger.info(f"ALL-NBA Rookie 2026 score: {results.season_scores_all_rookie[2026]}")
    logger.info(f"ALL-NBA 2026 score: {results.season_scores_all_nba[2026]}")
    logger.info(
        f"Total score: {results.season_scores_all_rookie[2026] + results.season_scores_all_nba[2026]}"
    )
    all_nba_team_1 = results.selected_2026_players_all_nba[
        results.selected_2026_players_all_nba["predicted_place"] == 1
    ]["PLAYER_NAME"].values.tolist()
    all_nba_team_2 = results.selected_2026_players_all_nba[
        results.selected_2026_players_all_nba["predicted_place"] == 2
    ]["PLAYER_NAME"].values.tolist()
    all_nba_team_3 = results.selected_2026_players_all_nba[
        results.selected_2026_players_all_nba["predicted_place"] == 3
    ]["PLAYER_NAME"].values.tolist()
    all_rookie_team_1 = results.selected_2026_players_all_rookie[
        results.selected_2026_players_all_rookie["predicted_place"] == 1
    ]["PLAYER_NAME"].values.tolist()
    all_rookie_team_2 = results.selected_2026_players_all_rookie[
        results.selected_2026_players_all_rookie["predicted_place"] == 2
    ]["PLAYER_NAME"].values.tolist()

    to_save = {
        "first all-nba team": all_nba_team_1,
        "second all-nba team": all_nba_team_2,
        "third all-nba team": all_nba_team_3,
        "first all-rookie team": all_rookie_team_1,
        "second all-rookie team": all_rookie_team_2,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved final results to {output_json_path}")


if __name__ == "__main__":
    main()
