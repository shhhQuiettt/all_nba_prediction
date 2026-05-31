import pandas as pd
import time
import random
import logging
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats, leaguestandings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_with_sleep(endpoint_class, **kwargs):
    """Helper function to fetch data and apply random sleep after the request."""
    response_df = endpoint_class(**kwargs).get_data_frames()[0]

    sleep_duration = random.uniform(3.6, 4.1)
    logger.info(f"API request successful. Sleeping for {sleep_duration:.2f} seconds...")
    time.sleep(sleep_duration)

    return response_df


def build_ml_dataset(season_string, raw_dir):
    logger.info(f"--- Processing Season: {season_string} ---")

    logger.info(f"Fetching Base Stats for {season_string}...")
    base = fetch_with_sleep(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season_string,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    )
    assert not base.empty, f"Base stats DataFrame for {season_string} is empty!"

    logger.info(f"Fetching Advanced Stats for {season_string}...")
    advanced = fetch_with_sleep(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season_string,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    assert (
        not advanced.empty
    ), f"Advanced stats DataFrame for {season_string} is empty! (Note: Advanced stats begin in 1996-97)"

    cols_to_use = advanced.columns.difference(base.columns).tolist()
    cols_to_use.append("PLAYER_ID")
    player_df = pd.merge(base, advanced[cols_to_use], on="PLAYER_ID", how="inner")
    assert not player_df.empty, f"Merged player DataFrame for {season_string} is empty!"

    logger.info(f"Fetching Team Standings for {season_string}...")
    standings = fetch_with_sleep(
        leaguestandings.LeagueStandings,
        season=season_string,
        season_type="Regular Season",
    )
    assert not standings.empty, f"Standings DataFrame for {season_string} is empty!"

    standings_subset = standings[
        ["TeamID", "Conference", "PlayoffRank", "WINS", "LOSSES", "WinPCT"]
    ].copy()
    standings_subset = standings_subset.rename(columns={"TeamID": "TEAM_ID"})

    final_df = pd.merge(player_df, standings_subset, on="TEAM_ID", how="inner")
    assert not final_df.empty, f"Final merged DataFrame for {season_string} is empty!"

    final_df["PlayoffRank"] = pd.to_numeric(
        final_df["PlayoffRank"], errors="coerce"
    ).astype(int)

    final_df.insert(0, "SEASON", season_string)

    file_path = raw_dir / f"{season_string}.csv"
    final_df.to_csv(file_path, index=False)
    logger.info(f"Successfully saved {season_string} to {file_path}")

    return final_df


def generate_season_strings(start_year, end_year):
    """Generates a list of NBA season strings (e.g., '1991-92')."""
    seasons = []
    for year in range(start_year, end_year + 1):
        next_year = str(year + 1)[-2:]
        seasons.append(f"{year}-{next_year}")
    return seasons


def main():
    dataset_dir = Path("dataset")
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate seasons from 1991 to 2025 (which outputs '1991-92' through '2025-26')
    seasons_to_fetch = generate_season_strings(1996, 2025)

    all_seasons_data = []

    for season in seasons_to_fetch:
        try:
            season_df = build_ml_dataset(season, raw_dir)
            all_seasons_data.append(season_df)
        except AssertionError as e:
            logger.error(f"Validation failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected API error occurred for {season}: {e}")

    if all_seasons_data:
        logger.info("Merging all successful season dataframes...")
        master_df = pd.concat(all_seasons_data, ignore_index=True)

        master_file_path = dataset_dir / "nba_master_dataset_1991_2026.csv"
        master_df.to_csv(master_file_path, index=False)
        logger.info(
            f"Pipeline Complete! Master dataset saved to: {master_file_path} with {len(master_df)} total rows."
        )
    else:
        logger.warning("No data was collected. Master CSV was not created.")


if __name__ == "__main__":
    main()
