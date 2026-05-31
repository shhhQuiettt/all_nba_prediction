from nba_api.stats.endpoints import commonallplayers, playerawards, playercareerstats
import random
import logging
import pandas as pd
from tqdm import tqdm

import time
import os
from pathlib import Path

DATASET_DIR = Path("dataset")
PLAYERS_CSV = DATASET_DIR.joinpath("players.csv")
PLAYERS_STATS_CSV = DATASET_DIR.joinpath("players_stats.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if not DATASET_DIR.exists():
    os.makedirs(DATASET_DIR)

if not PLAYERS_CSV.exists():
    players = commonallplayers.CommonAllPlayers().get_data_frames()[0]
    players.to_csv(PLAYERS_CSV, index=False)
    logger.info(
        f"Fetched {len(players)} players from the API and saved to {PLAYERS_CSV}"
    )
else:
    logger.info(f"{PLAYERS_CSV} already exists. Skipping data retrieval.")
    players = pd.read_csv(PLAYERS_CSV)

player_stats_df = pd.DataFrame()
player_ids_to_fetch = players["PERSON_ID"].tolist()

if PLAYERS_STATS_CSV.exists():
    player_stats_df = pd.read_csv(PLAYERS_STATS_CSV)
    existing_player_ids = set(player_stats_df["PLAYER_ID"].unique())
    player_ids_to_fetch = [
        pid for pid in player_ids_to_fetch if pid not in existing_player_ids
    ]

    logger.info(
        f"Found existing player stats for {len(existing_player_ids)} players. Will fetch stats for {len(player_ids_to_fetch)} remaining players."
    )

for i, player_id in tqdm(enumerate(player_ids_to_fetch)):
    try:
        res = playercareerstats.PlayerCareerStats(player_id=player_id)
    except Exception as e:
        logger.error(f"Error fetching data for player ID {player_id}: {e}. Skipping.")

        time.sleep(10.1)
        continue

    if not res.get_data_frames():
        logger.error(f"Failed to fetch data for player ID {player_id}. Skipping.")
        time.sleep(10.1)
        continue
    stats = res.get_data_frames()[0]
    player_stats_df = pd.concat([player_stats_df, stats], ignore_index=True)
    player_stats_df.to_csv(PLAYERS_STATS_CSV, index=False)
    time.sleep(random.uniform(0.01, 0.1))

    tqdm.write(f"Fetched stats for {i} players so far. Total remaining players to fetch: {len(player_ids_to_fetch) - i - 1}")
    if (i + 1) % 100 == 0:
        time.sleep(1.1)

