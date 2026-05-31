import logging
from pathlib import Path
import re

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

TABLES_PATH = Path("./dataset/tables")
FINAL_CSV_PATH = Path("./dataset/awards.csv")


def validate_final_df(df: pd.DataFrame):
    # assert that all years from 1990 to 2026 are present
    expected_years = set(range(1990, 2027))
    actual_years = set(df["YEAR"].unique())
    assert (
        expected_years == actual_years
    ), f"Expected years {expected_years} but found {actual_years}"

    # assert all year have 1 2 3 1000 places for ALL-NBA and 1 2 3 for ALL-ROOKIE
    for year in expected_years:
        for award in ["ALL-NBA", "ALL-ROOKIE"]:
            df_year_award = df[(df["YEAR"] == year) & (df["AWARD"] == award)]
            if award == "ALL-NBA":
                expected_places = set([1, 2, 3, 1000])
            else:
                expected_places = set([1, 2, 1000])
            actual_places = set(df_year_award["PLACE"].unique())
            assert (
                expected_places == actual_places
            ), f"Expected places {expected_places} for year {year} and award {award} but found {actual_places}"

    # assert that every year-place combintation has more than 5 players, but less than 8 players
    for year in expected_years:
        for award in ["ALL-NBA", "ALL-ROOKIE"]:
            for place in [1, 2, 3]:
                if place == 3 and award == "ALL-ROOKIE":
                    continue

                df_year_award_place = df[
                    (df["YEAR"] == year)
                    & (df["AWARD"] == award)
                    & (df["PLACE"] == place)
                ]
                num_players = len(df_year_award_place)
                assert (
                    num_players >= 4 and num_players <= 8
                ), f"Expected between 4 and 8 players for year {year}, award {award}, place {place} but found {num_players}"

    # assert that there are no duplicate player IDs for the same year, award, place combination
    for year in expected_years:
        for award in ["ALL-NBA", "ALL-ROOKIE"]:
            for place in [1, 2, 3, 1000]:
                df_year_award_place = df[
                    (df["YEAR"] == year)
                    & (df["AWARD"] == award)
                    & (df["PLACE"] == place)
                ]
                player_ids = df_year_award_place["PLAYER_ID"].values
                assert len(player_ids) == len(
                    set(player_ids)
                ), f"Found duplicate player IDs for year {year}, award {award}, place {place}"


def find_player_id(name: str) -> int:
    players = pd.read_csv("./dataset/players.csv")
    # match similar name
    player_id = players[players["DISPLAY_FIRST_LAST"] == name]["PERSON_ID"].values
    if len(player_id) > 0:
        return player_id[0]

    name_cleaned = name.strip().upper()
    if name_cleaned.split()[-1] in ["I", "II", "III", "IV", "V", "SR."]:
        name_cleaned = " ".join(name_cleaned.split()[:-1])

    players["NAME_CLEANED"] = players["DISPLAY_FIRST_LAST"].str.strip().str.upper()
    players["NAME_CLEANED"] = players["NAME_CLEANED"].apply(
        lambda x: (
            " ".join(x.split()[:-1])
            if x.split()[-1] in ["I", "II", "III", "IV", "V", "SR."]
            else x
        )
    )
    player_id = players[players["NAME_CLEANED"] == name_cleaned]["PERSON_ID"].values
    if len(player_id) > 0:
        return player_id[0]

    name_regex = re.sub(r"[^A-Z0-9]", r".*", name_cleaned)
    player_id = players[players["NAME_CLEANED"].str.match(name_regex)][
        "PERSON_ID"
    ].values
    if len(player_id) > 0:
        return player_id[0]

    last_name = name_cleaned.split()[-1]

    player_id = players[players["NAME_CLEANED"].str.endswith(last_name)][
        "PERSON_ID"
    ].values
    if len(player_id) == 1:
        logger.warning(
            f"Matched player {name} to {players[players['PERSON_ID'] == player_id[0]]['DISPLAY_FIRST_LAST'].values[0]} using last name only"
        )
        return player_id[0]

    if len(player_id) > 1:
        # match with first 3 letters of first name
        first_name_prefix = name_cleaned.split()[0][:3]
        player_id = players[
            players["NAME_CLEANED"].str.endswith(last_name)
            & players["NAME_CLEANED"].str.startswith(first_name_prefix)
        ]["PERSON_ID"].values
        if len(player_id) == 1:
            logger.warning(
                f"Matched player {name} to {players[players['PERSON_ID'] == player_id[0]]['DISPLAY_FIRST_LAST'].values[0]} using last name and first name prefix"
            )
            return player_id[0]

        logger.error(
            f"Multiple players matched for last name {last_name} when trying to find player ID for {name}. Possible matches: {players[players['PERSON_ID'].isin(player_id)]['DISPLAY_FIRST_LAST'].values}"
        )
    else:
        logger.error(
            f"No players matched for last name {last_name} when trying to find player ID for {name}"
        )

    return -1


def process_raw_df(df_raw: pd.DataFrame, rookie: bool, year: int) -> pd.DataFrame:
    df_raw.columns = df_raw.columns.droplevel(0)
    place_col_id = "# Tm"
    place_possibility = ["1st", "2nd", "3rd", "ORV", "1T", "2T", "3T"]
    share_col_id = "Share"
    name_col_id = "Player"

    place_map = {
        "1st": 1,
        "2nd": 2,
        "3rd": 3,
        "ORV": 1000,
        "1T": 1,
        "2T": 2,
        "3T": 3,
    }

    df = pd.DataFrame(
        columns=[
            "YEAR",
            "AWARD",
            "PLACE",
            "PLAYER_ID",
            "DISPLAY_FIRST_LAST",
            "VOTES_PERCENTAGE",
        ]
    )

    for _, row in df_raw.iterrows():
        place = row[place_col_id]
        if place not in place_possibility:
            continue

        name = row[name_col_id]
        try:
            player_id = find_player_id(name)
        except ValueError as e:
            logger.error(f"Error finding player ID for {name} in year {year}")
            logger.error(f"Row data: {row}")
            raise e

        votes_percentage = row[share_col_id]
        award_type = "ALL-ROOKIE" if rookie else "ALL-NBA"
        place_numeric = place_map[place]
        df.loc[len(df)] = [
            year,
            award_type,
            place_numeric,
            player_id,
            name,
            votes_percentage,
        ]

    return df


def main():
    years = list(range(1990, 2027))

    for year in years:
        html_path_normal = TABLES_PATH.joinpath(f"all_nba_{year}.html")
        html_path_rookie = TABLES_PATH.joinpath(f"all_nba_{year}_rookie.html")

        if html_path_normal.exists():
            csv_path_normal = TABLES_PATH.joinpath(f"all_nba_{year}.csv")
            if csv_path_normal.exists():
                continue

            df_old_raw = pd.read_html(html_path_normal, encoding="utf-8")[0]
            df = process_raw_df(df_old_raw, rookie=False, year=year)
            df.to_csv(csv_path_normal, index=False)
        else:
            logger.info(
                f"HTML file {html_path_normal} for ALL-NBA year {year} does not exist. Skipping."
            )

        if html_path_rookie.exists():
            csv_path_rookie = TABLES_PATH.joinpath(f"all_nba_{year}_rookie.csv")
            if csv_path_rookie.exists():
                continue
            df_rookie_raw = pd.read_html(html_path_rookie)[0]
            df = process_raw_df(df_rookie_raw, rookie=True, year=year)
            df.to_csv(csv_path_rookie, index=False)

        else:
            logger.info(
                f"HTML file {html_path_rookie} for ALL-ROOKIE year {year} does not exist. Skipping."
            )

    processed_files = list(TABLES_PATH.glob("all_nba_*.csv"))
    logger.info(
        f"Found {len(processed_files)} processed CSV files. Combining into final CSV."
    )
    df_final = pd.DataFrame(
        columns=[
            "YEAR",
            "AWARD",
            "PLACE",
            "PLAYER_ID",
            "DISPLAY_FIRST_LAST",
            "VOTES_PERCENTAGE",
        ]
    )
    for file in processed_files:
        df = pd.read_csv(file)
        df_final = pd.concat([df_final, df], ignore_index=True)

    validate_final_df(df_final)
    df_final.to_csv(FINAL_CSV_PATH, index=False)
    logger.info(f"Final CSV file saved to {FINAL_CSV_PATH}")


if __name__ == "__main__":
    main()
